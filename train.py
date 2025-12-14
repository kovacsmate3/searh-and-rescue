"""
Training logic for the Search & Rescue project.

This module defines functions to train a multi‑agent PPO agent with a
centralized critic and decentralized actors. The `run_training` function
expects a Hydra configuration object and sets up the environment, actor and
critic networks, and optimizers. The `train_ppo` function implements a
simplified PPO update loop.
"""

from __future__ import annotations

import math
from typing import List

import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical

from pettingzoo import ParallelEnv
from torchrl.envs.utils import check_env_specs

from env import SearchRescueEnv
from scenarios import ScenarioGenerator


def run_training(cfg) -> None:
    """Main entry point for training given a Hydra config."""
    # Unpack environment and scenario configs
    env_cfg = cfg.env
    # Apply scenario generator if provided
    if hasattr(cfg, "scenario") and cfg.scenario is not None:
        scen_cfg = cfg.scenario
        gen = ScenarioGenerator(
            num_rescuers=env_cfg.num_rescuers,
            victim_range=tuple(scen_cfg.victim_range),
            tree_range=tuple(scen_cfg.tree_range),
            safezone_range=tuple(scen_cfg.safezone_range),
            map_scale_range=tuple(scen_cfg.map_scale_range),
            curriculum_steps=scen_cfg.curriculum_steps,
            seed=scen_cfg.seed,
        )
        params = gen.sample() if scen_cfg.curriculum_steps <= 0 else next(gen.curriculum())
        env_cfg.num_victims = params.num_victims
        env_cfg.num_trees = params.num_trees
        env_cfg.num_safezones = params.num_safezones

    # Instantiate environment
    env_kwargs = dict(
        num_rescuers=env_cfg.num_rescuers,
        num_victims=env_cfg.num_victims,
        num_trees=env_cfg.num_trees,
        num_safezones=env_cfg.num_safezones,
        max_cycles=env_cfg.max_cycles,
        vision_radius=env_cfg.vision_radius,
        continuous_actions=env_cfg.continuous_actions,
        collision_penalty=env_cfg.collision_penalty,
        boundary_penalty=env_cfg.boundary_penalty,
        capture_radius=env_cfg.capture_radius,
        safezone_radius=env_cfg.safezone_radius,
        seed=env_cfg.seed,
    )
    from torchrl.envs import PettingZooWrapper
    spec_env = PettingZooWrapper(SearchRescueEnv(**env_kwargs))
    check_env_specs(spec_env)
    spec_env.close()
    env: ParallelEnv = SearchRescueEnv(**env_kwargs)

    # Define actor and critic networks
    n_agents = len(env.possible_agents)
    obs_space = env.single_observation_space()
    obs_dim = int(math.prod(obs_space.shape)) if obs_space.shape else 1
    action_space = env.single_action_space()
    if hasattr(action_space, "n"):
        action_dim = action_space.n
    else:
        action_dim = int(math.prod(action_space.shape)) if action_space.shape else 1

    class PolicyNet(nn.Module):
        def __init__(self, input_dim: int, output_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
            )

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self.net(obs)

    class CriticNet(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self.net(obs)

    actor = PolicyNet(obs_dim, action_dim)
    critic = CriticNet(obs_dim * n_agents)
    actor_opt = Adam(actor.parameters(), lr=cfg.algo.learning_rate)
    critic_opt = Adam(critic.parameters(), lr=cfg.algo.learning_rate)

    # Train with PPO
    train_ppo(
        env=env,
        actor=actor,
        critic=critic,
        actor_opt=actor_opt,
        critic_opt=critic_opt,
        n_agents=n_agents,
        num_steps=cfg.algo.num_steps,
        num_epochs=cfg.algo.num_epochs,
        gamma=cfg.algo.gamma,
        clip_param=cfg.algo.clip_param,
    )


def compute_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute discounted returns for a 1D tensor of rewards."""
    returns = torch.zeros_like(rewards)
    running_return = 0.0
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    return returns


def train_ppo(
    env: ParallelEnv,
    actor: nn.Module,
    critic: nn.Module,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    n_agents: int,
    num_steps: int,
    num_epochs: int,
    gamma: float,
    clip_param: float,
) -> None:
    """Simplified PPO training loop for multi‑agent CTDE."""
    actor.train()
    critic.train()
    for update in range(10):
        obs_buffer: List[torch.Tensor] = []
        actions_buffer: List[torch.Tensor] = []
        logprobs_buffer: List[torch.Tensor] = []
        rewards_buffer: List[float] = []
        values_buffer: List[torch.Tensor] = []
        global_obs_buffer: List[torch.Tensor] = []
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs_dict, _info_dict = reset_out
        else:
            obs_dict = reset_out
        obs = torch.stack([
            torch.tensor(obs_dict[agent], dtype=torch.float32)
            for agent in env.possible_agents
        ])
        for step in range(num_steps):
            logits = actor(obs)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            logprobs = dist.log_prob(actions)
            actions_dict = {agent: actions[i].item() for i, agent in enumerate(env.possible_agents)}
            next_obs_dict, reward_dict, terminations, truncations, *_ = env.step(actions_dict)
            reward = sum(reward_dict.values())
            obs_buffer.append(obs)
            actions_buffer.append(actions)
            logprobs_buffer.append(logprobs)
            global_state = obs.view(-1)
            values_buffer.append(critic(global_state).squeeze())
            global_obs_buffer.append(global_state.clone())
            rewards_buffer.append(reward)
            obs = torch.stack([
                torch.tensor(next_obs_dict[agent], dtype=torch.float32)
                for agent in env.possible_agents
            ])
            if all(terminations.values()) or all(truncations.values()):
                break
        if not rewards_buffer:
            continue
        rewards_tensor = torch.tensor(rewards_buffer, dtype=torch.float32)
        values_tensor = torch.stack(values_buffer)
        logprobs_tensor = torch.stack(logprobs_buffer)
        global_obs = torch.stack(global_obs_buffer).to(torch.float32)
        returns = compute_returns(rewards_tensor, gamma)
        advantages = returns - values_tensor.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages_expanded = advantages.repeat_interleave(n_agents)
        actor_obs = torch.cat(obs_buffer)
        actor_actions = torch.cat(actions_buffer)
        old_logprobs = logprobs_tensor.detach().flatten()
        for epoch in range(num_epochs):
            logits = actor(actor_obs)
            dist = Categorical(logits=logits)
            new_logprobs = dist.log_prob(actor_actions)
            entropy = dist.entropy().mean()
            ratio = (new_logprobs - old_logprobs).exp()
            surr1 = ratio * advantages_expanded
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages_expanded
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            values_pred = critic(global_obs).squeeze(-1)
            value_loss = (returns - values_pred).pow(2).mean()
            actor_opt.zero_grad()
            policy_loss.backward(retain_graph=True)
            actor_opt.step()
            critic_opt.zero_grad()
            value_loss.backward()
            critic_opt.step()
        print(f"Update {update}: policy loss {policy_loss.item():.3f}, value loss {value_loss.item():.3f}")
