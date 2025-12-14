"""
Training entry point for the Search & Rescue re‑implementation.

This script uses Hydra for configuration and instantiates a multi‑agent PettingZoo
environment. It constructs a centralized‑critic, decentralized‑actor (CTDE) PPO
algorithm using TorchRL. The current implementation is a skeleton that you
should extend with the full environment and training logic.
"""

import hydra
from omegaconf import DictConfig
import torch

from torch import nn
from torch.optim import Adam
from torchrl.envs import ParallelEnv
from torchrl.envs.utils import check_env_specs
from torchrl.data.replay_buffers import ReplayBuffer, LazyMemmapStorage
from torchrl.data import TensorSpec

try:
    from pettingzoo.mpe import simple_spread_v3
except ImportError:
    simple_spread_v3 = None

# Import our custom Search & Rescue environment
try:
    from .env import SearchRescueEnv
except Exception:
    SearchRescueEnv = None

# Scenario generator for domain randomization and curriculum
try:
    from .scenarios import ScenarioGenerator
except Exception:
    ScenarioGenerator = None


def make_env(env_cfg: DictConfig) -> ParallelEnv:
    """Create the Search & Rescue environment or fallback to simple_spread_v3.

    If the custom environment is available, instantiate it directly. Otherwise
    fallback to the PettingZoo simple_spread_v3 environment. The returned
    environment must implement the ParallelEnv interface for TorchRL.
    """
    # Prefer custom SearchRescueEnv if available
    if SearchRescueEnv is not None:
        return SearchRescueEnv(
            num_rescuers=env_cfg.num_rescuers,
            num_victims=env_cfg.get("num_victims", 0),
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
    # Fallback if pettingzoo is installed
    if simple_spread_v3 is None:
        raise ImportError(
            "Neither custom SearchRescueEnv nor simple_spread_v3 is available."
        )
    env = simple_spread_v3.parallel_env(
        N=env_cfg.num_rescuers,
        local_ratio=0.5,
        max_cycles=env_cfg.max_cycles,
        continuous_actions=env_cfg.continuous_actions,
    )
    env.seed(env_cfg.seed)
    # Wrap environment into TorchRL wrapper at runtime to provide observation/action specs
    from torchrl.envs import PettingZooWrapper
    return PettingZooWrapper(env)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Merge env/algo config from Hydra; Hydra will automatically compose from defaults
    env_cfg = cfg.env if "env" in cfg else {}

    # If scenario config provided, sample random environment parameters or use curriculum
    if "scenario" in cfg and ScenarioGenerator is not None:
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
        # Sample scenario
        if scen_cfg.curriculum_steps > 0:
            params = next(gen.curriculum())
        else:
            params = gen.sample()
        # Override environment parameters with sampled scenario
        env_cfg.num_victims = params.num_victims
        env_cfg.num_trees = params.num_trees
        env_cfg.num_safezones = params.num_safezones
        # Currently map_scale is unused in the environment but kept for future extension

    # Instantiate environment
    env = make_env(env_cfg)
    # Validate specs
    check_env_specs(env)

    # Print out observation and action specs
    print("Observation spec:", env.observation_spec())
    print("Action spec:", env.action_spec())

    # Define actor and critic networks
    n_agents = len(env.possible_agents)
    obs_dim = env.observation_spec().shape[-1]
    # Discrete action dimension or continuous size
    if hasattr(env.action_spec(), "n"):
        action_dim = env.action_spec().n  # discrete actions
    else:
        action_dim = env.action_spec().shape[-1]

    class PolicyNet(nn.Module):
        def __init__(self, input_dim: int, output_dim: int):
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
        def __init__(self, input_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self.net(obs)

    # Instantiate networks
    actor_net = PolicyNet(obs_dim, action_dim)
    # Critic sees global state (concatenated observations)
    critic_net = CriticNet(obs_dim * n_agents)

    actor_opt = Adam(actor_net.parameters(), lr=cfg.algo.learning_rate)
    critic_opt = Adam(critic_net.parameters(), lr=cfg.algo.learning_rate)

    # Start PPO training
    train_ppo(
        env=env,
        actor=actor_net,
        critic=critic_net,
        actor_opt=actor_opt,
        critic_opt=critic_opt,
        n_agents=n_agents,
        num_steps=cfg.algo.num_steps,
        num_epochs=cfg.algo.num_epochs,
        gamma=cfg.algo.gamma,
        clip_param=cfg.algo.clip_param,
    )


if __name__ == "__main__":
    main()


# === PPO TRAINING LOOP ===
import math
from torch.distributions import Categorical, Normal
from typing import List

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
    """Simplified PPO training loop for multi‑agent CTDE.

    Args:
        env: multi‑agent environment implementing ParallelEnv.
        actor: policy network that maps observations to action logits (discrete).
        critic: value network that maps concatenated observations to scalar value.
        actor_opt: optimizer for actor.
        critic_opt: optimizer for critic.
        n_agents: number of agents in the environment.
        num_steps: number of steps to collect per update.
        num_epochs: number of gradient epochs per update.
        gamma: discount factor.
        clip_param: PPO clipping epsilon.
    """
    actor.train()
    critic.train()

    for update in range(10):  # arbitrary number of updates; adjust as needed
        # Storage lists
        obs_buffer: List[torch.Tensor] = []
        actions_buffer: List[torch.Tensor] = []
        logprobs_buffer: List[torch.Tensor] = []
        rewards_buffer: List[float] = []
        values_buffer: List[torch.Tensor] = []

        # Reset env
        obs_dict = env.reset()
        # Flatten initial observations into per‑agent tensors
        obs = torch.stack(
            [torch.tensor(obs_dict[agent], dtype=torch.float32) for agent in env.possible_agents]
        )
        for step in range(num_steps):
            # Select actions per agent
            logits = actor(obs)  # shape (n_agents, action_dim)
            # Categorical distribution for each agent
            dist = Categorical(logits=logits)
            actions = dist.sample()  # shape (n_agents,)
            logprobs = dist.log_prob(actions)
            # Convert to environment action dict
            actions_dict = {
                agent: actions[i].item() for i, agent in enumerate(env.possible_agents)
            }
            # Step environment
            next_obs_dict, reward_dict, terminations, truncations, infos = env.step(actions_dict)
            # Team reward: sum across agents
            reward = sum(reward_dict.values())
            # Store data
            obs_buffer.append(obs)
            actions_buffer.append(actions)
            logprobs_buffer.append(logprobs)
            rewards_buffer.append(reward)
            # Compute value estimate using global state
            global_state = obs.view(1, -1).reshape(-1)
            value = critic(global_state)
            values_buffer.append(value.squeeze())
            # Prepare next obs
            obs = torch.stack(
                [torch.tensor(next_obs_dict[agent], dtype=torch.float32) for agent in env.possible_agents]
            )
            done = all(terminations.values()) or all(truncations.values())
            if done:
                break
        # Convert lists to tensors
        rewards_tensor = torch.tensor(rewards_buffer, dtype=torch.float32)
        values_tensor = torch.stack(values_buffer)
        logprobs_tensor = torch.stack(logprobs_buffer)
        # Compute returns and advantages
        returns = compute_returns(rewards_tensor, gamma)
        advantages = returns - values_tensor.detach()
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Flatten actor observations and actions for PPO update
        actor_obs = torch.cat([o for o in obs_buffer])
        actor_actions = torch.cat([a for a in actions_buffer])
        old_logprobs = logprobs_tensor.detach().flatten()
        # Update policy and value networks
        for epoch in range(num_epochs):
            # Forward pass through actor
            logits = actor(actor_obs)
            dist = Categorical(logits=logits)
            new_logprobs = dist.log_prob(actor_actions)
            entropy = dist.entropy().mean()
            # Ratio for PPO
            ratio = (new_logprobs - old_logprobs).exp()
            # Surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            # Critic loss
            # Recompute values for critic on stored global states
            global_states = actor_obs.view(-1, obs.shape[-1]).reshape(-1)
            values_pred = critic(global_states)
            value_loss = (returns - values_pred.squeeze()).pow(2).mean()
            # Update actor
            actor_opt.zero_grad()
            policy_loss.backward(retain_graph=True)
            actor_opt.step()
            # Update critic
            critic_opt.zero_grad()
            value_loss.backward()
            critic_opt.step()
        print(f"Update {update}: policy loss {policy_loss.item():.3f}, value loss {value_loss.item():.3f}")
