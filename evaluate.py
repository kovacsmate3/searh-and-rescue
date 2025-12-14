"""
Evaluation logic for the Search & Rescue project.

This module defines a `run_evaluation` function that executes evaluation
episodes using the configured environment and a dummy random policy, then
computes metrics using the provided metrics functions.
"""

from __future__ import annotations

from typing import List, Dict

import hydra
from omegaconf import DictConfig

from env import SearchRescueEnv
from scenarios import ScenarioGenerator
from metrics import rescues_completed, collision_count, coverage


def run_evaluation(cfg: DictConfig) -> None:
    env_cfg = cfg.env
    # Apply scenario sampling
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

    raw_env = SearchRescueEnv(
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
    # Wrap with TorchRL PettingZooWrapper to conform to TorchRL API (optional for evaluation)
    try:
        from torchrl.envs import PettingZooWrapper
        env = PettingZooWrapper(raw_env)
    except Exception:
        env = raw_env
    num_episodes = cfg.eval.num_episodes
    episodes_logs: List[Dict[str, float]] = []
    for ep_idx in range(num_episodes):
        obs = env.reset(seed=env_cfg.seed + ep_idx)
        done = False
        episode_log = {
            "num_victims": env.num_victims,
            "rescued_victims": 0,
            "collisions": 0,
            "coverage": 0,
        }
        visited_cells = set()
        while not done:
            actions = {agent: env.action_spaces()[agent].sample() for agent in env.possible_agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)
            # Count collisions via negative rewards
            for reward in rewards.values():
                if reward < 0:
                    episode_log["collisions"] += 1
            for pos in env.rescuer_pos:
                cell = tuple((pos / 0.1).astype(int))
                visited_cells.add(cell)
            done = all(terminations.values()) or all(truncations.values())
        episode_log["rescued_victims"] = int(env.victim_rescued.sum())
        episode_log["coverage"] = len(visited_cells)
        episodes_logs.append(episode_log)
    rc = rescues_completed(episodes_logs)
    cc = collision_count(episodes_logs)
    cov = coverage(episodes_logs)
    print("Evaluation results over", num_episodes, "episodes:")
    print(f"Rescues completed: {rc:.2f}")
    print(f"Average collisions: {cc:.2f}")
    print(f"Coverage: {cov:.2f}")
