"""
Evaluation script for the Search & Rescue project.

This script uses Hydra to load configuration, instantiates the environment
and a (dummy) agent policy, runs a specified number of evaluation episodes,
collects metrics (rescues completed, collision count, coverage) and prints
a summary. Extend this script by loading a trained policy and collecting
detailed logs as needed.
"""

import hydra
from omegaconf import DictConfig

from .env import SearchRescueEnv
from .scenarios import ScenarioGenerator
from .metrics import rescues_completed, collision_count, coverage


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    env_cfg = cfg.env
    # Sample scenario for evaluation
    if "scenario" in cfg:
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
        params = gen.sample()
        env_cfg.num_victims = params.num_victims
        env_cfg.num_trees = params.num_trees
        env_cfg.num_safezones = params.num_safezones

    env = SearchRescueEnv(
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

    num_episodes = cfg.eval.num_episodes
    episodes_logs = []
    for ep_idx in range(num_episodes):
        obs = env.reset(seed=cfg.env.seed + ep_idx)
        done = False
        episode_log = {
            "num_victims": env.num_victims,
            "rescued_victims": 0,
            "collisions": 0,
            "coverage": 0,
        }
        visited_positions = set()
        # Dummy random policy: choose random action per agent
        while not done:
            actions = {agent: env.action_spaces()[agent].sample() for agent in env.possible_agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)
            # Log collisions via negative rewards for collision (approximate)
            for agent, reward in rewards.items():
                # If reward is negative due to collision or boundary, count as collision
                if reward < 0:
                    episode_log["collisions"] += 1
            # Track coverage: discretize positions into 0.1 grid
            for pos in env.rescuer_pos:
                cell = tuple((pos / 0.1).astype(int))
                visited_positions.add(cell)
            # Check termination/truncation
            done = all(terminations.values()) or all(truncations.values())
        # Count rescued victims
        episode_log["rescued_victims"] = int(env.victim_rescued.sum())
        episode_log["coverage"] = len(visited_positions)
        episodes_logs.append(episode_log)

    # Compute metrics
    rc = rescues_completed(episodes_logs)
    cc = collision_count(episodes_logs)
    cov = coverage(episodes_logs)
    print("Evaluation results over", num_episodes, "episodes:")
    print(f"Rescues completed: {rc:.2f}")
    print(f"Average collisions: {cc:.2f}")
    print(f"Coverage (unique cells visited): {cov:.2f}")


if __name__ == "__main__":
    main()
