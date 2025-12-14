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

    # Print out specs for debugging
    print("Observation spec:", env.observation_spec())
    print("Action spec:", env.action_spec())

    # TODO: define actor and critic networks. For CTDE, critic sees concatenated
    # observations of all agents. The actor is per‑agent.
    n_agents = len(env.possible_agents)

    # Example network definitions (to be customized):
    input_dims = env.observation_spec().shape[-1]
    action_dims = env.action_spec().shape[-1]

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

    # Instantiate networks (placeholder sizes)
    actor_net = PolicyNet(input_dims, action_dims)
    critic_net = CriticNet(input_dims * n_agents)

    # Optimizers
    actor_opt = Adam(actor_net.parameters(), lr=cfg.algo.learning_rate)
    critic_opt = Adam(critic_net.parameters(), lr=cfg.algo.learning_rate)

    # TODO: Set up replay buffer and the PPO training loop.
    print("Initialized actor and critic networks. Ready to implement PPO training loop.")


if __name__ == "__main__":
    main()
