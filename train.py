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
from torchrl.envs import PettingZooWrapper, ParallelEnv
from torchrl.envs.utils import check_env_specs
from torchrl.data.replay_buffers import ReplayBuffer, LazyMemmapStorage
from torchrl.data import TensorSpec

try:
    from pettingzoo.mpe import simple_spread_v3
except ImportError:
    simple_spread_v3 = None


def make_env(env_cfg: DictConfig) -> ParallelEnv:
    """Create a PettingZoo environment wrapped for TorchRL.

    In the final project you should implement the Search and Rescue
    environment with proper vision and occlusion logic. Here we fallback to
    simple_spread_v3 as a placeholder.
    """
    if simple_spread_v3 is None:
        raise ImportError("pettingzoo is not installed. Please install pettingzoo to run training.")
    env = simple_spread_v3.parallel_env(
        N=env_cfg.num_rescuers,
        local_ratio=0.5,
        max_cycles=env_cfg.max_cycles,
        continuous_actions=env_cfg.continuous_actions,
    )
    env.seed(env_cfg.seed)
    return PettingZooWrapper(env)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Merge env/algo config from Hydra; Hydra will automatically compose from defaults
    env_cfg = cfg.env if "env" in cfg else {}  # fallback if env subconfig not present

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
