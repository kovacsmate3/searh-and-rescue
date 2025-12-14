"""
Main entry point for the Search & Rescue project.

This script uses Hydra to load configuration files and dispatches to training
or evaluation routines based on config flags. Use `train.active=true` or
`eval.active=true` on the command line to select the desired mode.
"""

import hydra
from omegaconf import DictConfig

from train import run_training
from evaluate import run_evaluation


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.train.active:
        run_training(cfg)
    if cfg.eval.active:
        run_evaluation(cfg)


if __name__ == "__main__":
    main()
