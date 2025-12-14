# Search & Rescue (Modernized)

This repository contains a re‑implementation of the Search & Rescue multi‑agent
reinforcement learning task using **TorchRL** with a centralized critic and
decentralized actors (CTDE). The goal is to escort victims to matching safe
zones under partial observability and occlusions.

## Features

* **TorchRL + CTDE**: Actors only see local observations while a centralized
  critic can access global information. MAPPO support via TorchRL.
* **Custom Environment**: Implements a 2D world with rescuer agents, victims,
  trees (occluders) and safe zones. Vision radius, occlusion logic,
  continuous/discrete control, victim dynamics and collision/boundary penalties
  are configurable.
* **Scenario Generator & Curriculum**: Domain randomization over number of
  victims, trees and safe zones with optional curricula over tree density.
* **Hydra Configs**: Separate configurations for environment, algorithm,
  evaluation and scenario. Override any parameter via CLI.
* **Evaluation & Metrics**: Scripts to evaluate trained policies and compute
  metrics such as rescues completed, collision count and coverage.
* **Unit Tests**: Basic tests for environment reset/step and occlusion logic.
* **Docker Support**: Build a reproducible Docker image with the provided
  `Dockerfile` and run experiments via `docker run`.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

### Training

Run training with default configuration:

```bash
python train.py
```

Override parameters via Hydra:

```bash
python train.py env.num_rescuers=3 env.continuous_actions=true
```

### Evaluation

Evaluate a (dummy) random policy over five episodes and report metrics:

```bash
python evaluate.py eval.num_episodes=5
```

### Docker

Build the Docker image:

```bash
docker build -t search-rescue .
```

Run training inside Docker:

```bash
docker run --rm search-rescue python train.py
```

## Metrics

Three metrics are implemented in `metrics.py` and used by `evaluate.py`:

1. **Rescues completed**: fraction of victims successfully escorted to safe zones.
2. **Collision count**: average number of tree or boundary collisions per episode.
3. **Coverage**: average number of unique grid cells visited by agents (proxy for exploration).

## Ablations

Suggested ablation studies include:

1. **Discrete vs continuous control**: set `env.continuous_actions` to `false` or `true`.
2. **Occlusion logic strict vs lenient**: vary `env.vision_radius` and adjust or disable the occlusion test in `env.py`.

Configure ablations via Hydra and keep seeds fixed for fair comparison.

## Tests

Run the unit tests with pytest:

```bash
pytest tests
```

The tests check correct operation of environment reset/step and the occlusion logic.

## Reproducibility

All experiments are driven by Hydra configuration files under `configs/` and can
be reproduced with fixed seeds. The `Dockerfile` provides an isolated
environment for consistent results across systems.
