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

The requirements pin `setuptools<81` to avoid the `pkg_resources` deprecation warning currently emitted by pygame wheels; upgrade once pygame drops that dependency.

### Quick start

With the default Hydra config (`train.active=false`, `eval.active=true`, `eval.render=true`), running the main entry point evaluates a random policy while opening the pygame viewer:

```powershell
python main.py
```

To suppress visualization when running on a headless machine, add `eval.render=false` to the command line. Likewise, toggle between modes via `train.active=true` or `eval.active=true`.

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

Set `eval.render=true` (default when using `main.py`) if you also want the pygame window while running `evaluate.py` directly.

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

## Visualization

To watch the environment in action, install `pygame` (already listed in `requirements.txt`) and run:

```powershell
cd C:\_git\elte-ik-msc\COLLIEG\assignment2\searh-and-rescue
python - <<"PY"
from env import SearchRescueEnv
from visualizer import run_random_episode

env = SearchRescueEnv()
run_random_episode(env)
PY
```

This opens a pygame window showing rescuers (blue), victims (red), trees (green), and safe zones (yellow outlines) while a random policy acts in the environment. Close the window or press Ctrl+C in the terminal to stop.
