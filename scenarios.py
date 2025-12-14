"""
Scenario generation utilities for the Search & Rescue project.

This module provides a `ScenarioGenerator` class that can sample environment
parameters for domain randomization and produce a curriculum over occluder
density or map size. It abstracts away the logic of selecting the number of
trees, safe zones, victims and map scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np


@dataclass
class ScenarioParams:
    num_rescuers: int
    num_victims: int
    num_trees: int
    num_safezones: int
    map_scale: float


class ScenarioGenerator:
    def __init__(
        self,
        num_rescuers: int,
        victim_range: Tuple[int, int] = (1, 3),
        tree_range: Tuple[int, int] = (2, 6),
        safezone_range: Tuple[int, int] = (2, 4),
        map_scale_range: Tuple[float, float] = (1.0, 1.0),
        curriculum_steps: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize a scenario generator.

        Args:
            num_rescuers: number of rescuers (fixed).
            victim_range: (min, max) victims to randomize.
            tree_range: (min, max) trees to randomize (controls occluder density).
            safezone_range: (min, max) safe zones to randomize.
            map_scale_range: (min, max) scaling factor of the world size.
            curriculum_steps: if >0, generate a curriculum over tree density.
            seed: random seed for reproducibility.
        """
        self.num_rescuers = num_rescuers
        self.victim_range = victim_range
        self.tree_range = tree_range
        self.safezone_range = safezone_range
        self.map_scale_range = map_scale_range
        self.curriculum_steps = curriculum_steps
        self.rng = np.random.default_rng(seed)

    def sample(self) -> ScenarioParams:
        """Sample a random scenario within specified ranges."""
        num_victims = self.rng.integers(*self.victim_range)
        num_trees = self.rng.integers(*self.tree_range)
        num_safezones = self.rng.integers(*self.safezone_range)
        map_scale = float(
            self.rng.uniform(self.map_scale_range[0], self.map_scale_range[1])
        )
        return ScenarioParams(
            num_rescuers=self.num_rescuers,
            num_victims=int(num_victims),
            num_trees=int(num_trees),
            num_safezones=int(num_safezones),
            map_scale=map_scale,
        )

    def curriculum(self) -> Iterator[ScenarioParams]:
        """Yield a curriculum of scenarios with increasing tree density."""
        if self.curriculum_steps <= 0:
            # fallback to infinite sampling
            while True:
                yield self.sample()
        else:
            min_trees, max_trees = self.tree_range
            tree_values = np.linspace(min_trees, max_trees, self.curriculum_steps, dtype=int)
            for t in tree_values:
                params = ScenarioParams(
                    num_rescuers=self.num_rescuers,
                    num_victims=self.victim_range[0],
                    num_trees=int(t),
                    num_safezones=self.safezone_range[0],
                    map_scale=self.map_scale_range[0],
                )
                yield params
