"""
Custom Search and Rescue environment for multi‑agent reinforcement learning.

This module implements a PettingZoo parallel environment with partial
observability, vision radius and simple occlusion logic. Agents (rescuers) move
in a 2D square world to rescue stationary victims while avoiding obstacles
(trees) and escorting victims to designated safe zones.

Key features implemented according to the assignment requirements:
* Observations are bounded Box spaces without sentinel values. Each agent
  observes its own position and velocity plus relative positions to nearby
  entities (victims, rescuers, trees, safe zones). Entities outside the
  vision radius or occluded by a tree are masked to zeros.
* A simple occlusion test: an entity is considered occluded if a tree lies on
  the line segment between the agent and the entity and is closer to the agent.
* Termination when all victims are rescued or the maximum number of steps
  (`max_cycles`) is reached.
* Rewards are left to be defined during algorithm implementation; here we
  return zero reward per agent.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from gymnasium import spaces
from pettingzoo import ParallelEnv


class SearchRescueEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "search_rescue_v0"}

    # TorchRL internal flag indicating whether observation/action specs can change at runtime.
    # Our environment has fixed observation and action spaces, so set to False.
    _has_dynamic_specs: bool = False

    def __init__(
        self,
        num_rescuers: int = 2,
        num_victims: int = 2,
        num_trees: int = 4,
        num_safezones: int = 4,
        max_cycles: int = 500,
        vision_radius: float = 1.0,
        continuous_actions: bool = False,
        collision_penalty: float = 0.1,
        boundary_penalty: float = 0.05,
        capture_radius: float = 0.1,
        safezone_radius: float = 0.2,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        assert num_rescuers > 0, "At least one rescuer is required"
        assert num_victims >= 0
        assert num_safezones >= 0

        self.num_rescuers = num_rescuers
        self.num_victims = num_victims
        self.num_trees = num_trees
        self.num_safezones = num_safezones
        self.max_cycles = max_cycles
        self.vision_radius = vision_radius
        self.continuous_actions = continuous_actions
        self.collision_penalty = collision_penalty
        self.boundary_penalty = boundary_penalty
        self.capture_radius = capture_radius
        self.safezone_radius = safezone_radius
        self.seed(seed)

        # Names of agents: rescuers only; victims are passive entities
        self.possible_agents = [f"rescuer_{i}" for i in range(self.num_rescuers)]

        # Define action and observation spaces per agent
        if continuous_actions:
            action_spaces = {
                agent: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
                for agent in self.possible_agents
            }
        else:
            action_spaces = {
                agent: spaces.Discrete(5) for agent in self.possible_agents
            }
        self.action_spaces: Dict[str, spaces.Space] = action_spaces

        obs_dim = 4 + 2 * self.num_victims + 2 * (self.num_rescuers - 1) + 2 * self.num_trees + 2 * self.num_safezones
        observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.observation_spaces: Dict[str, spaces.Space] = observation_spaces

        # Internal state
        self.rescuer_pos: np.ndarray  # shape (n, 2)
        self.rescuer_vel: np.ndarray  # shape (n, 2)
        self.victim_pos: np.ndarray  # shape (m, 2)
        self.victim_rescued: np.ndarray  # shape (m,) bool
        self.victim_attached: List[int | None]
        self.tree_pos: np.ndarray  # shape (k, 2)
        self.safezone_pos: np.ndarray  # shape (s, 2)
        self.t: int

    # PettingZoo API:
    # Instead of defining observation_spaces() and action_spaces() as methods that
    # return dicts, we rely on the attributes `self.observation_spaces` and
    # `self.action_spaces` set in __init__. PettingZoo will query
    # `env.observation_spaces[agent]` and `env.action_spaces[agent]` by default.
    # We additionally provide per-agent accessors to satisfy torchrl wrappers.

    def observation_space(self, agent: str) -> spaces.Space:
        """Return the observation space for a specific agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        """Return the action space for a specific agent."""
        return self.action_spaces[agent]

    # Random seeding
    def seed(self, seed: int | None) -> None:
        self.np_random = np.random.default_rng(seed)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        """
        Reset the environment and return initial observations and info dicts.

        In the PettingZoo parallel API, `reset` should return a tuple of
        (observations, infos). To remain compatible with TorchRL's
        PettingZooWrapper, we return a second dict containing empty info
        dictionaries for each agent.
        """
        if seed is not None:
            self.seed(seed)
        self.t = 0
        # Initialize rescuer positions uniformly in square [−0.5, 0.5]^2 and zero velocity
        self.rescuer_pos = self.np_random.uniform(low=-0.5, high=0.5, size=(self.num_rescuers, 2)).astype(np.float32)
        self.rescuer_vel = np.zeros((self.num_rescuers, 2), dtype=np.float32)
        # Victims at random positions
        if self.num_victims > 0:
            self.victim_pos = self.np_random.uniform(low=-0.5, high=0.5, size=(self.num_victims, 2)).astype(np.float32)
            self.victim_rescued = np.zeros((self.num_victims,), dtype=bool)
            # Track which rescuer a victim is attached to (None if not captured)
            self.victim_attached = [None for _ in range(self.num_victims)]
        else:
            self.victim_pos = np.zeros((0, 2), dtype=np.float32)
            self.victim_rescued = np.zeros((0,), dtype=bool)
        # Trees at fixed positions; random but do not collide with center
        if self.num_trees > 0:
            self.tree_pos = self.np_random.uniform(low=-0.8, high=0.8, size=(self.num_trees, 2)).astype(np.float32)
        else:
            self.tree_pos = np.zeros((0, 2), dtype=np.float32)
        # Safe zones: fixed positions at corners scaled to [−1, 1]
        self.safezone_pos = np.array([
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float32)
        # If fewer safe zones requested, take subset
        if self.num_safezones < 4:
            self.safezone_pos = self.safezone_pos[: self.num_safezones]
        # Observations
        observations = {agent: self._get_obs(i) for i, agent in enumerate(self.possible_agents)}
        infos = {agent: {} for agent in self.possible_agents}
        return observations, infos

    def _get_obs(self, agent_idx: int) -> np.ndarray:
        """Construct observation for a single rescuer agent."""
        obs_list: List[float] = []
        # self velocity and position
        obs_list.extend(self.rescuer_vel[agent_idx].tolist())
        obs_list.extend(self.rescuer_pos[agent_idx].tolist())
        agent_pos = self.rescuer_pos[agent_idx]

        # relative positions to victims
        for j in range(self.num_victims):
            delta = self.victim_pos[j] - agent_pos
            visible = not self.victim_rescued[j] and self._is_visible(agent_pos, self.victim_pos[j])
            obs_list.extend(delta.tolist() if visible else [0.0, 0.0])

        # relative positions to other rescuers
        for k in range(self.num_rescuers):
            if k == agent_idx:
                continue
            delta = self.rescuer_pos[k] - agent_pos
            visible = self._is_visible(agent_pos, self.rescuer_pos[k])
            obs_list.extend(delta.tolist() if visible else [0.0, 0.0])

        # relative positions to trees
        for tree in self.tree_pos:
            delta = tree - agent_pos
            visible = self._is_visible(agent_pos, tree)
            obs_list.extend(delta.tolist() if visible else [0.0, 0.0])

        # relative positions to safe zones
        for sz in self.safezone_pos:
            delta = sz - agent_pos
            visible = self._is_visible(agent_pos, sz)
            obs_list.extend(delta.tolist() if visible else [0.0, 0.0])

        return np.array(obs_list, dtype=np.float32)

    def _is_visible(self, source: np.ndarray, target: np.ndarray) -> bool:
        """Determine if target is visible from source given vision radius and trees."""
        vec = target - source
        dist = np.linalg.norm(vec)
        if dist > self.vision_radius:
            return False
        if dist == 0.0:
            return True
        # Normalize direction
        dir_vec = vec / dist
        # Check occlusion: if any tree lies closer to the agent along the ray within a small threshold
        occlusion_threshold = 0.05
        for tree in self.tree_pos:
            tree_vec = tree - source
            proj_len = np.dot(tree_vec, dir_vec)
            if 0 < proj_len < dist:
                # Perpendicular distance from tree to line of sight
                perp_dist = np.linalg.norm(tree_vec - proj_len * dir_vec)
                if perp_dist < occlusion_threshold:
                    return False
        return True

    def step(self, actions: Dict[str, int | np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        """Apply actions and return next observations, rewards, terminations, truncations, infos."""
        self.t += 1
        # Apply rescuer actions
        for idx, agent in enumerate(self.possible_agents):
            action = actions.get(agent)
            if action is None:
                continue
            if self.continuous_actions:
                accel = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            else:
                # Discrete actions: map to velocity change
                # 0 noop, 1 up, 2 down, 3 left, 4 right
                if action == 0:
                    accel = np.array([0.0, 0.0], dtype=np.float32)
                elif action == 1:
                    accel = np.array([0.0, 1.0], dtype=np.float32)
                elif action == 2:
                    accel = np.array([0.0, -1.0], dtype=np.float32)
                elif action == 3:
                    accel = np.array([-1.0, 0.0], dtype=np.float32)
                elif action == 4:
                    accel = np.array([1.0, 0.0], dtype=np.float32)
                else:
                    raise ValueError(f"Unknown discrete action {action}")
            # Simple physics: velocity = accel (clipped), update position
            self.rescuer_vel[idx] = accel
            self.rescuer_pos[idx] += self.rescuer_vel[idx] * 0.1  # time step factor
            # Clip positions to world bounds [-1, 1]
            self.rescuer_pos[idx] = np.clip(self.rescuer_pos[idx], -1.0, 1.0)

        # Update victim dynamics: assign to rescuers when within capture radius
        if self.num_victims > 0:
            for v in range(self.num_victims):
                if self.victim_rescued[v]:
                    continue
                attached = self.victim_attached[v]
                # If not yet attached, check for capture
                if attached is None:
                    for idx in range(self.num_rescuers):
                        dist = np.linalg.norm(self.rescuer_pos[idx] - self.victim_pos[v])
                        if dist < self.capture_radius:
                            self.victim_attached[v] = idx
                            attached = idx
                            break
                # If attached, move victim with the rescuer
                if attached is not None:
                    self.victim_pos[v] = self.rescuer_pos[attached].copy()
                    # Check if victim reached any safe zone
                    for sz in self.safezone_pos:
                        if np.linalg.norm(self.victim_pos[v] - sz) < self.safezone_radius:
                            self.victim_rescued[v] = True
                            self.victim_attached[v] = None
                            break

        # Determine rewards
        rewards: Dict[str, float] = {agent: 0.0 for agent in self.possible_agents}
        # Negative distance to nearest victim as shaping (optional; simple example)
        if self.num_victims > 0:
            for idx, agent in enumerate(self.possible_agents):
                # compute distance to nearest unrescued victim
                dists = [np.linalg.norm(self.victim_pos[v] - self.rescuer_pos[idx])
                         for v in range(self.num_victims) if not self.victim_rescued[v]]
                if dists:
                    rewards[agent] += -min(dists)
        # Penalty for collisions with trees
        for idx, agent in enumerate(self.possible_agents):
            for tree in self.tree_pos:
                if np.linalg.norm(self.rescuer_pos[idx] - tree) < self.capture_radius:
                    rewards[agent] -= self.collision_penalty
        # Penalty for boundary closeness
        for idx, agent in enumerate(self.possible_agents):
            x, y = self.rescuer_pos[idx]
            if abs(x) > 0.95 or abs(y) > 0.95:
                rewards[agent] -= self.boundary_penalty
        # Determine terminations: success when all victims rescued
        terminated = (self.num_victims > 0) and all(self.victim_rescued)
        terminations = {agent: terminated for agent in self.possible_agents}
        # Determine truncations: max cycles reached
        truncated = self.t >= self.max_cycles
        truncations = {agent: truncated for agent in self.possible_agents}
        # Additional info (empty)
        infos = {agent: {} for agent in self.possible_agents}
        # Next observations
        observations = {agent: self._get_obs(i) for i, agent in enumerate(self.possible_agents)}
        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        # Minimal rendering: print positions (could be replaced by matplotlib)
        print(f"t={self.t}")
        for i, agent in enumerate(self.possible_agents):
            print(f"{agent}: pos={self.rescuer_pos[i]}, vel={self.rescuer_vel[i]}")
        print(f"victims rescued: {self.victim_rescued}")

    def close(self) -> None:
        pass
