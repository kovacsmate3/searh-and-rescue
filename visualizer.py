"""Pygame visualizer for the Search & Rescue environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pygame

from env import SearchRescueEnv


@dataclass
class ViewerConfig:
    width: int = 800
    height: int = 800
    fps: int = 30
    agent_color: tuple[int, int, int] = (50, 100, 255)
    victim_color: tuple[int, int, int] = (220, 80, 80)
    tree_color: tuple[int, int, int] = (30, 160, 60)
    safezone_color: tuple[int, int, int] = (250, 190, 20)
    bg_color: tuple[int, int, int] = (15, 15, 25)
    world_half_extent: float = 1.1
    agent_radius: int = 10
    tree_radius: int = 6
    victim_radius: int = 6
    safezone_line_width: int = 2
    font_size: int = 18


class SearchRescueViewer:
    def __init__(self, env: SearchRescueEnv, config: Optional[ViewerConfig] = None) -> None:
        self.env = env
        self.cfg = config or ViewerConfig()
        pygame.init()
        self.screen = pygame.display.set_mode((self.cfg.width, self.cfg.height))
        pygame.display.set_caption("Search & Rescue")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", self.cfg.font_size)

    def _world_to_screen(self, pos: np.ndarray) -> tuple[int, int]:
        scale = min(self.cfg.width, self.cfg.height) / (2 * self.cfg.world_half_extent)
        x = (pos[0] + self.cfg.world_half_extent) * scale
        y = (self.cfg.world_half_extent - pos[1]) * scale
        return int(x), int(y)

    def _draw_agents(self) -> None:
        for idx, agent in enumerate(self.env.possible_agents):
            pos = self.env.rescuer_pos[idx]
            color = self.cfg.agent_color
            pygame.draw.circle(self.screen, color, self._world_to_screen(pos), self.cfg.agent_radius)
            label = self.font.render(agent, True, (240, 240, 240))
            self.screen.blit(label, (self._world_to_screen(pos)[0] + 5, self._world_to_screen(pos)[1] - 5))

    def _draw_victims(self) -> None:
        for pos in self.env.victim_pos:
            pygame.draw.circle(self.screen, self.cfg.victim_color, self._world_to_screen(pos), self.cfg.victim_radius)

    def _draw_trees(self) -> None:
        for pos in self.env.tree_pos:
            pygame.draw.circle(self.screen, self.cfg.tree_color, self._world_to_screen(pos), self.cfg.tree_radius)

    def _draw_safezones(self) -> None:
        for pos in self.env.safezone_pos:
            rect = pygame.Rect(0, 0, self.cfg.safezone_line_width * 4, self.cfg.safezone_line_width * 4)
            rect.center = self._world_to_screen(pos)
            pygame.draw.rect(self.screen, self.cfg.safezone_color, rect, self.cfg.safezone_line_width)

    def update(self, fps: Optional[int] = None) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
        self.screen.fill(self.cfg.bg_color)
        self._draw_safezones()
        self._draw_trees()
        self._draw_victims()
        self._draw_agents()
        pygame.display.flip()
        self.clock.tick(fps or self.cfg.fps)

    def render_once(self) -> None:
        self.update()

    def run_episode(self, actions_fn) -> None:
        reset_out = self.env.reset()
        if isinstance(reset_out, tuple):
            obs, _ = reset_out
        else:
            obs = reset_out
        done = False
        self.update()
        while not done:
            actions = actions_fn(obs)
            step_out = self.env.step(actions)
            if len(step_out) == 5:
                obs, rewards, terminations, truncations, _ = step_out
            else:
                obs, rewards, terminations, truncations = step_out
            self.update()
            done = all(terminations.values()) or all(truncations.values())

    @staticmethod
    def close() -> None:
        pygame.quit()


def random_policy(env: SearchRescueEnv):
    if env.continuous_actions:
        return {
            agent: env.action_space(agent).sample()
            for agent in env.possible_agents
        }
    return {
        agent: env.action_space(agent).sample()
        for agent in env.possible_agents
    }


def run_random_episode(env: SearchRescueEnv, cfg: Optional[ViewerConfig] = None) -> None:
    viewer = SearchRescueViewer(env, cfg)
    try:
        viewer.run_episode(lambda obs: random_policy(env))
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
