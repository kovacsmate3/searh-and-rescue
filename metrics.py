"""
Metrics computation for the Search & Rescue project.

This module defines functions to compute evaluation metrics from logged
trajectories. Each metric accepts a list of episode logs, where each log
contains per‑step information such as whether a victim was rescued, number of
collisions, positions visited, etc. For simplicity, this implementation
assumes that evaluation code will collect the necessary statistics in the
expected format.
"""

from __future__ import annotations

from typing import List, Dict, Any


def rescues_completed(episodes: List[Dict[str, Any]]) -> float:
    """Compute the percentage of victims rescued across episodes.

    Each episode log should contain keys:
        - 'num_victims': total victims in the episode
        - 'rescued_victims': number of victims rescued by the end of the episode

    Returns:
        float: fraction of rescued victims averaged over episodes.
    """
    if not episodes:
        return 0.0
    total = 0
    rescued = 0
    for ep in episodes:
        total += ep.get("num_victims", 0)
        rescued += ep.get("rescued_victims", 0)
    return rescued / total if total > 0 else 0.0


def collision_count(episodes: List[Dict[str, Any]]) -> float:
    """Compute the average number of collisions per episode.

    Each episode log should contain 'collisions': total collisions in the episode.
    """
    if not episodes:
        return 0.0
    return sum(ep.get("collisions", 0) for ep in episodes) / len(episodes)


def coverage(episodes: List[Dict[str, Any]]) -> float:
    """Compute average coverage (unique cells visited) across episodes.

    Each episode log should contain 'coverage': number of distinct regions or cells
    visited by all agents. Higher coverage implies better exploration.
    """
    if not episodes:
        return 0.0
    return sum(ep.get("coverage", 0) for ep in episodes) / len(episodes)
