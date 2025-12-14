import numpy as np

from env import SearchRescueEnv
from scenarios import ScenarioGenerator
from metrics import rescues_completed, collision_count, coverage


def test_action_space_modes():
    """Ensure the environment exposes correct action spaces for discrete and continuous modes."""
    # Discrete
    env_disc = SearchRescueEnv(num_rescuers=2, num_victims=1, continuous_actions=False)
    for agent, space in env_disc.action_spaces().items():
        assert space.n == 5  # five discrete actions
    # Continuous
    env_cont = SearchRescueEnv(num_rescuers=2, num_victims=1, continuous_actions=True)
    for agent, space in env_cont.action_spaces().items():
        assert space.shape == (2,) and np.all(space.low == -1.0) and np.all(space.high == 1.0)


def test_victim_capture_and_rescue():
    """Test that victims become rescued when escorted to a safe zone."""
    env = SearchRescueEnv(
        num_rescuers=1,
        num_victims=1,
        num_trees=0,
        num_safezones=1,
        max_cycles=5,
        continuous_actions=False,
        capture_radius=0.2,
        safezone_radius=0.2,
    )
    env.reset(seed=0)
    # Manually set positions: rescuer next to victim and safe zone aligned
    env.rescuer_pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    env.victim_pos[0] = np.array([0.05, 0.0], dtype=np.float32)
    env.safezone_pos = np.array([[0.1, 0.0]], dtype=np.float32)
    env.victim_rescued[0] = False
    env.victim_attached[0] = None
    # Step: take noop to trigger capture and escort
    for _ in range(3):
        obs, rewards, terminations, truncations, infos = env.step({"rescuer_0": 0})
        if env.victim_rescued[0]:
            break
    assert bool(env.victim_rescued[0])


def test_scenario_generator_sampling_and_curriculum():
    """Verify the ScenarioGenerator samples within specified ranges and produces a curriculum."""
    gen = ScenarioGenerator(
        num_rescuers=2,
        victim_range=(1, 3),
        tree_range=(2, 5),
        safezone_range=(1, 3),
        map_scale_range=(1.0, 2.0),
        curriculum_steps=3,
        seed=42,
    )
    # Sample random scenario
    params = gen.sample()
    assert 1 <= params.num_victims <= 2
    assert 2 <= params.num_trees <= 4
    assert 1 <= params.num_safezones <= 2
    assert 1.0 <= params.map_scale <= 2.0
    # Curriculum should yield increasing tree counts of length 3
    tree_counts = [p.num_trees for p in gen.curriculum()]
    # The generator is infinite when curriculum_steps > 0; take first 3 entries
    assert len(tree_counts) >= 3
    assert tree_counts[0] <= tree_counts[1] <= tree_counts[2]


def test_metrics_functions():
    """Check metrics computation functions on synthetic data."""
    episodes = [
        {"num_victims": 3, "rescued_victims": 2, "collisions": 1, "coverage": 10},
        {"num_victims": 1, "rescued_victims": 1, "collisions": 0, "coverage": 5},
    ]
    assert np.isclose(rescues_completed(episodes), 3 / 4)
    assert np.isclose(collision_count(episodes), 0.5)
    assert np.isclose(coverage(episodes), 7.5)
