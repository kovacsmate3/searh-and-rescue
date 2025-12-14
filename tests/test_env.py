import numpy as np

from env import SearchRescueEnv


def test_reset_and_step():
    """Basic smoke test: environment reset and step return valid structures."""
    env = SearchRescueEnv(num_rescuers=2, num_victims=1, num_trees=1, num_safezones=2, max_cycles=10)
    obs, infos = env.reset(seed=42)
    # Ensure observations for each agent present
    assert set(obs.keys()) == set(env.possible_agents)
    # Check observation dimensions
    for agent, o in obs.items():
        assert o.shape == env.observation_space(agent).shape
    # Take random actions and step
    actions = {agent: env.action_space(agent).sample() for agent in env.possible_agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    # Check types
    assert isinstance(obs, dict)
    assert isinstance(rewards, dict)
    assert isinstance(terminations, dict)
    assert isinstance(truncations, dict)


def test_occlusion_logic():
    """Verify that occlusion hides objects behind trees."""
    # Create environment with one rescuer and one victim and one tree
    env = SearchRescueEnv(num_rescuers=1, num_victims=1, num_trees=1, num_safezones=0, max_cycles=1, vision_radius=1.0)
    env.reset(seed=0)
    # Manually place rescuer at origin
    env.rescuer_pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    env.rescuer_vel[0] = np.zeros(2, dtype=np.float32)
    # Place victim at (0.8, 0.0)
    env.victim_pos[0] = np.array([0.8, 0.0], dtype=np.float32)
    env.victim_rescued[0] = False
    # Place tree directly between rescuer and victim at (0.4, 0.0)
    env.tree_pos[0] = np.array([0.4, 0.0], dtype=np.float32)
    # Observations should mask victim position to zero due to occlusion
    obs = env._get_obs(0)
    # Observation layout: [vel_x, vel_y, pos_x, pos_y, victim_dx, victim_dy, ...]
    victim_dx, victim_dy = obs[4], obs[5]
    assert np.isclose(victim_dx, 0.0) and np.isclose(victim_dy, 0.0)
    # Move tree off line of sight; should reveal victim
    env.tree_pos[0] = np.array([0.4, 0.2], dtype=np.float32)
    obs_visible = env._get_obs(0)
    victim_dx2, victim_dy2 = obs_visible[4], obs_visible[5]
    assert not (np.isclose(victim_dx2, 0.0) and np.isclose(victim_dy2, 0.0))
