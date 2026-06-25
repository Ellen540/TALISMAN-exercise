"""Basic sanity tests for the bench. These check the plumbing, not the physics."""

import numpy as np
import pytest

from talisman import LaserEnv


def collect(env, actions):
    """Reset and step through `actions`; return stacked images and sensors."""
    obs, _ = env.reset()
    images = [obs["image"]]
    sensors = [list(obs["sensors"].values())]
    for a in actions:
        obs, _, _, _, _ = env.step(a)
        images.append(obs["image"])
        sensors.append(list(obs["sensors"].values()))
    return np.stack(images), np.array(sensors)


def test_requires_reset_before_step():
    env = LaserEnv(seed=0)
    with pytest.raises(RuntimeError):
        env.step(np.zeros(2))


def test_observation_structure():
    env = LaserEnv(seed=0)
    obs, info = env.reset()
    assert set(obs) == {"image", "sensors"}
    assert obs["image"].shape == env.image_shape == (64, 64)
    assert obs["image"].dtype == np.float64
    assert np.all(np.isfinite(obs["image"]))
    assert tuple(obs["sensors"]) == env.sensor_names
    assert all(isinstance(v, float) for v in obs["sensors"].values())
    assert info["t"] == 0
    assert info["target"].shape == (2,)
    np.testing.assert_allclose(info["target"], env.target)


def test_step_returns_gym_style_tuple():
    env = LaserEnv(seed=0)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert set(obs) == {"image", "sensors"}
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["t"] == 1
    assert info["target"].shape == (2,)  # the setpoint rides along every step


def test_action_validation():
    env = LaserEnv(seed=0)
    env.reset()
    env.step([0.1, -0.2])  # plain lists are fine
    env.step(np.array([5.0, -5.0]))  # out-of-range values are clipped, not an error
    with pytest.raises(ValueError):
        env.step(np.zeros(3))
    with pytest.raises(ValueError):
        env.step([np.nan, 0.0])


def test_token_and_seed_are_mutually_exclusive():
    with pytest.raises(ValueError):
        LaserEnv(token="abc", seed=1)


def test_same_token_is_deterministic():
    actions = [np.zeros(2)] * 10 + [np.array([0.5, -0.3])] * 10
    img_a, sens_a = collect(LaserEnv(token="a3f9c2d4e5"), actions)
    img_b, sens_b = collect(LaserEnv(token="a3f9c2d4e5"), actions)
    np.testing.assert_array_equal(img_a, img_b)
    np.testing.assert_array_equal(sens_a, sens_b)


def test_reset_rewinds_the_same_session():
    env = LaserEnv(seed=7)
    img_a, _ = collect(env, [np.zeros(2)] * 5)
    img_b, _ = collect(env, [np.zeros(2)] * 5)
    np.testing.assert_array_equal(img_a, img_b)


def test_reset_with_seed_gives_a_fresh_session():
    env = LaserEnv(seed=7)
    obs_default, _ = env.reset()
    obs_fresh, _ = env.reset(seed=1)
    assert not np.array_equal(obs_default["image"], obs_fresh["image"])


def test_different_instances_differ():
    obs_a, _ = LaserEnv(seed=1).reset()
    obs_b, _ = LaserEnv(seed=2).reset()
    assert not np.array_equal(obs_a["image"], obs_b["image"])


def test_oracle_flag():
    obs, info = LaserEnv(seed=0).reset()
    assert "true_centroid" not in info
    env = LaserEnv(seed=0, oracle=True)
    obs, info = env.reset()
    assert info["true_centroid"].shape == (2,)
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["true_centroid"].shape == (2,)


def test_long_passive_run_keeps_spot_on_screen():
    env = LaserEnv(seed=42, oracle=True)
    _, info = env.reset()
    lo = np.full(2, np.inf)
    hi = np.full(2, -np.inf)
    for _ in range(2000):
        _, _, _, _, info = env.step(np.zeros(2))
        lo = np.minimum(lo, info["true_centroid"])
        hi = np.maximum(hi, info["true_centroid"])
    assert np.all(lo > 8.0) and np.all(hi < 56.0)
