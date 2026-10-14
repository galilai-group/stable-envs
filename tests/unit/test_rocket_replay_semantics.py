"""Official-engine Rocket State ownership and atomicity regressions."""

import numpy as np
import pytest
import torch

pytest.importorskip("PyFlyt")
pytest.importorskip("pybullet")

from stable_envs import make
from tests.helpers.reference.semantics import assert_same_state


@pytest.fixture
def pair():
    torch.set_num_threads(1)
    a, b = make("PyFlytRocketLanding", num_envs=2), make("PyFlytRocketLanding", num_envs=2, seed=42)
    try:
        yield a, b
    finally:
        a.close()
        b.close()


def test_whole_batch_invalid_action_does_not_step_first_lane(pair):
    a, b = pair
    b.set_state(a.get_state())
    saved = a.get_state()
    actions = np.zeros((2, 7), np.float64)
    actions[1, 5] = 2
    for _ in range(3):
        with pytest.raises(AssertionError):
            a.step(actions)
    assert_same_state(saved, a.get_state())
    assert_same_state(a.step(np.zeros((2, 7))), b.step(np.zeros((2, 7))))


@pytest.mark.parametrize(
    "kind", ["nan", "missing", "dtype", "history_mask", "config", "unreachable_second_lane", "rng_increment"]
)
def test_invalid_whole_state_is_atomic(pair, kind):
    a, b = pair
    original = a.get_state()
    corrupted = original.clone()
    if kind == "nan":
        corrupted["current", "wrapper", "reward"][1] = float("nan")
    elif kind == "missing":
        del corrupted["history", "length"]
    elif kind == "dtype":
        corrupted["history", "actions"] = corrupted["history", "actions"].float()
    elif kind == "history_mask":
        corrupted["history", "valid"][1, 0] = True
    elif kind == "config":
        corrupted["recipe", "config"][1, 0] += 1
    elif kind == "unreachable_second_lane":
        corrupted["current", "wrapper", "reward"][1] += 1
    elif kind == "rng_increment":
        corrupted["recipe", "main_rng"][1, 5] = 2
    with pytest.raises((ValueError, TypeError)):
        a.set_state(corrupted)
    assert_same_state(original, a.get_state())
    b.set_state(original)
    assert_same_state(a.step(np.zeros((2, 7))), b.step(np.zeros((2, 7))))


def test_independent_storage_mixed_dtypes_and_different_histories(pair):
    a, b = pair
    first = a.get_state()
    for i in range(12):
        a.step(np.array([[0.2, -0.3, 0.1, 1, 0.4, 0.1, -0.2]] * 2, dtype=np.float32 if i % 2 else np.float64))
    saved = a.get_state()
    a.set_state(first)
    b.set_state(saved)
    assert not torch.equal(a.get_state()["history", "length"], saved["history", "length"])
    a.set_state(saved)
    saved["current", "wrapper", "reward"].fill_(123)
    for _ in range(3):
        assert_same_state(a.step(np.zeros((2, 7))), b.step(np.zeros((2, 7))))
    assert a.get_state()["history", "length"].dtype == torch.int64
    assert_same_state(a.to_render_state(), b.to_render_state())


def test_seeded_future_reset_and_fresh_entropy(pair):
    a, b = pair
    a.reset(seed=17, options={"variation": ["all"]})
    b.set_state(a.get_state())
    a.set_variation({"environment.start_tilt": np.array([0.1, -0.1], np.float32)})
    b.set_state(a.get_state())
    assert_same_state(
        a.reset(seed=42, options={"variation": ["rocket"]}), b.reset(seed=42, options={"variation": ["rocket"]})
    )
    a.reset(seed=None, options={"variation": ["all"]})
    saved = a.get_state()
    b.set_state(saved)
    assert_same_state(saved, b.get_state())
    a.reset(seed=None, options={"variation": ["all"]})
    fresh = a.get_state()
    assert not torch.equal(
        saved["variation", "rocket", "body_color", "rng"], fresh["variation", "rocket", "body_color", "rng"]
    )
    b.set_state(fresh)
    assert_same_state(a.step(np.zeros((2, 7))), b.step(np.zeros((2, 7))))


def test_cpu_render_supplied_state_is_exact_and_nonmutating():
    env = make("PyFlytRocketLanding", num_envs=2, simulator_config={"render_mode": "rgb_array"})
    try:
        state = env.get_state()
        pixels = env.render(state)
        expected = np.stack([sim.env.render() for sim in env._simulator.envs])
        assert pixels.device.type == "cpu" and pixels.dtype == torch.uint8
        np.testing.assert_array_equal(pixels.numpy(), expected)
        assert_same_state(state, env.get_state())
        projected = env.to_render_state(state)
        before = projected.clone()
        out = torch.empty_like(pixels)
        assert env.render(projected, out=out).data_ptr() == out.data_ptr()
        assert torch.equal(pixels, out)
        assert_same_state(projected, before)
        for i in range(env.B):
            assert torch.equal(env.render(projected[i : i + 1])[0], pixels[i])
        bad = projected.clone()
        bad["renderer", "camera_pose"][-1, 0] = float("nan")
        with pytest.raises(ValueError):
            env.render(bad)
        assert_same_state(state, env.get_state())
    finally:
        env.close()


def test_render_projection_matches_actual_engine_materials(pair):
    from stable_envs.envs.rocket import legacy

    e = pair[0]
    e.reset(seed=17, options={"variation": ["all"]})
    projected = e.to_render_state()
    materials = legacy.material_table(projected["renderer", "style"].numpy())
    rgba = e.get_state()["render", "rgba"].numpy()
    for material, source in [(1, 0), (4, 1), (8, 2), (3, 3), (12, 4)]:
        np.testing.assert_allclose(materials[:, material], rgba[:, source, :3], rtol=0, atol=1e-7)


def test_compact_projection_cannot_restore_physics(pair):
    e = pair[0]
    saved = e.get_state()
    rows = e._pack_render_state(saved)
    assert torch.equal(rows, saved["current", "wrapper", "state"][:, :13].float())
    for incomplete in (rows, e._unpack_render_state(rows), e.to_render_state(saved)):
        with pytest.raises((ValueError, TypeError)):
            e.set_state(incomplete)
        assert_same_state(saved, e.get_state())


def test_variation_getter_tracks_reset_and_restore(pair):
    e, destination = pair
    e.reset(seed=17, options={"variation": ["all"]})
    actual = e.get_variation()
    for key, offset, width in e.variation_space.flat_layout():
        for lane, replay in enumerate(e._simulator.envs):
            value = np.asarray(replay.env.variation_space._walk(key).value).reshape(-1)
            np.testing.assert_array_equal(actual[lane, offset : offset + width].numpy(), value.astype(np.float32))
    destination.set_state(e.get_state())
    assert torch.equal(actual, destination.get_variation())
    anchor = e.set_variation({"environment.start_tilt": np.array([0.1, -0.1], np.float32)})
    assert anchor.shape == actual.shape
    assert torch.equal(actual, e.get_variation())  # Persistent change applies on reset.
    anchor.zero_()
    e.reset(seed=17)
    for replay in e._simulator.envs:
        np.testing.assert_array_equal(
            replay.env.variation_space._walk("environment.start_tilt").value, np.array([0.1, -0.1], np.float32)
        )


@pytest.mark.swm
@pytest.mark.parametrize("scalar_type", [float, np.float32, np.float64])
def test_persistent_height_preserves_source_precision(pair, scalar_type):
    from stable_worldmodel.envs.rocket_landing.pyflyt_rocketlanding import RocketLandingEnv as Oracle

    e, destination = pair
    height = scalar_type(0.8123456789012345)
    e.set_variation({"environment.start_height_ratio": height})
    e.reset(seed=17)
    source = Oracle()
    try:
        source.variation_space["environment"]["start_height_ratio"].set_init_value(height)
        source.reset(seed=17)
        np.testing.assert_array_equal(e._simulator.envs[0].env.state, source.state)
        saved = e.get_state()
        assert saved["variation", "environment", "start_height_ratio", "value"][0].item() == float(height)
        destination.set_state(saved)
        assert_same_state(saved, destination.get_state())
        source.step(np.zeros(7))
        e.step(np.zeros((2, 7)), frames=None)
        np.testing.assert_array_equal(e._simulator.envs[0].env.state, source.state)
    finally:
        source.close()


def test_sampled_configurations_are_supported_and_restorable(pair):
    e, destination = pair
    sampled = e.sample_variation(rng=np.random.default_rng(17))
    e.set_variation(sampled)
    e.reset(seed=17)
    assert torch.equal(e.get_variation(), torch.from_numpy(sampled))
    destination.set_state(e.get_state())
    assert_same_state(e.step(np.zeros((2, 7)), frames=None), destination.step(np.zeros((2, 7)), frames=None))


def test_set_variation_repaints_the_current_episode_and_restores(pair):
    e, destination = pair
    e.reset(seed=17)
    before = e.get_state()
    red, pad = np.array([200, 30, 90], np.uint8), np.array([104, 60, 3], np.uint8)
    tilt = np.full(2, 0.1, np.float32)
    e.set_variation({"rocket.body_color": red, "pad.color": pad, "environment.start_tilt": tilt})
    saved = e.get_state()
    np.testing.assert_array_equal(saved["render", "rgba"][:, 0, :3].numpy(), np.tile(red / 255.0, (2, 1)))
    pad_rgba = np.float32([0.408, 0.235, 0.012]).astype(np.float64)
    np.testing.assert_array_equal(saved["render", "rgba"][:, 4, :3].numpy(), np.tile(pad_rgba, (2, 1)))
    # The start leaves describe how an episode begins, so they wait for the next reset.
    for key in [("current", "wrapper", "state"), ("variation", "environment", "start_tilt", "value")]:
        assert torch.equal(saved[key], before[key])
    destination.set_state(saved)
    assert_same_state(saved, destination.get_state())
    tampered = saved.clone()
    tampered["variation", "rocket", "body_color", "value"][0, 0] = 10
    with pytest.raises(ValueError):
        destination.set_state(tampered)
    # Every later reset draws the configured defaults, and Variation reports what is drawn.
    for options in [None, {"variation": ["environment.start_tilt"]}]:
        plain = e.reset(seed=18, options=options)
        assert torch.equal(plain["render", "rgba"][:, [0, 4]], saved["render", "rgba"][:, [0, 4]])
        assert torch.equal(
            plain["variation", "rocket", "body_color", "value"], saved["variation", "rocket", "body_color", "value"]
        )
        destination.set_state(plain)
        assert_same_state(plain, destination.get_state())


def test_variation_values_last_one_reset_and_restore(pair):
    e, destination = pair
    tilt = np.array([[0.1, -0.1], [-0.2, 0.05]], np.float32)
    values = {"environment.start_tilt": tilt, "rocket.body_color": np.array([12, 140, 250], np.uint8)}
    saved = e.reset(seed=17, options={"variation": ["rocket.fin_color"], "variation_values": values})
    np.testing.assert_array_equal(saved["variation", "environment", "start_tilt", "value"].numpy(), tilt)
    destination.set_state(saved)
    assert_same_state(saved, destination.get_state())
    assert not e.reset(seed=17)["variation", "environment", "start_tilt", "value"].any()
    with pytest.raises(ValueError, match="both sampled and set"):
        e.reset(seed=17, options={"variation": ["environment"], "variation_values": values})
