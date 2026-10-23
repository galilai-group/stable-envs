"""Roadster family: the five task cells on CPU, their State contract, weather Variation and rendering."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")

import stable_envs as snv
from stable_envs.envs.roadster.envs import CELLS
from stable_envs.envs.roadster.simulation import adapter as A
from tests.helpers.reference.semantics import assert_same_state
from tests.helpers.stepping import branch_step

pytestmark = pytest.mark.unit

NAMES = [name for name, _ in CELLS]
LANES = 2
SEED = 3


def _actions(env, steps: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1, 1, (env.num_envs, 4)).astype(np.float32) for _ in range(steps)]


@pytest.fixture(scope="module")
def jam():
    with snv.RoadsterTrafficJam(num_envs=LANES, device="cpu", seed=SEED) as env:
        yield env


# ------------------------------------------------------------------ every cell


@pytest.mark.parametrize("name", NAMES)
def test_cell_builds_resets_steps_and_restores_state(name):
    with snv.make(name, num_envs=LANES, device="cpu", seed=SEED) as env:
        assert env.family == "roadster" and env.TASK in A.TASKS
        state = env.reset(seed=SEED)
        env.state_spec.validate(state, owner="state")
        assert state["task", "id"].flatten().tolist() == [A.TASKS.index(env.TASK)] * LANES
        assert (state["lifecycle", "elapsed_steps"] == 0).all()
        actions = _actions(env, 3)
        out = env.step(np.stack(actions, 1), frames=None)
        assert out["valid"].all()
        for key in ("reward", "cost"):
            assert out[key].dtype == torch.float64 and torch.isfinite(out[key]).all()
        for key in ("success", "failed", "terminated", "truncated"):
            assert out[key].dtype == torch.bool
        assert (out["state"]["lifecycle", "elapsed_steps"] == 3).all()
        assert not torch.equal(out["state"]["sim", "qpos"], state["sim", "qpos"]), "the car did not move"
        env.set_state(state)
        again = env.step(np.stack(actions, 1), frames=None)
        assert_same_state(again["state"], out["state"])
        assert torch.equal(again["reward"], out["reward"])


def test_the_integration_state_is_split_in_mujoco_order(jam):
    """The ``sim`` columns, concatenated, are the integration state vector ``mj_getState`` writes."""
    import mujoco

    lane = jam._simulator.envs[0]
    n = mujoco.mj_stateSize(lane.model, mujoco.mjtState.mjSTATE_INTEGRATION)
    phys = np.empty(n)
    mujoco.mj_getState(lane.model, lane.data, phys, mujoco.mjtState.mjSTATE_INTEGRATION)
    row = jam.get_state()[0]
    joined = np.concatenate([row["sim", name].numpy() for name in jam._simulator.layout])
    np.testing.assert_array_equal(joined, phys)
    assert set(jam._simulator.layout) == {
        "time",
        "qpos",
        "qvel",
        "warmstart",
        "ctrl",
        "qfrc_applied",
        "xfrc_applied",
        "mocap_pos",
        "mocap_quat",
    }


# ------------------------------------------------------------------ State contract


def test_set_state_validates_everything_before_changing_any_lane(jam):
    state = jam.reset(seed=SEED)
    jam.step(_actions(jam, 1)[0], frames=None)
    live = jam.get_state()

    def refused(bad, match):
        with pytest.raises((TypeError, ValueError), match=match):
            jam.set_state(bad)
        assert_same_state(jam.get_state(), live)

    refused(state.exclude(("task", "fuel")), "missing")
    extra = state.clone()
    extra["spare"] = torch.zeros(LANES, 1)
    refused(extra, "unexpected")
    refused(state[:1], "batch")
    wrong_dtype = state.clone()
    wrong_dtype["task", "fuel"] = wrong_dtype["task", "fuel"].to(torch.float32)
    refused(wrong_dtype, "dtype")
    wrong_shape = state.clone()
    wrong_shape["goal", "position"] = wrong_shape["goal", "position"][:, :2]
    refused(wrong_shape, "shape")
    nonfinite = state.clone()
    nonfinite["sim", "qpos"][1, 0] = float("nan")
    refused(nonfinite, "non-finite")
    other_task = state.clone()
    other_task["task", "id"][0] = A.TASKS.index("carhop")
    refused(other_task, "task")
    bad_route = state.clone()
    bad_route["task", "route_idx"][0] = 7
    refused(bad_route, "route_idx")
    bad_rng = state.clone()
    bad_rng["weather", "rng_state"][0, 0] = 2**32
    refused(bad_rng, "rng_state")
    refused(jam._pack_render_state(state), "TensorDict")
    refused(jam._unpack_render_state(jam._pack_render_state(state)), "missing")
    with pytest.raises((TypeError, ValueError)):
        branch_step(jam, jam._pack_render_state(state), torch.zeros(LANES, 4))


def test_snapshots_are_independent_of_each_other_and_of_the_simulator(jam):
    jam.reset(seed=SEED)
    first = jam.get_state()
    frozen = first.clone()
    jam.step(_actions(jam, 1)[0], frames=None)
    second = jam.get_state()
    assert_same_state(first, frozen)
    assert not torch.equal(first["sim", "qpos"], second["sim", "qpos"])
    for value in first.values(True, True):
        value.zero_()
    assert_same_state(jam.get_state(), second)


@pytest.mark.parametrize("name", ["RoadsterLoop", "RoadsterCarHop"])
def test_restore_after_a_different_history_continues_bit_exactly(name):
    with (
        snv.make(name, num_envs=LANES, device="cpu") as source,
        snv.make(name, num_envs=LANES, device="cpu") as target,
    ):
        source.reset(seed=SEED, options={"variation_values": {"weather.wind": [6.0], "weather.rain": [0.4]}})
        for action in _actions(source, 4):
            source.step(action, frames=None)
        saved = source.get_state()
        held = saved.clone()
        target.reset(seed=SEED + 10)
        for action in _actions(target, 6, seed=5):  # unrelated contacts, gusts and autopilot history
            target.step(action, frames=None)
        assert target.set_state(saved) is None
        assert_same_state(target.get_state(), saved)
        torch.testing.assert_close(target.get_variation()[:, 1], torch.full((LANES,), 6.0), rtol=0, atol=0)
        assert (saved["physics", "weather", "wind"] == 6.0).all() and (
            saved["physics", "weather", "rain"] == 0.4
        ).all()
        actions = np.stack(_actions(source, 8, seed=7), 1)
        a, b = source.step(actions, frames=None), target.step(actions, frames=None)
        assert_same_state(a, b)
        assert_same_state(saved, held)


def test_an_inactive_lane_holds_its_state_and_reports_zeros(jam):
    jam.reset(seed=SEED)
    before = jam.get_state()
    action = torch.as_tensor(_actions(jam, 1)[0])
    full = jam._step(action)
    jam.set_state(before)
    masked = jam._step(action, active=torch.tensor([True, False]))
    assert_same_state(masked["state"][1], before[1])
    assert_same_state(masked["state"][0], full["state"][0])
    for key in masked.exclude("state").keys(True, True):
        assert not masked[key][1].any(), key


def test_actions_are_validated_before_the_simulator_moves(jam):
    jam.reset(seed=SEED)
    live = jam.get_state()
    for bad in (
        np.zeros((LANES, 3), np.float32),
        np.full((LANES, 4), np.nan, np.float32),
        np.zeros((LANES, 4), np.int64),
    ):
        with pytest.raises((TypeError, ValueError)):
            jam._step(bad)
        assert_same_state(jam.get_state(), live)


# ------------------------------------------------------------------ Variation


def test_wind_and_rain_reach_the_physics_at_reset():
    with snv.RoadsterTrafficJam(num_envs=LANES, device="cpu") as env:
        assert env.variation_names == ["weather.rain", "weather.wind"]
        assert env.dynamics_variation_leaves == {"weather.wind", "weather.rain"}
        assert env.physics_variation_leaves == ("weather.wind", "weather.rain")
        calm = env.reset(seed=SEED)
        assert (calm["weather", "wind_mean"] == 0).all() and (calm["weather", "grip"] == 1).all()
        actions = np.stack(_actions(env, 10), 1)
        calm_end = env.step(actions, frames=None)["state"]
        windy = env.reset(seed=SEED, options={"variation_values": {"weather.wind": [10.0], "weather.rain": [0.3]}})
        assert (windy["physics", "weather", "wind"] == 10.0).all() and (
            windy["physics", "weather", "rain"] == 0.3
        ).all()
        assert (windy["weather", "rain"] == 0.3).all(), "the float64 value reaches the source exactly"
        torch.testing.assert_close(
            windy["weather", "wind_mean"].norm(dim=-1), torch.full((LANES,), 10.0, dtype=torch.float64)
        )
        assert (windy["weather", "gust_std"] == 3.0).all() and (windy["weather", "grip"] == 1 - 0.45 * 0.3).all()
        assert_same_state(windy["sim", "qpos"], calm["sim", "qpos"])  # the task draw does not depend on the weather
        windy_end = env.step(actions, frames=None)["state"]
        assert not torch.equal(windy_end["sim", "qpos"], calm_end["sim", "qpos"])
        again = env.reset(seed=SEED)  # a one-reset override does not persist
        assert (again["weather", "wind_mean"] == 0).all()
        env.set_variation({"weather.wind": [4.0]})
        state = env.reset(seed=SEED)
        torch.testing.assert_close(
            state["weather", "wind_mean"].norm(dim=-1), torch.full((LANES,), 4.0, dtype=torch.float64)
        )
        sampled = env.reset(seed=SEED, options={"variation": ["weather.rain"]})
        assert (sampled["weather", "rain"] > 0).all() and (sampled["weather", "grip"] < 1).all()


# ------------------------------------------------------------------ rendering


def test_render_draws_the_wm_camera_without_touching_the_simulation(jam):
    state = jam.reset(seed=SEED)
    held = state.clone()
    frames = jam.render(state)
    assert frames.shape == (LANES, 96, 96, 3) and frames.dtype == torch.uint8
    assert tuple(jam.frame_spec.shape) == (96, 96, 3)
    assert frames.float().std() > 10, "a blank frame"
    assert not torch.equal(frames[0], frames[1]), "the lanes start at different places"
    assert torch.equal(frames, jam.render(state)), "rendering is not deterministic"
    rows = jam._pack_render_state(state)
    assert torch.equal(frames, jam._render(rows))
    assert_same_state(state, held)
    assert_same_state(jam.get_state(), state)
    assert jam.render(state, size=48).shape == (LANES, 48, 48, 3)
    assert jam.render(state[:1]).shape == (1, 96, 96, 3)
    out = torch.empty((LANES, 96, 96, 3), dtype=torch.uint8)
    assert jam.render(state, out=out) is out and torch.equal(out, frames)
    control = jam.step(_actions(jam, 1)[0], frames=None)["state"]
    jam.set_state(state)
    jam.render(state)
    rendered = jam.step(_actions(jam, 1)[0], frames=None)["state"]
    assert_same_state(rendered, control)
    after = jam.render(rendered)
    assert not torch.equal(after, frames)


def test_restored_state_renders_the_same_smoke():
    """With the jets firing, a State restored into another env draws the original run's smoke."""
    from stable_envs.envs.roadster.source import GoalExpert

    with snv.RoadsterCarHop(num_envs=1, device="cpu") as source, snv.RoadsterCarHop(num_envs=1, device="cpu") as other:
        source.reset(seed=1)
        other.reset(seed=5)
        expert = GoalExpert()
        for _ in range(8):
            result = source.step(expert(source._simulator.envs[0])[None].astype(np.float32), frames=None)
            other.step(expert(other._simulator.envs[0])[None].astype(np.float32), frames=None)
        state = result["state"]
        assert (
            state["fx", "count"].item() == 40 and state["fx", "history"].reshape(75, -1)[:40, 12:21].abs().max() > 0.05
        )
        frame = source.render(state)
        other.set_state(state)
        assert torch.equal(other.render(other.get_state()), frame)
        blank = state.clone()
        blank["fx", "count"].zero_()
        blank["fx", "history"].zero_()
        assert not torch.equal(source.render(blank), frame), "the smoke comes from the history"
        bad = state.clone()
        bad["fx", "count"][0] = 76
        with pytest.raises(ValueError, match="fx"):
            source.set_state(bad)
        bad = state.clone()
        bad["fx", "history"][0, 0] = 5.0  # step indices no longer increase
        with pytest.raises(ValueError, match="increasing"):
            source.set_state(bad)


def test_rendered_weather_follows_the_state(jam):
    dry = jam.reset(seed=SEED)
    wet = jam.reset(seed=SEED, options={"variation_values": {"weather.rain": [1.0]}})
    assert_same_state(dry["sim", "qpos"], wet["sim", "qpos"])
    a, b = jam.render(dry), jam.render(wet)
    assert (a != b).float().mean() > 0.1, "rain changes the frame"
    assert torch.equal(jam.render(dry), a), "the dry look is restored after a wet frame"
    assert torch.equal(jam.render(dry, variation={"weather.rain": [1.0]}), b)
    with pytest.raises(KeyError):
        jam.render(dry, variation={"sky.color": [0, 0, 0]})


# ------------------------------------------------------------------ registration


def test_the_family_registers_five_cells_and_reports_a_missing_extra():
    assert snv.list_envs(family="roadster") == sorted(NAMES)
    assert "roadster" not in snv.import_errors()
    code = (
        "import sys; sys.modules['mujoco'] = None\n"
        "import stable_envs as snv\n"
        "print(snv.import_errors().get('roadster'), snv.list_envs(family='roadster'))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=True).stdout
    assert out.startswith(("ImportError", "ModuleNotFoundError")) and out.strip().endswith("[]"), out


def test_cinematic_look_renders_same_state():
    """look="cinematic" draws the same State at any size without touching the live simulation, and the
    post-processed frame is a deterministic function of the State (two calls, two envs)."""
    from stable_envs.envs.roadster import render as roadster_render

    with (
        snv.RoadsterCarHop(num_envs=1, device="cpu", look="cinematic", size=128) as env,
        snv.RoadsterCarHop(num_envs=1, device="cpu", look="cinematic", size=128) as twin,
    ):
        state = env.reset(seed=0)
        for _ in range(6):
            state = env.step(np.array([[0.2, 0.0, 1.0, 0.0]], np.float32), frames=None)["state"]
        before = env.get_state()
        frame = env.render(state)
        assert tuple(frame.shape) == (1, 128, 128, 3) and frame.dtype == torch.uint8
        assert (env.get_state() == before).all()
        assert torch.equal(frame, env.render(state))
        twin.reset(seed=9)
        assert torch.equal(frame, twin.render(state))
        raw = np.empty((1, 128, 128, 3), np.uint8)
        roadster_render.renderer(128, "cinematic", postfx=False).draw(
            env._pack_render_state(state).to(torch.float64).numpy(), raw
        )
        assert not np.array_equal(raw[0], frame[0].numpy()), "the post-processing changes the frame"
    with pytest.raises(ValueError):
        snv.RoadsterTrafficJam(num_envs=1, device="cpu", look="bogus")


def test_cinematic_frame_does_not_depend_on_what_was_drawn_before():
    """Rain changes MuJoCo's fog; a dry State drawn after a wet one gets the dark-space fog back."""
    with snv.RoadsterLoop(num_envs=1, device="cpu", look="cinematic", size=64) as env:
        dry = env.reset(seed=SEED)
        first = env.render(dry)
        wet = env.reset(seed=SEED, options={"variation_values": {"weather.rain": [0.8]}})
        assert not torch.equal(env.render(wet), first)
        assert torch.equal(env.render(dry), first)
