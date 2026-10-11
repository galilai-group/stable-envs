"""Exact semantic gate for the three historical native navigation defaults."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

import stable_envs as se
from tests.helpers.reference.native_navigation_semantics import NativeNavigationReference
from tests.helpers.reference.semantics import (
    check_reset_matches_reference,
    check_state_roundtrip,
    check_step_matches_reference,
    check_two_histories_same_state,
    compare_state,
)

SWM_PIN = "21446f1ede6d5284e981bd7b47f432b994e6d812"
CASES = {
    "two_room": {"historical_id": "swm/TwoRoom-v1", "stable_env": "TwoRoom"},
    "piecewise": {"historical_id": "swm/Piecewise-v0", "stable_env": "Piecewise"},
    "simple_point_maze": {
        "historical_id": "swm/SimplePointMaze-v0",
        "stable_env": "SimplePointMaze",
    },
}



_DIAGNOSTIC_CONTEXT = None


def _diagnostic_value(value):
    if torch.is_tensor(value):
        host = value.detach().cpu().contiguous()
        record = {"dtype": str(value.dtype), "device": str(value.device), "shape": list(value.shape)}
        if host.numel() <= 256:
            record["values"] = host.tolist()
        else:
            record["sha256"] = hashlib.sha256(host.numpy().tobytes()).hexdigest()
        return record
    if isinstance(value, np.ndarray):
        return _diagnostic_value(torch.from_numpy(value.copy()))
    if hasattr(value, "keys"):
        return {str(key): _diagnostic_value(value[key]) for key in value.keys()}
    return value


def _probe(name):
    if _DIAGNOSTIC_CONTEXT is not None:
        _DIAGNOSTIC_CONTEXT["probe"] = name


@contextmanager
def _diagnostic_case(env, ref, domain, case):
    """Observe the family gate without changing comparisons or simulator inputs."""
    global _DIAGNOSTIC_CONTEXT
    previous = _DIAGNOSTIC_CONTEXT
    context = {"domain": domain, "historical_id": CASES[domain]["historical_id"],
               "case": case, "probe": case, "batch": env.B, "lanes": list(range(env.B)),
               "seed": None, "candidate": {}, "reference": {}}
    _DIAGNOSTIC_CONTEXT = context
    saved = []

    def observe(obj, label, method):
        original = getattr(obj, method)
        # Restore the original instance dictionary, including class-bound methods.
        prior = obj.__dict__.get(method)
        had_instance_method = method in obj.__dict__
        saved.append((obj, method, prior, had_instance_method))

        def call(*args, **kwargs):
            record = context[label]
            if method == "reset":
                record["seed"] = kwargs.get("seed", args[0] if args else None)
                if label == "candidate":
                    context["seed"] = record["seed"]
                record["step_since_reset_or_restore"] = 0
            elif method == "set_state":
                record["restore_input"] = _diagnostic_value(args[0])
                record["step_since_reset_or_restore"] = 0
            elif method == "step":
                record["step_since_reset_or_restore"] = record.get("step_since_reset_or_restore", 0) + 1
                record["action"] = _diagnostic_value(args[0])
                record["state_before_step"] = _diagnostic_value(obj.get_state())
            result = original(*args, **kwargs)
            if method == "step":
                record["output"] = _diagnostic_value(result)
            return result

        setattr(obj, method, call)

    try:
        for obj, label in ((env, "candidate"), (ref, "reference")):
            for method in ("reset", "set_state", "step"):
                observe(obj, label, method)
        yield
    except Exception as error:
        context["error"] = repr(error)
        for obj, label in ((env, "candidate"), (ref, "reference")):
            try:
                state = obj.get_state()
                context[label]["state_at_failure"] = _diagnostic_value(state)
                goal = "goal" if domain == "simple_point_maze" else "target"
                context[label]["goal"] = _diagnostic_value(state[goal, "position"])
            except Exception as snapshot_error:
                context[label]["snapshot_error"] = repr(snapshot_error)
        context["candidate_variation"] = {
            path: _diagnostic_value(env._style[path]) for path in env._style.paths
        }
        print("NAVIGATION_FAILURE_CONTEXT " + json.dumps(context), flush=True)
        raise
    finally:
        for obj, method, prior, had_instance_method in reversed(saved):
            if had_instance_method:
                setattr(obj, method, prior)
            else:
                delattr(obj, method)
        _DIAGNOSTIC_CONTEXT = previous


def _actions(domain, seed, steps, batch):
    bound = 0.4 if domain == "simple_point_maze" else 1.5
    return np.random.default_rng(seed).uniform(-bound, bound, (steps, batch, 2)).astype(np.float32)


def _assert_independent_snapshot(env, snapshot):
    expected = snapshot.clone()
    for key in snapshot.keys(True, True):
        if snapshot[key].is_floating_point():
            snapshot[key].add_(1)
    compare_state(env.get_state(), expected)


def _assert_lossy_rejected(env, snapshot):
    packed = env._pack_render_state(snapshot)
    for bad in (packed, env._unpack_render_state(packed)):
        try:
            env.set_state(bad)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("Lossy renderer projection was accepted")


def _render_contract(env):
    state = env.get_state()
    before = state.clone()
    frame = env.render(state)
    assert torch.equal(frame, env._render(env._pack_render_state(state)))
    compare_state(env.get_state(), before)
    return {"shape": list(frame.shape), "full_state_private_projection_exact": True, "non_mutating": True}


def _reset_options(domain, device):
    if domain == "simple_point_maze":
        options = {"variation": []}
    else:
        options = {
            "variation": [],
            "state": np.asarray([31.25, 47.5], np.float32),
            "target_state": np.asarray([181.5, 164.25], np.float32),
        }
    ref = NativeNavigationReference(domain, 1, options=options)
    with se.make(CASES[domain]["stable_env"], num_envs=1, device=device) as env:
        try:
            with _diagnostic_case(env, ref, domain, "_reset_options"):
                actual = env.reset(seed=17, options=options)
                expected = ref.reset(seed=17)
                compare_state(actual, expected)
                return {
                    "variation_empty": True,
                    "explicit_state_and_target": domain != "simple_point_maze",
                    "default_reward_goal": domain == "simple_point_maze",
                }
        finally:
            ref.close()


def _variation_case(domain, device):
    if domain == "two_room":
        values = {
            "agent.speed": np.asarray([6.5], np.float32),
            "agent.radius": np.asarray([9.0], np.float32),
            "wall.thickness": np.asarray(14, np.int64),
            "wall.axis": np.asarray(0, np.int64),
            "door.number": np.asarray(2, np.int64),
            "door.position": np.asarray([49.0, 160.0, 49.0], np.float32),
            "door.size": np.asarray([14.0, 18.0, 14.0], np.float32),
        }
    elif domain == "piecewise":
        values = {
            "agent.speed": np.asarray([7.25], np.float32),
            "agent.radius": np.asarray([9.0], np.float32),
            "zones.bias_0": np.asarray([3.5, -1.25], np.float32),
            "zones.bias_1": np.asarray([-0.5, 3.0], np.float32),
            "zones.bias_2": np.asarray([-3.0, 0.75], np.float32),
            "zones.bias_3": np.asarray([1.0, -3.5], np.float32),
        }
    else:
        values = {
            "agent.speed": np.asarray([1.75], np.float32),
            "agent.radius": np.asarray([0.15], np.float32),
            "goal.radius": np.asarray([0.25], np.float32),
            "walls.number": np.asarray(4, np.int64),
            "walls.shape": np.asarray([[0.5, 0.5]] * 6, np.float32),
            "walls.positions": np.asarray(
                [[1.0, 1.0], [2.0, 1.0], [1.0, 3.0], [3.0, 3.0], [3.5, 1.0], [0.5, 3.5]],
                np.float32,
            ),
        }
    ref = NativeNavigationReference(domain, 1, persistent_values=values)
    with se.make(CASES[domain]["stable_env"], num_envs=1, device=device) as env:
        try:
            with _diagnostic_case(env, ref, domain, "_variation_case"):
                candidate_values = {
                    key: np.asarray(value).reshape(-1) if np.asarray(value).ndim > 1 else value
                    for key, value in values.items()
                }
                env.set_variation(candidate_values)
                check_reset_matches_reference(env, ref, seed=11)
                actions = _actions(domain, 219, 32, 1)
                _probe("trajectory")
                steps = check_step_matches_reference(env, ref, actions)
                varied_final = env.get_state()["agent", "position"].clone()
                with se.make(CASES[domain]["stable_env"], num_envs=1, device=device) as default_env:
                    default_env.reset(seed=11)
                    for action in actions:
                        default_env.step(action, frames=None)
                    assert not torch.equal(varied_final, default_env.get_state()["agent", "position"])
                return {
                    "values": {k: np.asarray(v).tolist() for k, v in values.items()},
                    "steps": len(steps),
                    "differs_from_same_seed_default": True,
                }
        finally:
            ref.close()


def _boundary_case(domain, device):
    ref = NativeNavigationReference(domain, 1)
    with se.make(CASES[domain]["stable_env"], num_envs=1, device=device) as env:
        try:
            with _diagnostic_case(env, ref, domain, "_boundary_case"):
                check_reset_matches_reference(env, ref, seed=5)
                if domain == "two_room":
                    probes = [
                        ([99.0, 80.0], [0.2, 0.0], "wall_block"),
                        ([99.0, 49.0], [1.0, 0.0], "door_passage"),
                    ]
                elif domain == "piecewise":
                    probes = [
                        ([111.99999, 50.0], [0.0, 0.0], "zone_left"),
                        ([112.0, 50.0], [0.0, 0.0], "zone_boundary_right"),
                    ]
                else:
                    shape = env._style["walls.shape"][0].detach().cpu().numpy().reshape(6, 2)[0]
                    corner = env._style["walls.positions"][0].detach().cpu().numpy().reshape(6, 2)[0]
                    radius = float(env._style["agent.radius"][0].item())
                    touch = [float(corner[0] - radius), float(corner[1] + shape[1] / 2)]
                    probes = [([touch[0] - 0.1, touch[1]], [0.1, 0.0], "inclusive_wall_touch")]
                observed = {}
                for position, action, name in probes:
                    _probe(name)
                    state = ref.get_state()
                    state["agent", "position"] = torch.tensor([position], dtype=torch.float32)
                    if domain != "simple_point_maze":
                        state["target", "position"] = torch.tensor([[190.0, 190.0]], dtype=torch.float32)
                    env.set_state(state)
                    ref.set_state(state)
                    got, expected = env.step(np.asarray([action], np.float32), frames=None), ref.step(np.asarray([action], np.float32))
                    compare_state(got, expected)
                    observed[name] = got["state", "agent", "position"][0].detach().cpu().tolist()
                return observed
        finally:
            ref.close()


def _piecewise_zone_boundaries(device):
    """Check both internal zone planes and their intersection with source continuation."""
    middle = np.float32(112.0)
    neighbours = (
        np.nextafter(middle, np.float32(-np.inf)),
        middle,
        np.nextafter(middle, np.float32(np.inf)),
    )
    coordinates = (np.float32(50.0), *neighbours, np.float32(174.0))
    positions = [(x, y) for x in coordinates for y in coordinates if x in neighbours or y in neighbours]
    cases = []
    for batch in (1, 3):
        ref = NativeNavigationReference("piecewise", batch)
        with se.make("Piecewise", num_envs=batch, device=device) as env:
            try:
                with _diagnostic_case(env, ref, "piecewise", "all_zone_boundaries"):
                    check_reset_matches_reference(env, ref, seed=5)
                    for index in range(len(positions)):
                        rows = [positions[(index + lane) % len(positions)] for lane in range(batch)]
                        _probe(f"zone_planes_{index}_positions_{rows}")
                        state = ref.get_state()
                        state["agent", "position"] = torch.tensor(rows, dtype=torch.float32)
                        state["target", "position"] = torch.full((batch, 2), 190.0, dtype=torch.float32)
                        env.set_state(state)
                        ref.set_state(state)
                        compare_state(env.get_state(), ref.get_state())
                        actions = _actions("piecewise", 701 + index, 8, batch)
                        actions[0].fill(0)
                        steps = check_step_matches_reference(env, ref, actions)
                        cases.append({
                            "batch": batch,
                            "seed": 5,
                            "positions": np.asarray(rows).tolist(),
                            "steps": len(steps),
                            "first_action_zero": True,
                            "max_error": 0.0,
                        })
            finally:
                ref.close()
    return {
        "internal_boundaries": {"x": 112.0, "y": 112.0},
        "float32_boundary_neighbours": [float(x) for x in neighbours],
        "covers_both_sides_and_intersection": True,
        "cases": cases,
    }


def _reward_and_lifecycle(domain, device):
    ref = NativeNavigationReference(domain, 1)
    with se.make(CASES[domain]["stable_env"], num_envs=1, device=device) as env:
        try:
            with _diagnostic_case(env, ref, domain, "_reward_and_lifecycle"):
                check_reset_matches_reference(env, ref, seed=71)
                state = ref.get_state()
                if domain == "simple_point_maze":
                    state["goal", "position"] = torch.tensor([[4.0, 4.0]])
                    state["agent", "position"] = torch.tensor([[4.0, 4.0]])
                else:
                    boundary = state.clone()
                    boundary["target", "position"] = torch.tensor([[80.0, 80.0]])
                    boundary["agent", "position"] = torch.tensor([[64.0, 80.0]])
                    hold = np.asarray([[-0.4, 0.0]], np.float32) if domain == "piecewise" else np.zeros((1, 2), np.float32)
                    env.set_state(boundary)
                    ref.set_state(boundary)
                    got, expected = env.step(hold, frames=None), ref.step(hold)
                    compare_state(got, expected)
                    assert not bool(got["terminated"].item())

                    inside = boundary.clone()
                    inside["agent", "position"] = torch.tensor(
                        [[np.nextafter(np.float32(64.0), np.float32(80.0)), 80.0]], dtype=torch.float32
                    )
                    env.set_state(inside)
                    ref.set_state(inside)
                    got, expected = env.step(hold, frames=None), ref.step(hold)
                    compare_state(got, expected)
                    assert bool(got["terminated"].item())
                    state["target", "position"] = torch.tensor([[80.0, 80.0]])
                    state["agent", "position"] = torch.tensor([[80.0, 80.0]])
                env.set_state(state)
                ref.set_state(state)
                zero = np.zeros((1, 2), np.float32)
                got, expected = env.step(zero, frames=None), ref.step(zero)
                compare_state(got, expected)
                assert bool(got["terminated"].item())
                expected_reward = 1.0 if domain == "simple_point_maze" else 0.0
                assert float(got["reward"].item()) == expected_reward

                # None of these source envs owns a horizon or elapsed-step counter.
                for _ in range(256):
                    got, expected = env.step(zero, frames=None), ref.step(zero)
                    compare_state(got, expected)
                    assert not bool(got["truncated"].item())
                return {
                    "success_reward": expected_reward,
                    "strict_distance_boundary": domain != "simple_point_maze",
                    "steps_without_truncation": 257,
                    "horizon": None,
                }
        finally:
            ref.close()


def _simple_goal_distinction(device):
    ref = NativeNavigationReference("simple_point_maze", 1)
    with se.make("SimplePointMaze", num_envs=1, device=device) as env:
        try:
            with _diagnostic_case(env, ref, "simple_point_maze", "_simple_goal_distinction"):
                check_reset_matches_reference(env, ref, seed=0)
                state = env.get_state()
                cached = state["task", "cached_goal_agent_position"]
                reward_goal = state["goal", "position"]
                assert not torch.equal(cached, reward_goal)
                got, expected = env.step(np.zeros((1, 2), np.float32), frames=None), ref.step(np.zeros((1, 2), np.float32))
                compare_state(got, expected)
                assert torch.equal(got["info", "goal"].cpu(), expected["info", "goal"])
                return {
                    "cached_agent_position": cached[0].cpu().tolist(),
                    "reward_goal_position": reward_goal[0].cpu().tolist(),
                    "cached_goal_image_byte_exact": True,
                }
        finally:
            ref.close()


def _field_necessity(domain, device):
    ref = NativeNavigationReference(domain, 1)
    with (
        se.make(CASES[domain]["stable_env"], num_envs=1, device=device) as env,
        se.make(CASES[domain]["stable_env"], num_envs=1, device=device) as other,
    ):
        try:
            with _diagnostic_case(env, ref, domain, "_field_necessity"):
                check_reset_matches_reference(env, ref, seed=37)
                snapshot = env.get_state()
                expected = ref.step(np.zeros((1, 2), np.float32))
                other_state = other.reset(seed=991)
                wrong = snapshot.clone()
                if domain == "simple_point_maze":
                    wrong["task", "cached_goal_agent_position"] = other_state[
                        "task", "cached_goal_agent_position"
                    ].clone()
                else:
                    wrong["target", "position"] = other_state["target", "position"].clone()
                env.set_state(wrong)
                omitted = env.step(np.zeros((1, 2), np.float32), frames=None)
                if domain == "simple_point_maze":
                    assert not torch.equal(omitted["info", "goal"].cpu(), expected["info", "goal"])
                    field = "task.cached_goal_agent_position"
                else:
                    assert not torch.equal(
                        omitted["info", "distance_to_target"].cpu(), expected["info", "distance_to_target"]
                    )
                    field = "target.position"
                env.set_state(snapshot)
                compare_state(env.step(np.zeros((1, 2), np.float32), frames=None), expected)
                return {"field": field, "omission_changes_next_output": True, "complete_restore_exact": True}
        finally:
            ref.close()


def _simple_action_dtypes(device):
    ref = NativeNavigationReference("simple_point_maze", 1)
    with se.make("SimplePointMaze", num_envs=1, device=device) as env:
        try:
            with _diagnostic_case(env, ref, "simple_point_maze", "_simple_action_dtypes"):
                check_reset_matches_reference(env, ref, seed=17)
                initial = env.get_state()
                observed = {}
                for dtype in (np.float16, np.float32, np.float64):
                    env.set_state(initial)
                    ref.reset(seed=17)
                    action = np.asarray([[0.1234567890123, -0.07]], dtype=dtype)
                    got, expected = env.step(action, frames=None), ref.step(action)
                    compare_state(got, expected)
                    observed[np.dtype(dtype).name] = {
                        "state_is_float64": bool(got["state", "task", "state_is_float64"].item()),
                        "observation_dtype": str(got["observation"].dtype),
                    }

                promoted = env.get_state()
                continuation_action = np.asarray([[0.031234567, -0.017]], np.float32)
                expected = ref.step(continuation_action)
                wrong_flag = promoted.clone()
                wrong_flag["task", "state_is_float64"].zero_()
                env.set_state(wrong_flag)
                omitted = env.step(continuation_action, frames=None)
                assert not torch.equal(
                    omitted["state", "agent", "position"].cpu(), expected["state", "agent", "position"]
                )
                env.set_state(promoted)
                compare_state(env.step(continuation_action, frames=None), expected)

                # A proposed float64 move that touches a wall is discarded by source assignment
                # ``next_state = self.state`` and therefore must retain the prior float32 dtype.
                corner = env._style["walls.positions"][0].reshape(6, 2)[0]
                shape = env._style["walls.shape"][0].reshape(6, 2)[0]
                radius = float(env._style["agent.radius"][0].item())
                blocked = initial.clone()
                blocked["agent", "position"] = torch.tensor(
                    [[float(corner[0]) - radius - 0.1, float(corner[1] + shape[1] / 2)]], dtype=torch.float64
                )
                blocked["task", "state_is_float64"].zero_()
                env.set_state(blocked)
                ref.set_state(blocked)
                got, expected = env.step(np.asarray([[0.1, 0.0]], np.float64), frames=None), ref.step(
                    np.asarray([[0.1, 0.0]], np.float64)
                )
                compare_state(got, expected)
                assert not bool(got["state", "task", "state_is_float64"].item())
                follow, expected_follow = env.step(np.asarray([[0.0, 0.0]], np.float32), frames=None), ref.step(
                    np.asarray([[0.0, 0.0]], np.float32)
                )
                compare_state(follow, expected_follow)
                return {
                    "accepted": observed,
                    "dtype_flag_omission_changes_continuation": True,
                    "blocked_float64_retains_float32": True,
                    "restored_continuation": True,
                }
        finally:
            ref.close()


def _simple_reward_boundary(device):
    values = {"goal.radius": np.asarray([0.25], np.float32)}
    ref = NativeNavigationReference("simple_point_maze", 1, persistent_values=values)
    with se.make("SimplePointMaze", num_envs=1, device=device) as env:
        try:
            with _diagnostic_case(env, ref, "simple_point_maze", "_simple_reward_boundary"):
                env.set_variation(values)
                check_reset_matches_reference(env, ref, seed=19)
                state = env.get_state()
                state["goal", "position"] = torch.tensor([[4.0, 4.0]])
                state["agent", "position"] = torch.tensor([[3.75, 4.0]], dtype=torch.float64)
                state["task", "state_is_float64"].zero_()
                env.set_state(state)
                ref.set_state(state)
                got, expected = env.step(np.zeros((1, 2), np.float32), frames=None), ref.step(np.zeros((1, 2), np.float32))
                compare_state(got, expected)
                assert not bool(got["terminated"].item())

                inside = state.clone()
                inside_x = np.nextafter(np.float32(3.75), np.float32(4.0))
                inside["agent", "position"] = torch.tensor([[inside_x, 4.0]], dtype=torch.float64)
                env.set_state(inside)
                ref.set_state(inside)
                got, expected = env.step(np.zeros((1, 2), np.float32), frames=None), ref.step(np.zeros((1, 2), np.float32))
                compare_state(got, expected)
                assert bool(got["terminated"].item()) and float(got["reward"].item()) == 1.0

                state64 = state.clone()
                state64["task", "state_is_float64"].fill_(True)
                env.set_state(state64)
                ref.set_state(state64)
                got, expected = env.step(np.zeros((1, 2), np.float64), frames=None), ref.step(np.zeros((1, 2), np.float64))
                compare_state(got, expected)
                assert not bool(got["terminated"].item())
                inside64 = state64.clone()
                inside64["agent", "position"] = torch.tensor(
                    [[np.nextafter(np.float64(3.75), np.float64(4.0)), 4.0]], dtype=torch.float64
                )
                env.set_state(inside64)
                ref.set_state(inside64)
                got, expected = env.step(np.zeros((1, 2), np.float64), frames=None), ref.step(np.zeros((1, 2), np.float64))
                compare_state(got, expected)
                assert bool(got["terminated"].item())
                return {
                    "radius": 0.25,
                    "float32_equal_is_terminal": False,
                    "float32_nextafter_inside_is_terminal": True,
                    "float64_equal_is_terminal": False,
                    "float64_nextafter_inside_is_terminal": True,
                }
        finally:
            ref.close()


def run_gate(device="cpu"):
    import stable_worldmodel

    source = Path(stable_worldmodel.__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert commit == SWM_PIN, (commit, SWM_PIN)
    assert not subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--"], text=True)
    evidence = {
        "swm_commit": commit,
        "swm_source": str(source),
        "device": device,
        "execution_mode": "NATIVE_TORCH",
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "gymnasium", "tensordict", "stable-worldmodel", "matplotlib")
        },
        "mappings": CASES,
        "scope_limit": (
            "Only empty-kwargs TwoRoom-v1, Piecewise-v0/grid_n=2 and SimplePointMaze-v0/show_goal=True. "
            "TwoRoomTarget, PiecewiseZones, nondefault task surfaces and full visual variation are excluded."
        ),
        "rng_scope": {
            "seeded_reset": "source-shaped Gymnasium Dict.seed routing, one seed+i per independent lane",
            "step_rng": "none in pinned source",
            "seed_none": (
                "Pinned SWM calls variation_space.seed(None) on every reset, drawing fresh external entropy; "
                "State therefore makes no restored-unseeded-reset replay claim."
            ),
        },
        "cases": [],
        "variation": {},
        "boundaries": {},
        "rewards_lifecycle": {},
        "reset_options": {},
        "field_necessity": {},
    }
    for domain, case in CASES.items():
        for batch in (1, 3):
            ref = NativeNavigationReference(domain, batch)
            with (
                se.make(case["stable_env"], num_envs=batch, device=device) as env,
                se.make(case["stable_env"], num_envs=batch, device=device) as other,
            ):
                try:
                    with _diagnostic_case(env, ref, domain, "run_gate"):
                        assert env.execution_mode == "NATIVE_TORCH"
                        for seed in (0, 17, 42):
                            check_reset_matches_reference(env, ref, seed=seed)
                            snapshot = env.get_state()
                            _assert_independent_snapshot(env, snapshot)
                            actions = _actions(domain, seed + 200, 32, batch)
                            _probe("roundtrip")
                            check_state_roundtrip(env, actions=actions[:4])
                            compare_state(env.get_state(), ref.get_state())
                            _probe("trajectory")
                            steps = check_step_matches_reference(env, ref, actions)

                            other.reset(seed=seed + 991)
                            for action in _actions(domain, seed + 991, 9, batch):
                                other.step(action, frames=None)
                            resume = env.get_state()
                            _probe("different_history")
                            history = check_two_histories_same_state(env, other, resume, actions[:8])
                            _probe("render")
                            render = _render_contract(env)
                            _assert_lossy_rejected(env, resume)
                            evidence["cases"].append(
                                {
                                    "domain": domain,
                                    "batch": batch,
                                    "seed": seed,
                                    "steps": len(steps),
                                    "history_steps": len(history),
                                    "max_error": 0.0,
                                    "render": render,
                                }
                            )
                finally:
                    ref.close()
        evidence["variation"][domain] = _variation_case(domain, device)
        evidence["boundaries"][domain] = _boundary_case(domain, device)
        evidence["rewards_lifecycle"][domain] = _reward_and_lifecycle(domain, device)
        evidence["reset_options"][domain] = _reset_options(domain, device)
        evidence["field_necessity"][domain] = _field_necessity(domain, device)
    evidence["simple_point_maze_goal_distinction"] = _simple_goal_distinction(device)
    evidence["simple_point_maze_action_dtypes"] = _simple_action_dtypes(device)
    evidence["simple_point_maze_reward_boundary"] = _simple_reward_boundary(device)
    evidence["piecewise_zone_boundaries"] = _piecewise_zone_boundaries(device)
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = run_gate(args.device)
    payload = json.dumps(evidence, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
