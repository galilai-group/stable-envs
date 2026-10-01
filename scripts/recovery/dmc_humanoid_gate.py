"""Gate historical ``swm/HumanoidDMControl-v0`` as Humanoid Walk."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

import stable_envs as se
from tests.helpers.reference.dmc_humanoid_semantics import DMCHumanoidReference
from tests.helpers.reference.semantics import (
    check_reset_matches_reference,
    check_state_roundtrip,
    check_step_matches_reference,
    check_two_histories_same_state,
    compare_state,
)

SWM_PIN = "176d54532b3c0a251a4f509b74f2fdacc143a63d"
ENV_NAME = "DMCHumanoidWalk"


def _max_error(actual, expected, *, exclude=()):
    assert set(actual.keys(True, True)) == set(expected.keys(True, True))
    error = 0.0
    for key in actual.keys(True, True):
        if key in exclude:
            continue
        left, right = actual[key].detach().cpu(), expected[key].detach().cpu()
        assert left.dtype == right.dtype and left.shape == right.shape
        if left.dtype == torch.bool:
            delta = float((left != right).any())
        else:
            delta = float(
                (left.to(torch.float64) - right.to(torch.float64)).abs().nan_to_num().max()
            )
        error = max(error, delta)
    return error


def _compare_models(candidate, reference):
    actual = candidate._simulator.envs[0].physics.model
    expected = reference.envs[0].env.physics.model
    compared = 0
    for name in dir(actual):
        if name.startswith("_"):
            continue
        try:
            left, right = np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name))
        except Exception:
            continue
        if left.dtype.kind not in "biufc" or right.dtype.kind not in "biufc":
            continue
        assert left.shape == right.shape, (name, left.shape, right.shape)
        assert left.dtype == right.dtype, (name, left.dtype, right.dtype)
        assert np.array_equal(left, right, equal_nan=True), name
        compared += 1
    assert compared >= 580
    candidate_base = candidate._simulator.envs[0]._env
    reference_base = reference.envs[0].env._env
    assert candidate_base._n_sub_steps == reference_base._n_sub_steps == 5
    assert candidate_base._step_limit == reference_base._step_limit == 1000
    assert candidate_base.control_timestep() == reference_base.control_timestep() == 0.025
    return compared


def _assert_lossy_rows_rejected(env, snapshot):
    for bad in (
        env._pack_render_state(snapshot),
        env._unpack_render_state(env._pack_render_state(snapshot)),
    ):
        try:
            env.set_state(bad)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("Lossy Humanoid renderer projection was accepted")


def _assert_bad_actions_rejected(env):
    for action in (
        np.zeros((env.B, 22)),
        np.full((env.B, 21), np.nan),
        np.full((env.B, 21), np.inf),
    ):
        try:
            env.step(action, capture="states")
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid Humanoid action was accepted")


def _warmstart_experiment(env, reference):
    """Restore only the candidate; reference continuation remains uninterrupted."""
    actions = np.random.default_rng(200).uniform(-1, 1, (32, 1, 21))
    check_reset_matches_reference(env, reference, seed=0)
    check_step_matches_reference(env, reference, actions[:1])
    saved = env.get_state()
    saved_control = np.stack(
        [item.physics.data.ctrl.copy() for item in env._simulator.envs]
    )
    for action in -actions[1:10]:
        env.step(action, capture="states")
    divergent_warmstart = env.get_state()["warmstart"].clone()
    assert not torch.equal(saved["warmstart"], divergent_warmstart)
    divergent_control = np.stack(
        [item.physics.data.ctrl.copy() for item in env._simulator.envs]
    )
    control_diverged = not np.array_equal(saved_control, divergent_control)
    assert control_diverged, {
        "seed": 0,
        "history_steps_before_snapshot": 1,
        "different_history_steps": 9,
        "control_diverged": control_diverged,
    }

    omitted = saved.clone()
    omitted["warmstart"] = divergent_warmstart
    env.set_state(omitted)
    continuation = actions[1:9]
    bad_outputs = [env.step(action, capture="states") for action in continuation]
    expected_outputs = [reference.step(action) for action in continuation]
    error_without = max(
        _max_error(actual, expected, exclude=(("state", "warmstart"),))
        for actual, expected in zip(bad_outputs, expected_outputs, strict=True)
    )

    env.set_state(saved)
    good_outputs = [env.step(action, capture="states") for action in continuation]
    error_with = max(
        _max_error(actual, expected)
        for actual, expected in zip(good_outputs, expected_outputs, strict=True)
    )
    result = {
        "seed": 0,
        "history_steps_before_snapshot": 1,
        "different_history_steps": 9,
        "continuation_steps": 8,
        "max_observable_error_without_warmstart": error_without,
        "max_error_with_schema": error_with,
        "control_diverged_before_restore": control_diverged,
    }
    assert error_without > 0, result
    assert error_with == 0, result
    return result


def _physical_error(varied, nominal):
    return max(
        _max_error(varied, nominal),
        float((varied["reward"].detach().cpu() - nominal["reward"].detach().cpu()).abs().max()),
        float((varied["observation"].detach().cpu() - nominal["observation"].detach().cpu()).abs().max()),
    )


def run_gate(device="cpu"):
    import stable_worldmodel

    source = Path(stable_worldmodel.__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert commit == SWM_PIN, (commit, SWM_PIN)
    assert not subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--"], text=True)

    evidence = {
        "swm_commit": commit,
        "device": device,
        "execution_mode": "DELEGATED_CPU",
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "dm-control", "mujoco", "tensordict", "stable-worldmodel")
        },
        "mapping": {"swm/HumanoidDMControl-v0": ENV_NAME},
        "task_identity": "Humanoid(move_speed=1, pure_state=False), historical default task='walk'",
        "state_schema": {
            "included": [
                "qpos[28]",
                "qvel[27]",
                "qacc_warmstart[27]",
                "time",
                "lifecycle.step_count",
                "lifecycle.reset_next_step",
                "task.cumulative_reward",
                "MT19937 state",
            ],
            "omitted": {
                "ctrl": "different histories retain different control; next public action overwrites it",
                "qacc": "derived by forward/step; exact continuation without restoration",
                "act": "model na=0",
                "task_cache": "move_speed/pure_state are immutable task construction; no mutable task cache",
            },
        },
        "cases": [],
    }

    for batch in (1, 3):
        reference = DMCHumanoidReference(batch)
        with (
            se.make(ENV_NAME, num_envs=batch, device=device) as env,
            se.make(ENV_NAME, num_envs=batch, device=device) as other,
        ):
            try:
                if batch == 1:
                    check_reset_matches_reference(env, reference, seed=0)
                    evidence["construction"] = {
                        "numeric_model_arrays_exact": _compare_models(env, reference),
                        "nq": 28,
                        "nv": 27,
                        "nu": 21,
                        "na": 0,
                        "observation_width": 67,
                        "control_timestep": 0.025,
                        "physics_substeps": 5,
                        "dmc_step_limit": 1000,
                        "action_repeat": 2,
                        "floor_friction": env._simulator.envs[0].physics.model.geom_friction[0].tolist(),
                    }
                    evidence["warmstart_experiment"] = _warmstart_experiment(env, reference)

                for seed in (0, 17, 42):
                    check_reset_matches_reference(env, reference, seed=seed)
                    saved, live = env.get_state(), env.get_state()
                    saved["qpos"].add_(1)
                    saved["warmstart"].add_(1)
                    saved["rng", "keys"].add_(1)
                    compare_state(env.get_state(), live)

                    actions = np.random.default_rng(seed + 200).uniform(-1, 1, (32, batch, 21))
                    check_state_roundtrip(env, actions=actions[:4])
                    compare_state(env.get_state(), reference.get_state())
                    steps = check_step_matches_reference(env, reference, actions)

                    other.reset(seed=seed + 991)
                    for action in -actions[:9]:
                        other.step(action, capture="states")
                    snapshot = env.get_state()
                    env.set_state(snapshot)
                    other.set_state(snapshot)
                    warmstart_equal = np.array_equal(
                        np.stack([item.physics.data.qacc_warmstart for item in env._simulator.envs]),
                        np.stack([item.physics.data.qacc_warmstart for item in other._simulator.envs]),
                    )
                    control_diverged = not np.array_equal(
                        np.stack([item.physics.data.ctrl for item in env._simulator.envs]),
                        np.stack([item.physics.data.ctrl for item in other._simulator.envs]),
                    )
                    assert warmstart_equal and control_diverged
                    history = check_two_histories_same_state(env, other, snapshot, actions[:8])

                    frame = env.render(snapshot)
                    assert torch.equal(frame, env._render(env._pack_render_state(snapshot)))
                    before = env.get_state()
                    env.render(snapshot)
                    compare_state(env.get_state(), before)
                    _assert_lossy_rows_rejected(env, snapshot)
                    evidence["cases"].append(
                        {
                            "batch": batch,
                            "seed": seed,
                            "reset": True,
                            "independent_snapshot": True,
                            "roundtrip": True,
                            "steps": len(steps),
                            "two_history_steps": len(history),
                            "warmstart_restored": warmstart_equal,
                            "control_omission_proved": control_diverged,
                            "full_state_render": True,
                            "max_state_error": max(max(step["state"].values()) for step in steps),
                            "max_output_error": 0.0,
                        }
                    )

                check_reset_matches_reference(env, reference, seed=18)
                zeros = np.zeros((batch, 21))
                got, expected = env.step(zeros, capture="states"), reference.step(zeros)
                compare_state(got, expected)
                assert (got["reward"] > 0).all()
                evidence["cases"].append(
                    {
                        "batch": batch,
                        "positive_reward_min": float(got["reward"].min()),
                        "terminated": bool(got["terminated"].any()),
                        "truncated": bool(got["truncated"].any()),
                    }
                )

                check_reset_matches_reference(env, reference, seed=5)
                checkpoints = {}
                rng_before = env.get_state().select("rng").clone()
                boundary_state = None
                for public_step in range(1, 503):
                    got, expected = env.step(zeros, capture="states"), reference.step(zeros)
                    compare_state(got, expected)
                    if public_step in (499, 500, 501, 502):
                        state = got["state"]
                        if public_step == 499:
                            boundary_state = state.clone()
                        checkpoints[str(public_step)] = {
                            "step_count": int(state["lifecycle", "step_count"][0].item()),
                            "reset_next_step": bool(
                                state["lifecycle", "reset_next_step"][0].item()
                            ),
                            "time": float(state["time"][0].item()),
                        }
                after = env.get_state()
                assert any(
                    not torch.equal(rng_before[key], after[key]) for key in rng_before.keys(True, True)
                )
                assert checkpoints["500"]["reset_next_step"]
                assert checkpoints["501"]["step_count"] == 1
                assert boundary_state is not None
                other.reset(seed=77)
                boundary_rng = boundary_state.select("rng").clone()
                boundary_history = check_two_histories_same_state(
                    env, other, boundary_state, np.zeros((4, batch, 21))
                )
                replayed = env.get_state()
                replay_rng_changed = any(
                    not torch.equal(boundary_rng[key], replayed[key])
                    for key in boundary_rng.keys(True, True)
                )
                assert replay_rng_changed
                evidence["cases"].append(
                    {
                        "batch": batch,
                        "dmc_step_limit": 1000,
                        "boundary_public_steps": 502,
                        "implicit_reset_rng_changes": True,
                        "lifecycle_rng_replay": len(boundary_history) == 4,
                        "replayed_boundary_rng_changes": replay_rng_changed,
                        "checkpoints": checkpoints,
                    }
                )

                if batch == 1:
                    check_reset_matches_reference(env, reference, seed=31)
                    _assert_bad_actions_rejected(env)
                    check_reset_matches_reference(env, reference, seed=31)
                    outside = np.full((1, 21), 1.1)
                    compare_state(env.step(outside, capture="states"), reference.step(outside))
                    evidence["action_validation"] = {
                        "shape_and_nonfinite_rejected": True,
                        "finite_out_of_bounds_matches_historical_passthrough": True,
                    }
            finally:
                reference.close()

    nominal = DMCHumanoidReference(1)
    with se.make(ENV_NAME, num_envs=1, device=device) as env:
        try:
            env.set_variation({"light.diffuse_scale": [1.1]})
            assert np.array_equal(
                env._simulator.envs[0].physics.model.opt.gravity,
                np.array([0.0, 0.0, -9.81], np.float64),
            )
            check_reset_matches_reference(env, nominal, seed=23)
            actions = np.random.default_rng(23).uniform(-1, 1, (8, 1, 21))
            check_step_matches_reference(env, nominal, actions)
            evidence["visual_only_variation"] = {
                "nominal_gravity_exact": True,
                "steps": 8,
                "max_error": 0.0,
            }
        finally:
            nominal.close()

    values = {"gravity.x": [1.25], "gravity.y": [-0.75], "gravity.z": [-7.5]}
    options = {"variation_values": {key: np.asarray(value, np.float64) for key, value in values.items()}}
    reference = DMCHumanoidReference(1, options=options)
    nominal = DMCHumanoidReference(1)
    with se.make(ENV_NAME, num_envs=1, device=device) as env:
        try:
            env.set_variation(values)
            check_reset_matches_reference(env, reference, seed=11)
            nominal.reset(seed=11)
            actions = np.random.default_rng(11).uniform(-1, 1, (32, 1, 21))
            varied_outputs, nominal_outputs = [], []
            for action in actions:
                got, expected = env.step(action, capture="states"), reference.step(action)
                compare_state(got, expected)
                varied_outputs.append(got)
                nominal_outputs.append(nominal.step(action, capture="states"))
            divergence = max(
                _physical_error(varied, default)
                for varied, default in zip(varied_outputs, nominal_outputs, strict=True)
            )
            assert divergence > 0
            evidence["physical_variation"] = {
                "gravity": [1.25, -0.75, -7.5],
                "steps": 32,
                "max_reference_error": 0.0,
                "max_default_divergence": divergence,
            }
        finally:
            reference.close()
            nominal.close()
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_gate(args.device)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
