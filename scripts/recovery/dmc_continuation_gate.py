"""Gate the bounded Cheetah run, Hopper hop and Walker walk recovery."""

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
from tests.helpers.reference.dmc_semantics import DMCReference as DMCContinuationReference
from tests.helpers.reference.semantics import (
    check_reset_matches_reference,
    check_state_roundtrip,
    check_step_matches_reference,
    check_two_histories_same_state,
    compare_state,
)

SWM_PIN = "176d54532b3c0a251a4f509b74f2fdacc143a63d"
CASES = {
    "cheetah": "DMCCheetahRun",
    "hopper": "DMCHopperHop",
    "walker": "DMCWalkerWalk",
}
POSITIVE_SPEED = {"cheetah": 15.0, "hopper": 3.0, "walker": 1.0}


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
            raise AssertionError("Lossy renderer projection was accepted")


def _positive_state(domain, state):
    state = state.clone()
    state["qpos"].zero_()
    state["qvel"].zero_()
    state["qvel"][:, 0] = POSITIVE_SPEED[domain]
    state["time"].zero_()
    state["lifecycle", "step_count"].zero_()
    state["lifecycle", "reset_next_step"].zero_()
    state["task", "cumulative_reward"].zero_()
    return state


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
        if left.shape != right.shape or left.dtype.kind not in "biufc":
            continue
        assert np.array_equal(left, right, equal_nan=True), name
        compared += 1
    assert compared >= 500
    candidate_base, reference_base = candidate._simulator.envs[0]._env, reference.envs[0].env._env
    assert candidate_base._n_sub_steps == reference_base._n_sub_steps
    assert candidate_base._step_limit == reference_base._step_limit
    assert candidate_base.control_timestep() == reference_base.control_timestep()
    return compared


def _assert_bad_actions_rejected(env, action_dim):
    for action in (
        np.zeros((env.B, action_dim + 1)),
        np.full((env.B, action_dim), np.nan),
        np.full((env.B, action_dim), np.inf),
    ):
        try:
            env.step(action, frames=None)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid action was accepted")


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
            delta = float((left.to(torch.float64) - right.to(torch.float64)).abs().nan_to_num().max())
        error = max(error, delta)
    return error


def _warmstart_experiment(domain, env, reference):
    """Candidate-only restore against an uninterrupted reference, with warmstart omitted."""
    seed = 0
    prefix_steps = {"cheetah": 5, "hopper": 5, "walker": 2}[domain]
    actions = np.random.default_rng(seed + 200).uniform(-1, 1, (32, 1, reference.action_dim))
    compare_state(env.reset(seed=seed), reference.reset(seed=seed))
    check_step_matches_reference(env, reference, actions[:prefix_steps])
    saved = env.get_state()
    for action in -actions[prefix_steps : prefix_steps + 9]:
        env.step(action, frames=None)
    omitted = saved.clone()
    omitted["warmstart"] = env.get_state()["warmstart"]
    env.set_state(omitted)
    continuation = actions[prefix_steps : prefix_steps + 8]
    bad_outputs = [env.step(action, frames=None) for action in continuation]
    expected_outputs = [reference.step(action) for action in continuation]
    error_without = max(
        _max_error(actual, expected, exclude=(("state", "warmstart"),))
        for actual, expected in zip(bad_outputs, expected_outputs, strict=True)
    )
    env.set_state(saved)
    good_outputs = [env.step(action, frames=None) for action in continuation]
    error_with_schema = max(
        _max_error(actual, expected)
        for actual, expected in zip(good_outputs, expected_outputs, strict=True)
    )
    assert error_without > 0
    assert error_with_schema == 0
    return {
        "seed": seed,
        "history_steps_before_snapshot": prefix_steps,
        "different_history_steps": 9,
        "continuation_steps": 8,
        "max_observable_error_without_warmstart": error_without,
        "max_error_with_schema": error_with_schema,
    }


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
        "mappings": CASES,
        "task_identity": {
            "cheetah": "CustomCheetah(goal='run', move_speed=10): default branch delegates to Cheetah.get_reward",
            "hopper": "CustomHopper(goal='hop', hopping=True): default branch delegates to Hopper.get_reward",
            "walker": "CustomPlanarWalker(goal='walk', move_speed=1): default branch delegates to PlanarWalker.get_reward",
        },
        "state_schema": {
            "common": [
                "qpos", "qvel", "time", "lifecycle counters", "task.cumulative_reward", "MT19937 state"
            ],
            "domain_specific": {
                "cheetah": ["qacc_warmstart"],
                "hopper": ["qacc_warmstart"],
                "walker": ["qacc_warmstart"],
            },
            "omitted": {
                "ctrl": "different histories retain different values; the next public action overwrites control",
                "qacc": "derived by forward/step and exact without restoration",
                "act": "all three models have na=0",
                "task_cache": "goal/speed/hopping are persistent task configuration; timeout_progress is reset-only and unread",
            },
        },
        "cases": [],
    }

    for domain, env_name in CASES.items():
        for batch in (1, 3):
            ref = DMCContinuationReference(domain, batch)
            with (
                se.make(env_name, num_envs=batch, device=device) as env,
                se.make(env_name, num_envs=batch, device=device) as other,
            ):
                try:
                    if batch == 1:
                        check_reset_matches_reference(env, ref, seed=0)
                        arrays = _compare_models(env, ref)
                        evidence.setdefault("construction", {})[domain] = {
                            "numeric_model_arrays_exact": arrays,
                            "control_timestep": ref.envs[0].env._env.control_timestep(),
                            "dmc_step_limit": ref.step_limit,
                            "action_dim": ref.action_dim,
                        }
                        evidence.setdefault("warmstart_experiment", {})[domain] = _warmstart_experiment(
                            domain, env, ref
                        )
                    for seed in (0, 17, 42):
                        check_reset_matches_reference(env, ref, seed=seed)
                        saved, live = env.get_state(), env.get_state()
                        saved["qpos"].add_(1)
                        saved["rng", "keys"].add_(1)
                        compare_state(env.get_state(), live)

                        actions = np.random.default_rng(seed + 200).uniform(
                            -1, 1, (32, batch, ref.action_dim)
                        )
                        check_state_roundtrip(env, actions=actions[:4])
                        compare_state(env.get_state(), ref.get_state())
                        steps = check_step_matches_reference(env, ref, actions)

                        other.reset(seed=seed + 991)
                        for action in -actions[:9]:
                            other.step(action, frames=None)
                        snapshot = env.get_state()
                        env.set_state(snapshot)
                        other.set_state(snapshot)
                        warmstart_equal = np.array_equal(
                            np.stack([wrapper.physics.data.qacc_warmstart for wrapper in env._simulator.envs]),
                            np.stack([wrapper.physics.data.qacc_warmstart for wrapper in other._simulator.envs]),
                        )
                        control_diverged = not np.array_equal(
                            np.stack([wrapper.physics.data.ctrl for wrapper in env._simulator.envs]),
                            np.stack([wrapper.physics.data.ctrl for wrapper in other._simulator.envs]),
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
                                "domain": domain,
                                "batch": batch,
                                "seed": seed,
                                "reset": True,
                                "independent_snapshot": True,
                                "roundtrip": True,
                                "one_step": True,
                                "steps": len(steps),
                                "two_history_steps": len(history),
                                "warmstart_restored": warmstart_equal,
                                "control_omission_proved": control_diverged,
                                "full_state_render": True,
                                "max_state_error": max(max(step["state"].values()) for step in steps),
                                "max_output_error": 0.0,
                            }
                        )

                    check_reset_matches_reference(env, ref, seed=18)
                    reward_state = _positive_state(domain, ref.get_state())
                    env.set_state(reward_state)
                    ref.set_state(reward_state)
                    got, expected = env.step(np.zeros((batch, ref.action_dim)), frames=None), ref.step(
                        np.zeros((batch, ref.action_dim))
                    )
                    compare_state(got, expected)
                    assert (got["reward"] > 0).all()
                    evidence["cases"].append(
                        {
                            "domain": domain,
                            "batch": batch,
                            "positive_reward_min": float(got["reward"].min()),
                            "terminated": False,
                            "truncated": False,
                        }
                    )

                    check_reset_matches_reference(env, ref, seed=5)
                    public_at_last = ref.step_limit // 2
                    assert ref.step_limit % 2 == 0
                    boundary_actions = np.zeros((public_at_last + 2, batch, ref.action_dim))
                    check_step_matches_reference(env, ref, boundary_actions[: public_at_last - 1])
                    near_limit = env.get_state()
                    rng_before = near_limit.select("rng").clone()
                    check_step_matches_reference(env, ref, boundary_actions[public_at_last - 1 :])
                    after = env.get_state()
                    assert any(
                        not torch.equal(rng_before[key], after[key]) for key in rng_before.keys(True, True)
                    )
                    other.reset(seed=77)
                    check_two_histories_same_state(env, other, near_limit, boundary_actions[:4])
                    evidence["cases"].append(
                        {
                            "domain": domain,
                            "batch": batch,
                            "dmc_step_limit": ref.step_limit,
                            "action_repeat": 2,
                            "boundary_public_steps": len(boundary_actions),
                            "implicit_reset_rng_changes": True,
                            "lifecycle_rng_replay": True,
                        }
                    )

                    if batch == 1:
                        check_reset_matches_reference(env, ref, seed=31)
                        _assert_bad_actions_rejected(env, ref.action_dim)
                        check_reset_matches_reference(env, ref, seed=31)
                        outside = np.full((1, ref.action_dim), 1.1)
                        compare_state(env.step(outside, frames=None), ref.step(outside))
                        evidence.setdefault("action_validation", {})[domain] = {
                            "shape_and_nonfinite_rejected": True,
                            "finite_out_of_bounds_matches_historical_passthrough": True,
                        }
                finally:
                    ref.close()

        ref = DMCContinuationReference(domain, 1)
        with se.make(env_name, num_envs=1, device=device) as env:
            try:
                env.set_variation({"light.diffuse_scale": [1.1]})
                assert np.array_equal(
                    env._simulator.envs[0].physics.model.opt.gravity,
                    np.array([0.0, 0.0, -9.81], np.float64),
                )
                check_reset_matches_reference(env, ref, seed=23)
                actions = np.random.default_rng(23).uniform(-1, 1, (8, 1, ref.action_dim))
                check_step_matches_reference(env, ref, actions)
                evidence.setdefault("visual_only_variation", {})[domain] = {
                    "nominal_gravity_exact": True,
                    "steps": 8,
                    "max_error": 0.0,
                }
            finally:
                ref.close()

        options = {
            "variation_values": {
                "gravity.x": np.array([1.25], np.float64),
                "gravity.y": np.array([-0.75], np.float64),
                "gravity.z": np.array([-7.5], np.float64),
            }
        }
        ref = DMCContinuationReference(domain, 1, options=options)
        with se.make(env_name, num_envs=1, device=device) as env:
            try:
                env.set_variation({"gravity.x": [1.25], "gravity.y": [-0.75], "gravity.z": [-7.5]})
                check_reset_matches_reference(env, ref, seed=11)
                actions = np.random.default_rng(11).uniform(-1, 1, (32, 1, ref.action_dim))
                check_step_matches_reference(env, ref, actions)
                evidence.setdefault("physical_variation", {})[domain] = {
                    "gravity": [1.25, -0.75, -7.5],
                    "steps": 32,
                    "max_error": 0.0,
                }
            finally:
                ref.close()
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_gate(args.device)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
