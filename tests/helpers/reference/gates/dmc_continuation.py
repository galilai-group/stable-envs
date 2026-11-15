"""Semantic gate for DMC Cheetah run, Hopper hop and Walker walk against pinned stable-worldmodel."""

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
from tests.helpers.reference.dmc_semantics import compare_models
from tests.helpers.reference.saved import SWM_PIN
from tests.helpers.reference.semantics import (
    assert_lossy_projection_rejected,
    check_reset_matches_reference,
    check_state_roundtrip,
    check_step_matches_reference,
    check_two_histories_same_state,
    compare_state,
    max_state_error,
)

CASES = {
    "cheetah": "DMCCheetahRun",
    "hopper": "DMCHopperHop",
    "walker": "DMCWalkerWalk",
}
POSITIVE_SPEED = {"cheetah": 15.0, "hopper": 3.0, "walker": 1.0}


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


def _assert_bad_actions_rejected(env, action_dim):
    for action in (
        np.zeros((env.B, action_dim + 1)),
        np.full((env.B, action_dim), np.nan),
        np.full((env.B, action_dim), np.inf),
    ):
        try:
            env.step(action, states=None, frames=None).exclude("valid")
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid action was accepted")


def _warmstart_experiment(domain, env, reference):
    """Candidate-only restore against an uninterrupted reference, with warmstart omitted."""
    seed = 0
    prefix_steps = {"cheetah": 5, "hopper": 5, "walker": 2}[domain]
    actions = np.random.default_rng(seed + 200).uniform(-1, 1, (32, 1, reference.action_dim))
    compare_state(env.reset(seed=seed), reference.reset(seed=seed))
    check_step_matches_reference(env, reference, actions[:prefix_steps])
    saved = env.get_state()
    for action in -actions[prefix_steps : prefix_steps + 9]:
        env.step(action, states=None, frames=None).exclude("valid")
    omitted = saved.clone()
    omitted["warmstart"] = env.get_state()["warmstart"]
    env.set_state(omitted)
    continuation = actions[prefix_steps : prefix_steps + 8]
    bad_outputs = [env.step(action, states=None, frames=None).exclude("valid") for action in continuation]
    expected_outputs = [reference.step(action) for action in continuation]
    error_without = max(
        max_state_error(actual, expected, exclude=(("state", "warmstart"),))
        for actual, expected in zip(bad_outputs, expected_outputs, strict=True)
    )
    env.set_state(saved)
    good_outputs = [env.step(action, states=None, frames=None).exclude("valid") for action in continuation]
    error_with_schema = max(
        max_state_error(actual, expected) for actual, expected in zip(good_outputs, expected_outputs, strict=True)
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
            "common": ["qpos", "qvel", "time", "lifecycle counters", "task.cumulative_reward", "MT19937 state"],
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
                        arrays = compare_models(env, ref, minimum=500)
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

                        actions = np.random.default_rng(seed + 200).uniform(-1, 1, (32, batch, ref.action_dim))
                        check_state_roundtrip(env, actions=actions[:4])
                        compare_state(env.get_state(), ref.get_state())
                        steps = check_step_matches_reference(env, ref, actions)

                        other.reset(seed=seed + 991)
                        for action in -actions[:9]:
                            other.step(action, states=None, frames=None).exclude("valid")
                        # set_state reloads every lane's ctrl from dm_control, so the divergence is read before it.
                        control_diverged = not np.array_equal(env._simulator.b.ctrl, other._simulator.b.ctrl)
                        snapshot = env.get_state()
                        env.set_state(snapshot)
                        other.set_state(snapshot)
                        warmstart_equal = np.array_equal(
                            env._simulator.b.qacc_warmstart, other._simulator.b.qacc_warmstart
                        )
                        assert warmstart_equal and control_diverged
                        history = check_two_histories_same_state(env, other, snapshot, actions[:8])

                        frame = env.render(snapshot)
                        assert torch.equal(frame, env._render(env._pack_render_state(snapshot)))
                        before = env.get_state()
                        env.render(snapshot)
                        compare_state(env.get_state(), before)
                        assert_lossy_projection_rejected(env, snapshot)
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
                    got, expected = (
                        env.step(np.zeros((batch, ref.action_dim)), states=None, frames=None).exclude("valid"),
                        ref.step(np.zeros((batch, ref.action_dim))),
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
                    assert any(not torch.equal(rng_before[key], after[key]) for key in rng_before.keys(True, True))
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
                        compare_state(env.step(outside, states=None, frames=None).exclude("valid"), ref.step(outside))
                        evidence.setdefault("action_validation", {})[domain] = {
                            "shape_and_nonfinite_rejected": True,
                            "finite_out_of_bounds_matches_swm_passthrough": True,
                        }
                finally:
                    ref.close()

        ref = DMCContinuationReference(domain, 1)
        with se.make(env_name, num_envs=1, device=device) as env:
            try:
                env.set_variation({"light.diffuse_scale": [1.1]})
                assert np.array_equal(
                    env._simulator.lanes.envs[0].physics.model.opt.gravity,
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

    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_gate(args.device)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
