"""Gate four source-aligned historical DMC defaults."""

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
from tests.helpers.reference.dmc_semantics import DMCReference as DMCAlignedReference
from tests.helpers.reference.semantics import (
    check_reset_matches_reference,
    check_state_roundtrip,
    check_step_matches_reference,
    check_two_histories_same_state,
    compare_state,
)

SWM_PIN = "176d54532b3c0a251a4f509b74f2fdacc143a63d"
CASES = {
    "ball_in_cup": "DMCBallInCupCatch",
    "finger": "DMCFingerTurnHard",
    "manipulator": "DMCManipulatorBringBall",
    "quadruped": "DMCQuadrupedWalk",
}
OMISSIONS = {
    "ball_in_cup": [("warmstart",)],
    "finger": [("warmstart",), ("model", "target_site_xz")],
    "manipulator": [
        ("warmstart",),
        ("model", "target_body_xz"),
        ("model", "target_body_qwqy"),
    ],
    "quadruped": [("warmstart",), ("activation",)],
}
# Ball-in-Cup prefix 5 has a zero omission witness on Oscar/MuJoCo 3.11.
# Prefix 6 is a contact snapshot with nonzero observable omission error on
# both the local ARM/3.12 and Oscar x86/3.11 profiles; restore stays exact.
PREFIXES = {"ball_in_cup": 6, "finger": 7, "manipulator": 1, "quadruped": 1}


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
            delta = float((left.double() - right.double()).abs().nan_to_num().max())
        error = max(error, delta)
    return error


def _omission_experiments(domain, env, reference):
    results = {}
    prefix = PREFIXES[domain]
    actions = np.random.default_rng(200).uniform(-1, 1, (32, 1, reference.action_dim))
    for omitted_key in OMISSIONS[domain]:
        compare_state(env.reset(seed=0), reference.reset(seed=0))
        check_step_matches_reference(env, reference, actions[:prefix])
        saved = env.get_state()
        physics = env._simulator.envs[0].physics
        contact_count = int(physics.data.ncon)
        constraint_count = int(physics.data.nefc)
        if omitted_key[0] == "model":
            env.reset(seed=991)
        else:
            for action in -actions[prefix : prefix + 9]:
                env.step(action, states=None, frames=None).exclude("valid")
        omitted = saved.clone()
        omitted[omitted_key] = env.get_state()[omitted_key]
        env.set_state(omitted)
        continuation = actions[prefix : prefix + 8]
        bad = [env.step(action, states=None, frames=None).exclude("valid") for action in continuation]
        expected = [reference.step(action) for action in continuation]
        error_without = max(
            _max_error(actual, wanted, exclude=(("state", *omitted_key),))
            for actual, wanted in zip(bad, expected, strict=True)
        )
        env.set_state(saved)
        good = [env.step(action, states=None, frames=None).exclude("valid") for action in continuation]
        error_with = max(_max_error(actual, wanted) for actual, wanted in zip(good, expected, strict=True))
        diagnostic = {
            "domain": domain,
            "omitted_key": "/".join(omitted_key),
            "seed": 0,
            "action_seed": 200,
            "prefix_steps": prefix,
            "error_without": error_without,
            "error_with": error_with,
        }
        print("DMC_OMISSION_RESULT", json.dumps(diagnostic, sort_keys=True), flush=True)
        assert error_with == 0, ("Restored continuation differs", diagnostic)
        assert error_without > 0, ("Omission witness has no observable effect", diagnostic)
        results["/".join(omitted_key)] = {
            "prefix_steps": prefix,
            "snapshot_contact_count": contact_count,
            "snapshot_constraint_count": constraint_count,
            "different_history": "reset seed 991" if omitted_key[0] == "model" else "9 actions",
            "continuation_steps": 8,
            "max_observable_error_without_field": error_without,
            "max_error_with_schema": error_with,
        }
    return results


def _compare_models(candidate, reference):
    actual, expected = candidate._simulator.envs[0].physics.model, reference.envs[0].env.physics.model
    compared = 0
    for name in dir(actual):
        if name.startswith("_") or not hasattr(expected, name):
            continue
        try:
            left, right = np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name))
        except Exception:
            continue
        if left.dtype.kind not in "biufc" or right.dtype.kind not in "biufc":
            continue
        assert left.shape == right.shape, (name, left.shape, right.shape)
        assert np.array_equal(left, right, equal_nan=True), name
        compared += 1
    assert compared >= 500
    return compared


def _assert_lossy_rejected(env, snapshot):
    projection = env._unpack_render_state(env._pack_render_state(snapshot))
    try:
        env.set_state(projection)
    except (TypeError, ValueError):
        return
    raise AssertionError("Lossy renderer projection was accepted")


def _assert_bad_actions(env, action_dim):
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
            raise AssertionError("Invalid action accepted")


def _positive_sparse_state(domain, reference):
    physics = reference.envs[0].env.physics
    if domain == "ball_in_cup":
        target = physics.named.data.site_xpos["target", ["x", "z"]].copy()
        current = physics.named.data.xpos["ball", ["x", "z"]].copy()
        physics.named.data.qpos[["ball_x", "ball_z"]] += target - current
    elif domain == "finger":
        best = None
        for angle in np.linspace(-np.pi, np.pi, 4097):
            physics.named.data.qpos["hinge"] = angle
            physics.forward()
            distance = physics.dist_to_target()
            if best is None or distance < best[0]:
                best = (distance, angle)
        assert best is not None and best[0] <= 0
        physics.named.data.qpos["hinge"] = best[1]
    else:
        raise ValueError(domain)
    physics.data.qvel[:] = 0
    physics.forward()
    return reference.get_state()


def run_gate(device="cpu"):
    import stable_worldmodel

    source = Path(stable_worldmodel.__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert commit == SWM_PIN
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
        "state_omission_experiments": {},
        "construction": {},
        "cases": [],
    }
    for domain, env_name in CASES.items():
        for batch in (1, 3):
            reference = DMCAlignedReference(domain, batch)
            with (
                se.make(env_name, num_envs=batch, device=device) as env,
                se.make(env_name, num_envs=batch, device=device) as other,
            ):
                try:
                    if batch == 1:
                        check_reset_matches_reference(env, reference, seed=0)
                        evidence["construction"][domain] = {
                            "numeric_model_arrays_exact": _compare_models(env, reference),
                            "nq": env._simulator.envs[0].physics.model.nq,
                            "nv": env._simulator.envs[0].physics.model.nv,
                            "na": env._simulator.envs[0].physics.model.na,
                            "action_dim": reference.action_dim,
                            "observation_dim": int(reference.envs[0].observation_space.shape[0]),
                            "control_timestep": reference.envs[0].env._env.control_timestep(),
                            "step_limit": reference.step_limit,
                        }
                        evidence["state_omission_experiments"][domain] = _omission_experiments(domain, env, reference)
                    for seed in (0, 17, 42):
                        check_reset_matches_reference(env, reference, seed=seed)
                        saved, live = env.get_state(), env.get_state()
                        saved["qpos"].add_(1)
                        saved["rng", "keys"].add_(1)
                        compare_state(env.get_state(), live)
                        actions = np.random.default_rng(seed + 200).uniform(-1, 1, (32, batch, reference.action_dim))
                        check_state_roundtrip(env, actions=actions[:4])
                        compare_state(env.get_state(), reference.get_state())
                        steps = check_step_matches_reference(env, reference, actions)
                        snapshot = env.get_state()
                        other.reset(seed=seed + 991)
                        for action in -actions[:9]:
                            other.step(action, states=None, frames=None).exclude("valid")
                        history = check_two_histories_same_state(env, other, snapshot, actions[:8])
                        frame = env.render(snapshot)
                        assert torch.equal(frame, env._render(env._pack_render_state(snapshot)))
                        before = env.get_state()
                        env.render(snapshot)
                        compare_state(env.get_state(), before)
                        _assert_lossy_rejected(env, snapshot)
                        evidence["cases"].append(
                            {
                                "domain": domain,
                                "batch": batch,
                                "seed": seed,
                                "reset": True,
                                "snapshot_independent": True,
                                "roundtrip": True,
                                "steps": len(steps),
                                "different_history_steps": len(history),
                                "full_state_render": True,
                                "max_error": 0.0,
                            }
                        )
                    check_reset_matches_reference(env, reference, seed=5)
                    boundary = np.zeros((502, batch, reference.action_dim))
                    check_step_matches_reference(env, reference, boundary[:499])
                    near_limit = env.get_state()
                    rng_before = near_limit.select("rng").clone()
                    check_step_matches_reference(env, reference, boundary[499:])
                    after = env.get_state()
                    assert any(not torch.equal(rng_before[key], after[key]) for key in rng_before.keys(True, True))
                    other.reset(seed=77)
                    check_two_histories_same_state(env, other, near_limit, boundary[:4])
                    evidence["cases"].append(
                        {
                            "domain": domain,
                            "batch": batch,
                            "lifecycle_public_steps": 502,
                            "implicit_reset_rng_changes": True,
                            "lifecycle_replay": True,
                        }
                    )
                    if batch == 1:
                        check_reset_matches_reference(env, reference, seed=31)
                        _assert_bad_actions(env, reference.action_dim)
                        check_reset_matches_reference(env, reference, seed=31)
                        outside = np.full((1, reference.action_dim), 1.1)
                        compare_state(
                            env.step(outside, states=None, frames=None).exclude("valid"), reference.step(outside)
                        )
                        if domain in ("ball_in_cup", "finger"):
                            check_reset_matches_reference(env, reference, seed=13)
                            positive = _positive_sparse_state(domain, reference)
                            env.set_state(positive)
                            reference.set_state(positive)
                            got = env.step(np.zeros((1, reference.action_dim)), states=None, frames=None).exclude(
                                "valid"
                            )
                            expected = reference.step(np.zeros((1, reference.action_dim)))
                            compare_state(got, expected)
                            assert got["reward"].item() > 0
                            if domain == "ball_in_cup":
                                assert got["success"].item() == 1
                            evidence.setdefault("positive_sparse_reward", {})[domain] = {
                                "reward": got["reward"].item(),
                                "success": got["success"].item(),
                            }
                finally:
                    reference.close()

        reference = DMCAlignedReference(domain, 1)
        with se.make(env_name, num_envs=1, device=device) as env:
            try:
                env.set_variation({"light.diffuse_scale": [1.1]})
                check_reset_matches_reference(env, reference, seed=23)
                actions = np.random.default_rng(23).uniform(-1, 1, (8, 1, reference.action_dim))
                check_step_matches_reference(env, reference, actions)
                evidence.setdefault("visual_only_variation", {})[domain] = {"steps": 8, "max_error": 0.0}
            finally:
                reference.close()

    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_gate(args.device)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
