"""Exact semantic gate for the first two historical native classic-control IDs."""

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
from tests.helpers.reference.native_classic_semantics import NativeClassicReference
from tests.helpers.reference.semantics import (
    check_reset_matches_reference,
    check_state_roundtrip,
    check_step_matches_reference,
    check_two_histories_same_state,
    compare_state,
)

SWM_PIN = "176d54532b3c0a251a4f509b74f2fdacc143a63d"
CASES = {
    "cartpole": {"historical_id": "swm/CartPoleControl-v1", "stable_env": "CartPole"},
    "pendulum": {"historical_id": "swm/PendulumControl-v1", "stable_env": "Pendulum"},
}


def _actions(domain, seed, steps, batch):
    rng = np.random.default_rng(seed)
    if domain == "cartpole":
        return rng.integers(0, 2, size=(steps, batch, 1), dtype=np.int64)
    return rng.uniform(-2, 2, size=(steps, batch, 1)).astype(np.float32)


def _assert_independent_snapshot(env, snapshot):
    live = env.get_state()
    first_float = next(key for key in snapshot.keys(True, True) if snapshot[key].is_floating_point())
    snapshot[first_float].add_(1)
    snapshot["rng", "state"].add_(1)
    compare_state(env.get_state(), live)


def _assert_lossy_rejected(env, snapshot):
    packed = env._pack_render_state(snapshot)
    for bad in (packed, env._unpack_render_state(packed)):
        try:
            env.set_state(bad)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("Lossy renderer projection was accepted")


def _lifecycle_boundary(domain, env, ref, batch):
    check_reset_matches_reference(env, ref, seed=31)
    zero = np.zeros((batch, 1), dtype=np.int64 if domain == "cartpole" else np.float32)
    limit = 500 if domain == "cartpole" else 200
    for _ in range(limit - 1):
        got, expected = env.step(zero, capture="states"), ref.step(zero)
        compare_state(got, expected)
        assert not bool(got["truncated"].any())
    near_limit = env.get_state()
    got, expected = env.step(zero, capture="states"), ref.step(zero)
    compare_state(got, expected)
    assert bool(got["truncated"].all())
    # Gymnasium TimeLimit does not latch or auto-reset.  It advances again and
    # continues returning truncated after the exact boundary.
    post, expected_post = env.step(zero, capture="states"), ref.step(zero)
    compare_state(post, expected_post)
    assert bool(post["truncated"].all())
    return {"time_limit": limit, "first_truncated_step": limit, "post_truncation_advances": True}, near_limit


def _cartpole_terminal(env, ref, batch):
    check_reset_matches_reference(env, ref, seed=73)
    state = ref.get_state()
    state["x"].fill_(2.5)
    state["x_dot"].zero_()
    state["theta"].zero_()
    state["theta_dot"].zero_()
    state["task", "steps_beyond_terminated"].zero_()
    state["task", "has_steps_beyond_terminated"].zero_()
    env.set_state(state)
    ref.set_state(state)
    zero = np.zeros((batch, 1), np.int64)
    rewards = []
    for _ in range(3):
        got, expected = env.step(zero, capture="states"), ref.step(zero)
        compare_state(got, expected)
        assert bool(got["terminated"].all())
        rewards.append(float(got["reward"][0]))
    assert rewards == [1.0, 0.0, 0.0]
    return {"terminal_reward_sequence": rewards, "post_termination_advances": True}


def _pendulum_rewards(env, ref, batch):
    check_reset_matches_reference(env, ref, seed=74)
    upright = ref.get_state()
    upright["theta"].zero_()
    upright["theta_dot"].zero_()
    upright["torque"].zero_()
    upright["task", "has_last_u"].fill_(True)
    env.set_state(upright)
    ref.set_state(upright)
    zero = np.zeros((batch, 1), np.float32)
    got, expected = env.step(zero, capture="states"), ref.step(zero)
    compare_state(got, expected)
    assert (got["reward"] == 0).all()

    costly = upright.clone()
    costly["theta"].fill_(1.0)
    env.set_state(costly)
    ref.set_state(costly)
    got, expected = env.step(zero, capture="states"), ref.step(zero)
    compare_state(got, expected)
    assert (got["reward"] < 0).all()
    return {"upright_reward": 0.0, "nonzero_cost_reward": float(got["reward"][0])}


def _action_semantics(domain, env, ref, batch):
    check_reset_matches_reference(env, ref, seed=81)
    before = env.get_state()
    if domain == "cartpole":
        invalid = np.zeros((batch, 1), np.float32)
        for target in (env, ref):
            try:
                target.step(invalid)
            except AssertionError:
                pass
            else:
                raise AssertionError("CartPole accepted float action for Discrete(2)")
        compare_state(env.get_state(), before)
        return {"integer_discrete_only": True, "float_rejected": True}

    # Pendulum warns for a float64 action but accepts it, clips it, and keeps
    # the caller's precision in last_u.  Normalise last_u losslessly in State.
    action = np.full((batch, 1), 3.25, np.float64)
    got, expected = env.step(action, capture="states"), ref.step(action)
    compare_state(got, expected)
    assert (got["state", "torque"] == 2.0).all()

    precise_value = 0.1234567890123
    precise = np.full((batch, 1), precise_value, np.float64)
    got, expected = env.step(precise, capture="states"), ref.step(precise)
    compare_state(got, expected)
    assert (got["state", "torque"] == precise_value).all()
    saved = got["state"]
    env.step(np.zeros((batch, 1), np.float32), capture="states")
    env.set_state(saved)
    compare_state(env.get_state(), ref.get_state())
    got, expected = env.step(np.zeros((batch, 1), np.float32), capture="states"), ref.step(np.zeros((batch, 1), np.float32))
    compare_state(got, expected)
    return {
        "float64_accepted": True,
        "out_of_bounds_clipped_to": 2.0,
        "in_bounds_float64_torque_exact": precise_value,
        "float64_state_restore": True,
    }


def _physical_variation(domain, device):
    if domain == "cartpole":
        values = {
            "gravity": 12.25,
            "masscart": 1.25,
            "masspole": 0.25,
            "length": 0.625,
            "force_mag": 12.5,
            "tau": 0.03,
        }
    else:
        values = {"g": 12.5, "m": 1.25, "l": 0.75, "max_torque": 3.0, "max_speed": 10.0, "dt": 0.075}
    options = {
        "variation_values": {f"physics.{key}": np.asarray([value], np.float32) for key, value in values.items()}
    }
    ref = NativeClassicReference(domain, 1, options=options)
    with se.make(CASES[domain]["stable_env"], num_envs=1, device=device) as env:
        try:
            env.set_variation({f"physics.{key}": [value] for key, value in values.items()})
            check_reset_matches_reference(env, ref, seed=11)
            steps = check_step_matches_reference(env, ref, _actions(domain, 11, 32, 1))
            return {"values": values, "steps": len(steps), "max_error": 0.0}
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
        "execution_mode": "DELEGATED_CPU",
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "gymnasium", "tensordict", "stable-worldmodel")
        },
        "mappings": CASES,
        "scope_limit": "Only the exact empty-kwargs CartPole-v1 and Pendulum-v1 historical defaults are gated.",
        "native_precision_probe": {
            "method": "existing torch equations from identical pinned-wrapper State and float32 SWM physics defaults",
            "steps": 32,
            "actions": "CartPole alternating 0/1; Pendulum seeded uniform float32",
            "observed": {
                "cartpole_seed17": {
                    "exact_steps": 13,
                    "max_theta": 8.326672684688674e-17,
                    "max_theta_dot": 4.440892098500626e-16,
                },
                "cartpole_seed42": {"exact_steps": 31, "max_theta_dot": 5.551115123125783e-17},
                "pendulum_seed17": {"exact_steps": 30, "max_theta_dot": 4.440892098500626e-16},
            },
            "conclusion": "Observed nonzero last-bit drift prevents an exact native gate; no tolerance introduced.",
        },
        "cases": [],
        "physical_variation": {},
    }
    for domain, case in CASES.items():
        for batch in (1, 3):
            ref = NativeClassicReference(domain, batch)
            with (
                se.make(case["stable_env"], num_envs=batch, device=device) as env,
                se.make(case["stable_env"], num_envs=batch, device=device) as other,
            ):
                try:
                    assert env.execution_mode == "DELEGATED_CPU"
                    for seed in (0, 17, 42):
                        check_reset_matches_reference(env, ref, seed=seed)
                        snapshot = env.get_state()
                        _assert_independent_snapshot(env, snapshot)
                        actions = _actions(domain, seed + 200, 32, batch)
                        check_state_roundtrip(env, actions=actions[:4])
                        compare_state(env.get_state(), ref.get_state())
                        steps = check_step_matches_reference(env, ref, actions)

                        other.reset(seed=seed + 991)
                        other_actions = _actions(domain, seed + 991, 9, batch)
                        for action in other_actions:
                            other.step(action, capture="states")
                        resume = env.get_state()
                        history = check_two_histories_same_state(env, other, resume, actions[:8])

                        frame = env.render(resume)
                        assert torch.equal(frame, env._render(env._pack_render_state(resume)))
                        before = env.get_state()
                        env.render(resume)
                        compare_state(env.get_state(), before)
                        _assert_lossy_rejected(env, resume)

                        # Seed=None must continue the restored PCG64 streams.
                        check_reset_matches_reference(env, ref, seed=None)
                        evidence["cases"].append(
                            {
                                "domain": domain,
                                "historical_id": case["historical_id"],
                                "stable_env": case["stable_env"],
                                "batch": batch,
                                "seed": seed,
                                "reset": True,
                                "unseeded_reset": True,
                                "independent_snapshot": True,
                                "roundtrip": True,
                                "steps": len(steps),
                                "two_history_steps": len(history),
                                "full_state_render": True,
                                "max_error": 0.0,
                            }
                        )

                    lifecycle, near_limit = _lifecycle_boundary(domain, env, ref, batch)
                    other.reset(seed=701)
                    replay = check_two_histories_same_state(env, other, near_limit, _actions(domain, 702, 4, batch))
                    terminal = _cartpole_terminal(env, ref, batch) if domain == "cartpole" else {}
                    rewards = _pendulum_rewards(env, ref, batch) if domain == "pendulum" else {}
                    actions = _action_semantics(domain, env, ref, batch)
                    evidence["cases"].append(
                        {
                            "domain": domain,
                            "batch": batch,
                            **lifecycle,
                            **terminal,
                            **rewards,
                            "boundary_replay_steps": len(replay),
                            "action_validation": actions,
                        }
                    )
                finally:
                    ref.close()
        evidence["physical_variation"][domain] = _physical_variation(domain, device)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_gate(args.device)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
