"""Exact semantic gate for the remaining three historical classic-control IDs."""

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
from tests.helpers.reference.native_classic_remaining_semantics import NativeClassicRemainingReference
from tests.helpers.reference.semantics import (
    check_reset_matches_reference,
    check_state_roundtrip,
    check_step_matches_reference,
    check_two_histories_same_state,
    compare_state,
)

SWM_PIN = "176d54532b3c0a251a4f509b74f2fdacc143a63d"
CASES = {
    "acrobot": {"historical_id": "swm/AcrobotControl-v1", "stable_env": "Acrobot", "limit": 500},
    "mountain_car": {
        "historical_id": "swm/MountainCarControl-v0",
        "stable_env": "MountainCar",
        "limit": 200,
    },
    "mountain_car_continuous": {
        "historical_id": "swm/MountainCarContinuousControl-v0",
        "stable_env": "MountainCarContinuous",
        "limit": 999,
    },
}


def _actions(domain, seed, steps, batch):
    rng = np.random.default_rng(seed)
    if domain in {"acrobot", "mountain_car"}:
        return rng.integers(0, 3, size=(steps, batch, 1), dtype=np.int64)
    return rng.uniform(-1, 1, size=(steps, batch, 1)).astype(np.float32)


def _zero(domain, batch):
    if domain in {"acrobot", "mountain_car"}:
        return np.ones((batch, 1), np.int64)
    return np.zeros((batch, 1), np.float32)


def _assert_independent_snapshot(env, snapshot):
    frozen = snapshot.clone()
    first_float = next(key for key in snapshot.keys(True, True) if snapshot[key].is_floating_point())
    snapshot[first_float].add_(1)
    snapshot["rng", "state"].add_(1)
    compare_state(env.get_state(), frozen)


def _assert_lossy_rejected(env, snapshot):
    packed = env._pack_render_state(snapshot)
    for bad in (packed, env._unpack_render_state(packed)):
        try:
            env.set_state(bad)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("Lossy renderer projection was accepted")


def _candidate_restore_against_uninterrupted_reference(domain, env, ref, actions):
    saved = env.get_state()
    compare_state(saved, ref.get_state())
    for action in actions[:5]:
        env.step(action, frames=None)
    env.set_state(saved)
    compare_state(env.get_state(), ref.get_state())
    return check_step_matches_reference(env, ref, actions)


def _lifecycle_boundary(domain, env, ref, batch):
    check_reset_matches_reference(env, ref, seed=31)
    action = _zero(domain, batch)
    limit = CASES[domain]["limit"]
    for _ in range(limit - 1):
        got, expected = env.step(action, frames=None), ref.step(action)
        compare_state(got, expected)
        assert not bool(got["truncated"].any())
    near_limit = env.get_state()
    got, expected = env.step(action, frames=None), ref.step(action)
    compare_state(got, expected)
    assert bool(got["truncated"].all())
    post, expected_post = env.step(action, frames=None), ref.step(action)
    compare_state(post, expected_post)
    assert bool(post["truncated"].all())
    return {"time_limit": limit, "first_truncated_step": limit, "post_truncation_advances": True}, near_limit


def _terminal_and_reward(domain, env, ref, batch):
    check_reset_matches_reference(env, ref, seed=74)
    state = ref.get_state()
    action = _zero(domain, batch)
    if domain == "acrobot":
        state["theta1"].fill_(np.pi)
        state["theta2"].zero_()
        state["dtheta1"].zero_()
        state["dtheta2"].zero_()
        expected_reward = 0.0
    else:
        state["position"].fill_(0.55 if domain == "mountain_car" else 0.50)
        state["velocity"].fill_(0.01)
        if domain == "mountain_car_continuous":
            state["task", "state_is_float32"].fill_(True)
        expected_reward = -1.0 if domain == "mountain_car" else 100.0
    env.set_state(state)
    ref.set_state(state)
    got, expected = env.step(action, frames=None), ref.step(action)
    compare_state(got, expected)
    assert bool(got["terminated"].all())
    assert (got["reward"] == expected_reward).all()
    post, expected_post = env.step(action, frames=None), ref.step(action)
    compare_state(post, expected_post)
    return {
        "terminal_reward": float(got["reward"][0]),
        "post_termination_advances": True,
        "post_termination_terminated": bool(post["terminated"].all()),
    }


def _action_semantics(domain, env, ref, batch):
    check_reset_matches_reference(env, ref, seed=81)
    before = env.get_state()
    if domain in {"acrobot", "mountain_car"}:
        invalid = np.ones((batch, 1), np.float64)
        for target in (env, ref):
            try:
                target.step(invalid)
            except (AssertionError, TypeError):
                pass
            else:
                raise AssertionError(f"{domain} accepted float action for Discrete(3)")
        compare_state(env.get_state(), before)
        return {"integer_discrete_only": True, "float_rejected": True}

    precise_value = 0.1234567890123
    precise = np.full((batch, 1), precise_value, np.float64)
    got, expected = env.step(precise, frames=None), ref.step(precise)
    compare_state(got, expected)
    assert bool(got["state", "task", "state_is_float32"].all())
    saved = got["state"]
    env.step(np.zeros((batch, 1), np.float32), frames=None)
    env.set_state(saved)
    compare_state(env.get_state(), ref.get_state())
    got, expected = env.step(np.zeros((batch, 1), np.float32), frames=None), ref.step(np.zeros((batch, 1), np.float32))
    compare_state(got, expected)

    check_reset_matches_reference(env, ref, seed=82)
    out_of_range = np.full((batch, 1), 2.0, np.float64)
    got, expected = env.step(out_of_range, frames=None), ref.step(out_of_range)
    compare_state(got, expected)
    assert (got["reward"] == -0.4).all()
    return {
        "float64_accepted": True,
        "in_bounds_float64_action": precise_value,
        "post_step_dtype_restored": True,
        "physics_clip": 1.0,
        "unclipped_reward_action": 2.0,
        "out_of_range_reward": -0.4,
    }


def _variation_values(domain):
    if domain == "acrobot":
        return {
            "link_length_1": 1.25,
            "link_length_2": 0.75,
            "link_mass_1": 1.125,
            "link_mass_2": 0.875,
            "link_com_pos_1": 0.625,
            "link_com_pos_2": 0.375,
            "link_moi": 1.25,
        }
    if domain == "mountain_car":
        return {"gravity": 0.003, "force": 0.00125, "max_speed": 0.09}
    return {"power": 0.002, "max_speed": 0.09}


def _physical_variation(domain, device):
    values = _variation_values(domain)
    options = {
        "variation_values": {f"physics.{key}": np.asarray([value], np.float32) for key, value in values.items()}
    }
    ref = NativeClassicRemainingReference(domain, 1, options=options)
    with se.make(CASES[domain]["stable_env"], num_envs=1, device=device) as env:
        try:
            env.set_variation({f"physics.{key}": [value] for key, value in values.items()})
            check_reset_matches_reference(env, ref, seed=11)
            steps = check_step_matches_reference(env, ref, _actions(domain, 11, 32, 1))
            return {"values": values, "steps": len(steps), "max_error": 0.0}
        finally:
            ref.close()


def _assert_default_float32_writes(domain, env):
    base = env._simulator.envs[0].unwrapped
    attrs = {
        "acrobot": {"LINK_LENGTH_1": 1.0, "LINK_COM_POS_1": 0.5, "LINK_MOI": 1.0},
        "mountain_car": {"gravity": 0.0025, "force": 0.001, "max_speed": 0.07},
        "mountain_car_continuous": {"power": 0.0015, "max_speed": 0.07},
    }[domain]
    for name, value in attrs.items():
        assert getattr(base, name) == float(np.float32(value))


def _continuous_dtype_flag_necessity(device):
    """Show that omitting the reset-vs-stepped dtype bit changes continuation."""
    action = np.asarray([[0.1234567890123]], np.float64)
    ref = NativeClassicRemainingReference("mountain_car_continuous", 1)
    with se.make("MountainCarContinuous", num_envs=1, device=device) as env:
        try:
            snapshot = ref.reset(seed=17)
            compare_state(env.reset(seed=17), snapshot)
            assert not snapshot["task", "state_is_float32"].item()

            wrong = snapshot.clone()
            wrong["task", "state_is_float32"].fill_(True)
            env.set_state(wrong)
            got_wrong, expected = env.step(action, frames=None), ref.step(action)
            position_error = float((
                got_wrong["state", "position"].detach().cpu()
                - expected["state", "position"].detach().cpu()
            ).abs().max())
            observation_error = float((
                got_wrong["observation"].detach().cpu()
                - expected["observation"].detach().cpu()
            ).abs().max())
            assert position_error > 0 or observation_error > 0

            env.set_state(snapshot)
            got = env.step(action, frames=None)
            compare_state(got, expected)
            return {
                "seed": 17,
                "action_float64": float(action[0, 0]),
                "wrong_dtype_position_error": position_error,
                "wrong_dtype_observation_error": observation_error,
                "correct_flag_max_error": 0.0,
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
        "execution_mode": "DELEGATED_CPU",
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "gymnasium", "tensordict", "stable-worldmodel")
        },
        "mappings": CASES,
        "scope_limit": "Only empty-kwargs Acrobot-v1, MountainCar-v0, and MountainCarContinuous-v0 historical defaults are gated.",
        "dtype_probe": {
            "acrobot": "reset ndarray float32; step ndarray float64 after torque append",
            "mountain_car": "reset ndarray float64; step tuple of numpy.float64 scalars",
            "mountain_car_continuous": "reset ndarray float64; every step ndarray float32; dtype flag is continuation state",
        },
        "continuous_dtype_flag_necessity": _continuous_dtype_flag_necessity(device),
        "cases": [],
        "physical_variation": {},
    }
    for domain, case in CASES.items():
        for batch in (1, 3):
            ref = NativeClassicRemainingReference(domain, batch)
            with (
                se.make(case["stable_env"], num_envs=batch, device=device) as env,
                se.make(case["stable_env"], num_envs=batch, device=device) as other,
            ):
                try:
                    assert env.execution_mode == "DELEGATED_CPU"
                    _assert_default_float32_writes(domain, env)
                    for seed in (0, 17, 42):
                        check_reset_matches_reference(env, ref, seed=seed)
                        snapshot = env.get_state()
                        _assert_independent_snapshot(env, snapshot)
                        snapshot = env.get_state()
                        actions = _actions(domain, seed + 200, 32, batch)
                        check_state_roundtrip(env, actions=actions[:4])
                        compare_state(env.get_state(), ref.get_state())
                        steps = _candidate_restore_against_uninterrupted_reference(domain, env, ref, actions)

                        other.reset(seed=seed + 991)
                        for action in _actions(domain, seed + 991, 9, batch):
                            other.step(action, frames=None)
                        resume = env.get_state()
                        history = check_two_histories_same_state(env, other, resume, actions[:8])

                        frame = env.render(resume)
                        assert torch.equal(frame, env._render(env._pack_render_state(resume)))
                        before = env.get_state()
                        env.render(resume)
                        compare_state(env.get_state(), before)
                        _assert_lossy_rejected(env, resume)

                        check_reset_matches_reference(env, ref, seed=None)
                        custom_ref = NativeClassicRemainingReference(
                            domain, batch, options={"low": -0.02, "high": 0.03}
                        )
                        try:
                            compare_state(
                                env.reset(seed=seed, options={"low": -0.02, "high": 0.03}),
                                custom_ref.reset(seed=seed),
                            )
                        finally:
                            custom_ref.close()
                        evidence["cases"].append(
                            {
                                "domain": domain,
                                "historical_id": case["historical_id"],
                                "stable_env": case["stable_env"],
                                "batch": batch,
                                "seed": seed,
                                "reset": True,
                                "custom_reset_bounds": True,
                                "unseeded_reset": True,
                                "independent_snapshot": True,
                                "candidate_only_restore": True,
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
                    terminal = _terminal_and_reward(domain, env, ref, batch)
                    actions = _action_semantics(domain, env, ref, batch)
                    evidence["cases"].append(
                        {
                            "domain": domain,
                            "batch": batch,
                            **lifecycle,
                            **terminal,
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
