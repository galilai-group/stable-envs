"""Classic-control acceptance checks against pinned SWM.

Seeds, batches, horizons, task/action edges, Variation values and rejected actions are fixed
per case. Source fields are never read through the candidate's export.
"""

import numpy as np
import torch

import stable_envs as se
from tests.helpers.reference.native_classic_semantics import NativeClassicReference
from tests.helpers.reference.profiles import PROFILES
from tests.helpers.reference.semantics import (
    assert_lossy_projection_rejected,
    check_reset_matches_reference,
    check_state_roundtrip,
    check_step_matches_reference,
    check_two_histories_same_state,
)
from tests.helpers.reference.semantics import (
    compare_state as _compare_state,
)

CASES = {
    case.domain: dict(historical_id=case.registration, stable_env=case.name, limit=limit)
    for case, limit in zip(PROFILES["native"], (500, 200, 500, 200, 999), strict=True)
}
EARLY = {"cartpole", "pendulum"}


def compare_state(actual, expected):
    # Source outputs have no chunk validity; the native profile checks it separately.
    if "valid" in actual.keys() and "valid" not in expected.keys():
        assert bool(actual["valid"].all())
        actual = actual.exclude("valid")
    return _compare_state(actual, expected)


def _actions(domain, seed, steps, batch):
    rng = np.random.default_rng(seed)
    shape = (steps, batch, 1)
    if domain in {"cartpole", "acrobot", "mountain_car"}:
        return rng.integers(0, 2 if domain == "cartpole" else 3, shape, dtype=np.int64)
    scale = 2 if domain == "pendulum" else 1
    return rng.uniform(-scale, scale, shape).astype(np.float32)


def _zero(domain, batch):
    if domain in {"acrobot", "mountain_car"}:
        return np.ones((batch, 1), np.int64)
    return np.zeros((batch, 1), np.int64 if domain == "cartpole" else np.float32)


def _assert_independent_snapshot(env, snapshot):
    frozen = snapshot.clone()
    first_float = next(key for key in snapshot.keys(True, True) if snapshot[key].is_floating_point())
    snapshot[first_float].add_(1)
    snapshot["rng", "state"].add_(1)
    compare_state(env.get_state(), frozen)


def _candidate_restore_against_uninterrupted_reference(domain, env, ref, actions):
    saved = env.get_state()
    compare_state(saved, ref.get_state())
    for action in actions[:5]:
        env.step(action, states=None, frames=None)
    env.set_state(saved)
    compare_state(env.get_state(), ref.get_state())
    return check_step_matches_reference(env, ref, actions)


def _lifecycle_boundary(domain, env, ref, batch):
    check_reset_matches_reference(env, ref, seed=31)
    action = _zero(domain, batch)
    limit = CASES[domain]["limit"]
    for _ in range(limit - 1):
        got, expected = env.step(action, states=None, frames=None), ref.step(action)
        compare_state(got, expected)
        assert not bool(got["truncated"].any())
    near_limit = env.get_state()
    got, expected = env.step(action, states=None, frames=None), ref.step(action)
    compare_state(got, expected)
    assert bool(got["truncated"].all())
    post, expected_post = env.step(action, states=None, frames=None), ref.step(action)
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
    got, expected = env.step(action, states=None, frames=None), ref.step(action)
    compare_state(got, expected)
    assert bool(got["terminated"].all())
    assert (got["reward"] == expected_reward).all()
    post, expected_post = env.step(action, states=None, frames=None), ref.step(action)
    compare_state(post, expected_post)
    return {
        "terminal_reward": float(got["reward"][0]),
        "post_termination_advances": True,
        "post_termination_terminated": bool(post["terminated"].all()),
    }


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
    ref = NativeClassicReference("mountain_car_continuous", 1)
    with se.make("MountainCarContinuous", num_envs=1, device=device) as env:
        try:
            snapshot = ref.reset(seed=17)
            compare_state(env.reset(seed=17), snapshot)
            assert not snapshot["task", "state_is_float32"].item()

            wrong = snapshot.clone()
            wrong["task", "state_is_float32"].fill_(True)
            env.set_state(wrong)
            got_wrong, expected = env.step(action, states=None, frames=None), ref.step(action)
            position_error = float(
                (got_wrong["state", "position"].detach().cpu() - expected["state", "position"].detach().cpu())
                .abs()
                .max()
            )
            observation_error = float(
                (got_wrong["observation"].detach().cpu() - expected["observation"].detach().cpu()).abs().max()
            )
            assert position_error > 0 or observation_error > 0

            env.set_state(snapshot)
            got = env.step(action, states=None, frames=None)
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
        got, expected = env.step(zero, states=None, frames=None), ref.step(zero)
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
    got, expected = env.step(zero, states=None, frames=None), ref.step(zero)
    compare_state(got, expected)
    assert (got["reward"] == 0).all()

    costly = upright.clone()
    costly["theta"].fill_(1.0)
    env.set_state(costly)
    ref.set_state(costly)
    got, expected = env.step(zero, states=None, frames=None), ref.step(zero)
    compare_state(got, expected)
    assert (got["reward"] < 0).all()
    return {"upright_reward": 0.0, "nonzero_cost_reward": float(got["reward"][0])}


def _action_semantics(domain, env, ref, batch):
    check_reset_matches_reference(env, ref, seed=81)
    before = env.get_state()
    if domain in {"cartpole", "acrobot", "mountain_car"}:
        invalid = np.zeros((batch, 1), np.float32) if domain == "cartpole" else np.ones((batch, 1), np.float64)
        errors = (AssertionError,) if domain == "cartpole" else (AssertionError, TypeError)
        for target in (env, ref):
            try:
                target.step(invalid)
            except errors:
                pass
            else:
                raise AssertionError(f"{domain} accepted a floating-point discrete action")
        compare_state(env.get_state(), before)
        return {"integer_discrete_only": True, "float_rejected": True}

    if domain == "pendulum":
        action = np.full((batch, 1), 3.25, np.float64)
        got, expected = env.step(action, states=None, frames=None), ref.step(action)
        compare_state(got, expected)
        assert (got["state", "torque"] == 2.0).all()

    precise_value = 0.1234567890123
    precise = np.full((batch, 1), precise_value, np.float64)
    got, expected = env.step(precise, states=None, frames=None), ref.step(precise)
    compare_state(got, expected)
    if domain == "pendulum":
        assert (got["state", "torque"] == precise_value).all()
    else:
        assert bool(got["state", "task", "state_is_float32"].all())
    saved = got["state"]
    zero = np.zeros((batch, 1), np.float32)
    env.step(zero, states=None, frames=None)
    env.set_state(saved)
    compare_state(env.get_state(), ref.get_state())
    got, expected = env.step(zero, states=None, frames=None), ref.step(zero)
    compare_state(got, expected)
    if domain == "pendulum":
        return {
            "float64_accepted": True,
            "out_of_bounds_clipped_to": 2.0,
            "in_bounds_float64_torque_exact": precise_value,
            "float64_state_restore": True,
        }

    check_reset_matches_reference(env, ref, seed=82)
    out_of_range = np.full((batch, 1), 2.0, np.float64)
    got, expected = env.step(out_of_range, states=None, frames=None), ref.step(out_of_range)
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
    if domain == "cartpole":
        return dict(gravity=12.25, masscart=1.25, masspole=0.25, length=0.625, force_mag=12.5, tau=0.03)
    if domain == "pendulum":
        return dict(g=12.5, m=1.25, l=0.75, max_torque=3.0, max_speed=10.0, dt=0.075)
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
    ref = NativeClassicReference(domain, 1, options=options)
    with se.make(CASES[domain]["stable_env"], num_envs=1, device=device) as env:
        try:
            env.set_variation({f"physics.{key}": [value] for key, value in values.items()})
            check_reset_matches_reference(env, ref, seed=11)
            steps = check_step_matches_reference(env, ref, _actions(domain, 11, 32, 1))
            return {"values": values, "steps": len(steps), "max_error": 0.0}
        finally:
            ref.close()


def run_case(domain, batch, device):
    case = CASES[domain]
    ref = NativeClassicReference(domain, batch)
    records = []
    with (
        se.make(case["stable_env"], num_envs=batch, device=device) as env,
        se.make(case["stable_env"], num_envs=batch, device=device) as other,
    ):
        try:
            assert env.execution_mode == "DELEGATED_CPU"
            if domain not in EARLY:
                _assert_default_float32_writes(domain, env)
            for seed in (0, 17, 42):
                check_reset_matches_reference(env, ref, seed=seed)
                _assert_independent_snapshot(env, env.get_state())
                actions = _actions(domain, seed + 200, 32, batch)
                check_state_roundtrip(env, actions=actions[:4])
                compare_state(env.get_state(), ref.get_state())
                steps = (
                    check_step_matches_reference(env, ref, actions)
                    if domain in EARLY
                    else _candidate_restore_against_uninterrupted_reference(domain, env, ref, actions)
                )
                other.reset(seed=seed + 991)
                for action in _actions(domain, seed + 991, 9, batch):
                    other.step(action, states=None, frames=None)
                resume = env.get_state()
                history = check_two_histories_same_state(env, other, resume, actions[:8])
                frame = env.render(resume)
                assert torch.equal(frame, env._render(env._pack_render_state(resume)))
                before = env.get_state()
                env.render(resume)
                compare_state(env.get_state(), before)
                assert_lossy_projection_rejected(env, resume)
                check_reset_matches_reference(env, ref, seed=None)
                if domain not in EARLY:
                    custom_ref = NativeClassicReference(domain, batch, options={"low": -0.02, "high": 0.03})
                    try:
                        compare_state(
                            env.reset(seed=seed, options={"low": -0.02, "high": 0.03}), custom_ref.reset(seed=seed)
                        )
                    finally:
                        custom_ref.close()
                records.append(
                    dict(
                        domain=domain,
                        historical_id=case["historical_id"],
                        stable_env=case["stable_env"],
                        batch=batch,
                        seed=seed,
                        reset=True,
                        unseeded_reset=True,
                        independent_snapshot=True,
                        roundtrip=True,
                        steps=len(steps),
                        two_history_steps=len(history),
                        full_state_render=True,
                        max_error=0.0,
                        **({} if domain in EARLY else dict(custom_reset_bounds=True, candidate_only_restore=True)),
                    )
                )
            lifecycle, near_limit = _lifecycle_boundary(domain, env, ref, batch)
            other.reset(seed=701)
            replay = check_two_histories_same_state(env, other, near_limit, _actions(domain, 702, 4, batch))
            terminal = (
                _cartpole_terminal(env, ref, batch)
                if domain == "cartpole"
                else _pendulum_rewards(env, ref, batch)
                if domain == "pendulum"
                else _terminal_and_reward(domain, env, ref, batch)
            )
            actions = _action_semantics(domain, env, ref, batch)
            records.append(
                dict(
                    domain=domain,
                    batch=batch,
                    **lifecycle,
                    **terminal,
                    boundary_replay_steps=len(replay),
                    action_validation=actions,
                )
            )
        finally:
            ref.close()
    return records
