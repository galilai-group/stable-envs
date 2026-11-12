"""The Numba Rocket kernel against the PyFlyt source with the same seed, actions and thrust noise."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")
pytest.importorskip("PyFlyt")

from stable_envs.envs.rocket.simulation import numba_physics as nk  # noqa: E402
from stable_envs.envs.rocket.simulation.task import RocketLandingEnv  # noqa: E402
from tests.helpers.rocket_policies import OPTIONS, policy  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.reference]

MAX_STEPS = 1300
#: Free flight follows PyBullet to double rounding; the bound leaves room for growth over 650 steps.
ATOL = 1e-9


def _body(env):
    a = env.env
    rocket = a.drones[0].Id
    pos, orn = a.getBasePositionAndOrientation(rocket)
    vel, ang = a.getBaseVelocity(rocket)
    return np.concatenate([pos, orn, vel, ang])


#: Kernel fields named after ``RocketLandingEnv`` attributes; flags and counters are compared as float64.
WRAPPER_FIELDS = (
    "ang_vel", "ang_pos", "lin_vel", "lin_pos", "ground_lin_vel", "previous_ang_vel", "previous_lin_vel",
    "previous_lin_pos", "previous_ground_lin_vel", "ignition_state", "throttle_state", "previous_ignition_state",
    "previous_throttle_state", "landing_pad_contact", "step_count", "action", "state", "reward",
)  # fmt: skip
#: Kernel fields that hold PyFlyt drone and aviary attributes.
ACTUATOR_FIELDS = (
    "fin",
    "gimbal",
    "throttle",
    "fuel",
    "booster_ignition",
    "command",
    "physics_steps",
    "link_velocity",
)
SNAPSHOT_STEP = 100


def source_fields(env):
    drone = env.env.drones[0]
    surfaces = drone.lifting_surfaces.surfaces
    fields = {k: np.asarray(getattr(env, k), dtype=np.float64) for k in WRAPPER_FIELDS}
    fields.update(
        fin=np.array([s.actuation for s in surfaces]),
        gimbal=drone.booster_gimbal.gimbal_state[0],
        throttle=np.float64(drone.boosters.throttle[0]),
        fuel=np.float64(drone.boosters.ratio_fuel_remaining[0]),
        booster_ignition=np.float64(drone.boosters.ignition_state[0]),
        command=np.asarray(drone.cmd, dtype=np.float64),
        physics_steps=np.float64(env.env.physics_steps),
        link_velocity=np.stack(
            [drone.bodies.local_body_velocities[0]] + [s.local_surface_velocity for s in surfaces[1:]]
        ),
    )
    return {k: np.array(v, dtype=np.float64) for k, v in fields.items()}


def kernel_fields(state, b):
    keys = (*WRAPPER_FIELDS, *ACTUATOR_FIELDS)
    return {k: np.asarray(state[k][b], dtype=np.float64).copy() for k in keys}


def run_source(kind, seed):
    """Closed-loop source rollout: actions, body states, rewards, flags, observations and the start pose."""
    env = RocketLandingEnv(render_mode=None)
    try:
        obs, _ = env.reset(seed=seed, options=OPTIONS[kind])
        rec = {
            "start_pos": env.start_pos[0].copy(),
            "start_orn": env.start_orn[0].copy(),
            "reset_body": _body(env),
            "reset_obs": obs.copy(),
            "fields_reset": source_fields(env),
            "action": [],
            "body": [],
            "obs": [],
            "reward": [],
            "pad": [],
            "term": [],
            "trunc": [],
            "fatal": [],
            "complete": [],
            "oob": [],
        }
        for t in range(MAX_STEPS):
            action = policy(kind, t, env)
            obs, reward, term, trunc, info = env.step(action.copy())
            rec["action"].append(action)
            rec["body"].append(_body(env))
            rec["obs"].append(obs.copy())
            rec["reward"].append(float(reward))
            rec["pad"].append(float(env.landing_pad_contact))
            rec["term"].append(bool(term))
            rec["trunc"].append(bool(trunc))
            rec["fatal"].append(bool(info["fatal_collision"]))
            rec["complete"].append(bool(info["env_complete"]))
            rec["oob"].append(bool(info["out_of_bounds"]))
            if t == SNAPSHOT_STEP:
                rec["fields_step"] = source_fields(env)
            if term or trunc:
                break
        rec["rng_end"] = nk.rng_words([env.np_random])[0]
    finally:
        env.close()
    return {k: np.array(v) if isinstance(v, list) else v for k, v in rec.items()}


def start_columns(recs):
    """``start_state`` input for the source rollouts (default ``accelerate_drop``, spawn offset clipped to 2 m)."""
    start_pos = np.stack([r["start_pos"] for r in recs])
    position = start_pos.copy()
    position[:, :2] = np.clip(position[:, :2], -2.0, 2.0)
    num = len(recs)
    return {
        "start_pos": start_pos,
        "start_orn": np.stack([r["start_orn"] for r in recs]),
        "position": position,
        "linear_velocity": np.tile([0.0, 0.0, -30.0], (num, 1)),
        "angular_velocity": np.zeros((num, 3)),
    }


def run_kernel(recs, seeds):
    """Replay the source actions on the kernel, all rollouts in one batch, with the source noise streams."""
    num = len(recs)
    state = nk.new_state(num)
    state["rng"][:] = nk.rng_words([np.random.default_rng(s) for s in seeds])
    everyone = np.ones(num, bool)
    nk.start_state(state, everyone, start_columns(recs))
    nk.settle(state, None, everyone)
    out = {
        "reset_body": np.concatenate(
            [state["position"], state["quaternion"], state["linear_velocity"], state["angular_velocity"]], 1
        ),
        "reset_obs": nk.observation(state),
        "fields_reset": [kernel_fields(state, b) for b in range(num)],
        "fields_step": [None] * num,
    }
    steps = max(len(r["action"]) for r in recs)
    body = np.full((num, steps, 13), np.nan)
    obs = np.full((num, steps, nk.OBSERVATION_DIM), np.nan)
    reward = np.full((num, steps), np.nan)
    end = np.full(num, -1)
    live = everyone.copy()
    for t in range(steps):
        actions = np.stack([r["action"][t] if t < len(r["action"]) else np.zeros(7) for r in recs])
        rew, term, trunc, info = nk.step(state, actions, None, live)
        if t == SNAPSHOT_STEP:
            out["fields_step"] = [kernel_fields(state, b) for b in range(num)]
        for b in np.flatnonzero(live):
            body[b, t] = np.concatenate(
                [
                    state["position"][b],
                    state["quaternion"][b],
                    state["linear_velocity"][b],
                    state["angular_velocity"][b],
                ]
            )
            obs[b, t] = state["state"][b]
            reward[b, t] = rew[b]
            if term[b] or trunc[b]:
                end[b] = t
        live &= ~(term | trunc)
    out.update(
        body=body,
        obs=obs,
        reward=reward,
        end=end,
        term=state["termination"].copy(),
        trunc=state["truncation"].copy(),
        fatal=state["fatal_collision"].copy(),
        complete=state["env_complete"].copy(),
        oob=state["out_of_bounds"].copy(),
        rng=state["rng"].copy(),
    )
    return out


OUTCOME = ("fatal", "complete", "term", "trunc", "oob")


def first_contact(rec):
    """Index of the first step with pad contact or the end of the episode."""
    hits = np.flatnonzero((rec["pad"] > 0) | rec["term"] | rec["trunc"])
    return int(hits[0]) if len(hits) else len(rec["action"])


def quaternion_gap(a, b):
    return np.minimum(np.abs(a - b).max(-1), np.abs(a + b).max(-1))


def compare(kinds_seeds):
    """Source and kernel rollouts for ``(kind, seed)`` pairs; return one summary row per rollout."""
    recs = [run_source(kind, seed) for kind, seed in kinds_seeds]
    got = run_kernel(recs, [seed for _, seed in kinds_seeds])
    rows = []
    for b, ((kind, seed), rec) in enumerate(zip(kinds_seeds, recs, strict=True)):
        free = first_contact(rec)
        src, ours = rec["body"][:free], got["body"][b, :free]
        rows.append(
            {
                "kind": kind,
                "seed": seed,
                "steps": len(rec["action"]),
                "free_steps": free,
                "end": int(got["end"][b]),
                "src": tuple(bool(rec[k][-1]) for k in OUTCOME),
                "ours": tuple(bool(got[k][b]) for k in OUTCOME),
                "reset": float(
                    max(
                        np.abs(got["reset_body"][b] - rec["reset_body"]).max(),
                        np.abs(got["reset_obs"][b] - rec["reset_obs"]).max(),
                    )
                ),
                "dp": float(np.abs(ours[:, 0:3] - src[:, 0:3]).max(initial=0.0)),
                "dq": float(quaternion_gap(ours[:, 3:7], src[:, 3:7]).max(initial=0.0)),
                "dv": float(np.abs(ours[:, 7:10] - src[:, 7:10]).max(initial=0.0)),
                "dw": float(np.abs(ours[:, 10:13] - src[:, 10:13]).max(initial=0.0)),
                "dobs": float(np.abs(got["obs"][b, :free] - rec["obs"][:free]).max(initial=0.0)),
                "dreward": float(np.abs(got["reward"][b, :free] - rec["reward"][:free]).max(initial=0.0)),
                "reward_src_end": float(rec["reward"][-1]),
                "reward_ours_end": float(got["reward"][b, got["end"][b]]) if got["end"][b] >= 0 else float("nan"),
                "return_src": float(rec["reward"].sum()),
                "return_ours": float(np.nansum(got["reward"][b])),
                "same_rng": bool(np.array_equal(got["rng"][b], rec["rng_end"])),
                "dfields": max(
                    float(np.abs(got[key][b][k] - rec[key][k]).max())
                    for key in ("fields_reset", "fields_step")
                    if key in rec
                    for k in rec[key]
                ),
            }
        )
    return rows


#: ``(kind, seed)`` and the source outcome ``(fatal, landed)``. ``pad`` 1 touches the pad for two steps before it
#: lands, ``pad`` 4 crashes on the pad, ``firm`` 10 crashes on the ground with a leg first, ``hover`` runs into the
#: step limit, ``climb`` leaves through the ceiling.
CASES = {
    ("free", 0): (True, False),
    ("free", 3): (True, False),
    ("burn", 1): (True, False),
    ("burn", 4): (True, False),
    ("smooth", 0): (True, False),
    ("smooth", 4): (True, False),
    ("land", 0): (False, True),
    ("land", 2): (False, True),
    ("pad", 1): (False, True),
    ("pad", 4): (True, False),
    ("firm", 10): (True, False),
    ("hover", 0): (False, False),
    ("climb", 0): (False, False),
}
IDS = [f"{k}-{s}" for k, s in CASES]


@pytest.fixture(scope="module")
def rows():
    return compare(list(CASES))


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
def test_kernel_follows_pybullet_until_contact(rows, index):
    row = rows[index]
    assert row["free_steps"] >= 90
    assert row["reset"] < ATOL
    for key in ("dp", "dq", "dv", "dw", "dobs", "dreward", "dfields"):
        assert row[key] < ATOL, (key, row)


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
def test_kernel_ends_the_episode_as_the_source(rows, index):
    row = rows[index]
    assert row["ours"] == row["src"], row
    assert row["end"] == row["steps"] - 1, row
    # the kernel draws exactly the thrust noise the source draws
    assert row["same_rng"], row
    assert row["src"][:2] == CASES[row["kind"], row["seed"]]


@pytest.mark.parametrize("seed", [0, 1, 12345])
def test_rng_rows_draw_numpy_normals(seed):
    reference = np.random.default_rng(seed)
    rng = nk.rng_words([np.random.default_rng(seed), np.random.default_rng(seed + 1)])
    draws = nk.normals(rng, 200_000)
    np.testing.assert_array_equal(draws[0], reference.standard_normal(200_000))
    np.testing.assert_array_equal(draws[1], np.random.default_rng(seed + 1).standard_normal(200_000))
    assert nk.generator_state(rng[0]) == reference.bit_generator.state


def test_an_ended_environment_steps_nothing():
    state = nk.new_state(2)
    everyone = np.ones(2, bool)
    nk.start_state(
        state,
        everyone,
        {
            "start_pos": np.array([[0.0, 0.0, 450.0], [0.0, 0.0, 450.0]]),
            "start_orn": np.zeros((2, 3)),
            "position": np.array([[0.0, 0.0, 450.0], [0.0, 0.0, 450.0]]),
            "linear_velocity": np.tile([0.0, 0.0, -30.0], (2, 1)),
            "angular_velocity": np.zeros((2, 3)),
        },
    )
    nk.settle(state, np.zeros((2, nk.NOISE_PER_SETTLE)), everyone)
    state["termination"][1] = True
    before = {k: v.copy() for k, v in state.items()}
    reward, term, trunc, info = nk.step(state, np.zeros((2, 7)), np.zeros((2, nk.NOISE_PER_STEP)), everyone)
    assert reward[1] == 0.0 and term[1] and info["noise_used"][1] == 0 and info["noise_used"][0] == nk.NOISE_PER_STEP
    assert state["step_count"][1] == 1
    np.testing.assert_array_equal(state["position"][1], before["position"][1])
    assert not np.array_equal(state["position"][0], before["position"][0])
    position = state["position"][0].copy()
    nk.step(state, np.zeros((2, 7)), np.zeros((2, nk.NOISE_PER_STEP)), np.array([False, True]))
    np.testing.assert_array_equal(state["position"][0], position)
    assert state["step_count"][0] == 1
