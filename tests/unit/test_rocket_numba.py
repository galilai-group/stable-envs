"""Rocket on the Numba kernel: backend routing, seeded starts, State round trips, frames and rollout fidelity."""

import numpy as np
import pytest
import torch

pytest.importorskip("PyFlyt")
pytest.importorskip("pybullet")
pytest.importorskip("numba")

from stable_envs import make  # noqa: E402
from stable_envs.envs.rocket.simulation.numba_batch import RocketNumbaAdapter  # noqa: E402
from stable_envs.envs.rocket.simulation.task import RocketLandingEnv  # noqa: E402

RESET_OPTIONS = [
    None,
    {"variation": ["environment"]},
    {"variation": ["all"]},
    {"randomize_drop": True},
    {"randoimize_drop": True, "accelerate_drop": False},
    {"randomize_drop": True, "variation": ["rocket"]},
]


def _generator_words(generator):
    state = generator.bit_generator.state
    return state["state"]["state"], state["state"]["inc"], state["has_uint32"], state["uinteger"]


def _smooth(batch, t):
    phase = np.arange(batch)[:, None] + np.arange(7)
    action = 0.3 * np.sin(t / 15.0 + phase)
    action[:, 3] = 1.0
    action[:, 4] = 0.5 + 0.4 * np.sin(t / 25.0 + phase[:, 4])
    return action


def test_routing():
    with make("PyFlytRocketLanding", num_envs=2) as env:
        assert (env.simulation_backend, env.execution_mode) == ("numba", "NATIVE_NUMBA")
        assert isinstance(env._simulator, RocketNumbaAdapter)
    with make("PyFlytRocketLanding", num_envs=2, simulation_backend="pybullet") as env:
        assert (env.simulation_backend, env.execution_mode) == ("pybullet", "DELEGATED_CPU")
        assert not isinstance(env._simulator, RocketNumbaAdapter)
    if torch.cuda.is_available():
        with make("PyFlytRocketLanding", num_envs=2, device="cuda") as env:
            assert (env.simulation_backend, env.execution_mode) == ("numba", "HOST_NUMBA")
            assert env.get_state()["physics", "float"].device.type == "cuda"
    with pytest.raises(ValueError, match="simulation_backend"):
        make("PyFlytRocketLanding", num_envs=1, simulation_backend="bullet")
    with pytest.raises(ValueError, match="agent_hz"):
        make("PyFlytRocketLanding", num_envs=1, simulator_config={"agent_hz": 20})


@pytest.mark.parametrize("options", RESET_OPTIONS)
def test_a_seed_gives_the_source_start(options):
    """The start observation, the start pose and the noise stream position equal ``RocketLandingEnv``'s."""
    seed, batch = 31, 3
    with make("PyFlytRocketLanding", num_envs=batch) as env:
        state = env.reset(seed=seed, options=options)
        for lane in range(batch):
            source = RocketLandingEnv()
            try:
                observation, _ = source.reset(seed=seed + lane, options=options)
                rocket = source.env.drones[0].Id
                position, quaternion = source.env.getBasePositionAndOrientation(rocket)
                ours = state["current", "wrapper", "state"][lane].numpy()
                np.testing.assert_allclose(ours, observation, rtol=0, atol=1e-12)
                np.testing.assert_allclose(
                    state["current", "physics", "body_1", "position"][lane].numpy(), position, rtol=0, atol=1e-12
                )
                np.testing.assert_allclose(
                    state["current", "physics", "body_1", "quaternion"][lane].numpy(), quaternion, rtol=0, atol=1e-12
                )
                assert _generator_words(env._simulator._generator(lane)) == _generator_words(source.np_random)
            finally:
                source.close()


def test_save_restore_continue_is_exact():
    batch = 4
    with (
        make("PyFlytRocketLanding", num_envs=batch, seed=5) as env,
        make("PyFlytRocketLanding", num_envs=batch) as other,
    ):
        env.reset(seed=9, options={"variation": ["all"]})
        # Environments that end on different steps.
        for t in range(90):
            env.step(_smooth(batch, t), frames=None)
        saved = env.get_state()
        first = [env.step(_smooth(batch, t), states=None, frames=None)["state"] for t in range(90, 160)]
        env.set_state(saved)
        other.set_state(saved)
        for t, expected in zip(range(90, 160), first, strict=True):
            for restored in (env, other):
                out = restored.step(_smooth(batch, t), states=None, frames=None)["state"]
                for key, value in expected.items(True, True):
                    assert torch.equal(out[key], value), key


def test_invalid_state_changes_nothing():
    with make("PyFlytRocketLanding", num_envs=2, seed=1) as env:
        before = env.get_state()
        bad = before.clone()
        bad["physics", "float"][1, 0] = float("nan")
        with pytest.raises(ValueError):
            env.set_state(bad)
        bad = before.clone()
        bad["recipe", "config"][0, 0] = 1.0
        with pytest.raises(ValueError, match="configuration"):
            env.set_state(bad)
        bad = before.clone()
        group, start, width = env._simulator._columns["rng"]
        bad["physics", group][0, start + width - 1] = 2
        with pytest.raises(ValueError, match="noise"):
            env.set_state(bad)
        after = env.get_state()
        for key, value in before.items(True, True):
            assert torch.equal(after[key], value), key


def test_frames_of_a_state_equal_between_backends():
    """The renderer reads only the poses and colours, so a State with the same values draws the same frames."""
    batch = 2
    with (
        make("PyFlytRocketLanding", num_envs=batch, size=64, simulation_backend="pybullet", seed=3) as reference,
        make("PyFlytRocketLanding", num_envs=batch, size=64, seed=3) as env,
    ):
        reference.reset(seed=4, options={"variation": ["all"]})
        env.reset(seed=4, options={"variation": ["all"]})
        for t in range(40):
            reference.step(_smooth(batch, t), frames=None)
            env.step(_smooth(batch, t), frames=None)
        source, ours = reference.get_state(), env.get_state()
        for i in range(3):
            for leaf in ("position", "quaternion"):
                ours["current", "physics", f"body_{i}", leaf] = source["current", "physics", f"body_{i}", leaf]
        ours["current", "diagnostic", "cached_links"] = source["current", "diagnostic", "cached_links"][:, :1, :7]
        ours["current", "wrapper", "state"] = source["current", "wrapper", "state"]
        ours["render", "rgba"] = source["render", "rgba"]
        assert torch.equal(env.render(ours), reference.render(source))
        live = env.step(_smooth(batch, 40), frames=slice(-1, None))
        assert torch.equal(live["frames"][:, -1], env.render(env.get_state()))


def _landing(source, t):
    """A suicide burn with attitude hold, from the source's own task values."""
    z, vz = source.lin_pos[2], source.ground_lin_vel[2]
    h = z - 2.575
    target = -max(1.0, np.sqrt(2 * 6.0 * max(h, 0.0)))
    armed = h < 1.15 * vz * vz / 60.0 + 3.0 and vz < -5.0
    ignition = 1.0 if (armed or (source.ignition_state and vz < target)) and vz < target else 0.0
    roll, pitch = source.ang_pos[:2]
    gimbal = np.clip(3.0 * np.array([roll, pitch]) + source.ang_vel[:2], -1, 1)
    return np.array([0.0, 0.0, 0.0, ignition, 1.0, *gimbal])


def _late_burn(source, t):
    """Burn too late: the rocket reaches the pad or the ground too fast."""
    h, vz = source.lin_pos[2] - 2.575, source.ground_lin_vel[2]
    ignition = 1.0 if vz < -3.0 and h < 0.5 * vz * vz / 72.0 + 1.0 else 0.0
    return np.array([0.1 * np.sin(0.04 * t), 0.0, 0.0, ignition, 1.0, 0.0, 0.0])


def _smooth_policy(source, t):
    ignition = 1.0 if (80 <= t < 140 or 200 <= t < 260) else 0.0
    return np.array(
        [
            0.5 * np.sin(0.05 * t),
            0.5 * np.sin(0.07 * t + 1),
            0.3 * np.sin(0.03 * t),
            ignition,
            0.5 + 0.5 * np.sin(0.1 * t),
            0.5 * np.sin(0.2 * t),
            0.5 * np.cos(0.15 * t),
        ]
    )


#: Largest position difference (m) of a rollout from the source's, and the most agent steps by which the end of
#: an episode may differ. The kernel's free flight matches PyBullet to rounding; its contacts are approximate.
POSITION_TOLERANCE = 0.1
END_STEP_TOLERANCE = 3


@pytest.mark.parametrize(
    "policy, options",
    [
        (_landing, None),
        (_landing, {"variation": ["environment"]}),
        (_late_burn, {"variation": ["environment"]}),
        (_smooth_policy, {"variation": ["environment"]}),
    ],
)
def test_rollouts_keep_the_source_outcome(policy, options):
    """Scripted rollouts of at least 300 steps on the source, replayed on the kernel: same outcome and end."""
    seeds = [0, 1, 2]
    rollouts = []
    for seed in seeds:
        source = RocketLandingEnv()
        try:
            source.reset(seed=seed, options=options)
            actions, positions, outcome = [], [], None
            for t in range(1200):
                action = policy(source, t)
                _, _, terminated, truncated, info = source.step(action.copy())
                actions.append(action)
                positions.append(np.array(source.env.getBasePositionAndOrientation(source.env.drones[0].Id)[0]))
                if terminated or truncated:
                    outcome = (bool(terminated), bool(truncated), bool(info["env_complete"]))
                    break
            assert len(actions) >= 300
            rollouts.append((np.array(actions), np.array(positions), outcome))
        finally:
            source.close()
    length = max(len(a) for a, _, _ in rollouts) + END_STEP_TOLERANCE
    actions = np.zeros((len(seeds), length, 7))
    for lane, (a, _, _) in enumerate(rollouts):
        actions[lane, : len(a)] = a
        actions[lane, len(a) :] = a[-1]
    with make("PyFlytRocketLanding", num_envs=len(seeds), seed=0) as env:
        # Environment ``lane`` resets with seed ``lane``, as its source rollout did.
        env.reset(seed=0, options=options)
        positions, ends, outcomes = [], [None] * len(seeds), [None] * len(seeds)
        for t in range(length):
            out = env.step(torch.as_tensor(actions[:, t]), frames=None)
            positions.append(out["state"]["current", "physics", "body_1", "position"].numpy().copy())
            terminated, truncated, success = (
                out[k].reshape(len(seeds), -1)[:, -1].numpy() for k in ("terminated", "truncated", "success")
            )
            for lane in np.flatnonzero(terminated | truncated):
                if ends[lane] is None:
                    ends[lane] = t + 1
                    outcomes[lane] = (bool(terminated[lane]), bool(truncated[lane]), bool(success[lane]))
    positions = np.stack(positions, 1)
    for lane, (a, source_positions, outcome) in enumerate(rollouts):
        assert outcomes[lane] == outcome, (lane, outcomes[lane], outcome)
        assert abs(ends[lane] - len(a)) <= END_STEP_TOLERANCE, (lane, ends[lane], len(a))
        common = min(ends[lane], len(a))
        deviation = np.abs(positions[lane, :common] - source_positions[:common]).max()
        assert deviation <= POSITION_TOLERANCE, (lane, deviation)
