"""Batched NumPy and Torch classic-control steps against the Gymnasium adapter: same resets, State and outputs."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pygame")
torch = pytest.importorskip("torch")

import stable_envs as se  # noqa: E402
from stable_envs.envs.classic_control.simulation.batched import (  # noqa: E402
    MAX_EPISODE_STEPS,
    NUMPY_MIN_ENVS,
    TORCH_MIN_ENVS,
    BatchedClassicAdapter,
    classic_backend,
)
from stable_envs.envs.classic_control.simulation.local import CLASSIC_FIELDS, ClassicControlAdapter  # noqa: E402
from tests.helpers.reference.semantics import assert_same_state  # noqa: E402

LANES = 6
#: Physics values per lane; the extremes of each range make clamps and terminations frequent.
PHYSICS = {
    "cartpole": {
        "gravity": [9.8, 4.9, 14.7, 7.25, 12.5, 9.8],
        "masscart": [1.0, 0.5, 1.5, 0.8, 1.2, 1.0],
        "masspole": [0.1, 0.05, 0.5, 0.3, 0.12, 0.1],
        "length": [0.5, 0.25, 0.75, 0.4, 0.6, 0.5],
        "force_mag": [10.0, 5.0, 15.0, 7.5, 12.0, 10.0],
        "tau": [0.02, 0.01, 0.04, 0.03, 0.015, 0.02],
    },
    "pendulum": {
        "g": [10.0, 5.5, 14.25, 9.81, 5.0, 15.0],
        "m": [1.0, 0.6, 1.4, 1.1, 0.5, 1.5],
        "l": [1.0, 1.3, 0.55, 0.9, 0.5, 1.5],
        "max_torque": [2.0, 1.5, 3.7, 2.5, 1.0, 4.0],
        "max_speed": [8.0, 5.0, 11.0, 8.0, 4.0, 12.0],
        "dt": [0.05, 0.03, 0.08, 0.05, 0.1, 0.025],
    },
    "acrobot": {
        "link_length_1": [1.0, 0.5, 1.5, 0.75, 1.25, 1.0],
        "link_length_2": [1.0, 1.5, 0.5, 1.25, 0.75, 1.0],
        "link_mass_1": [1.0, 0.5, 1.5, 0.7, 1.3, 1.0],
        "link_mass_2": [1.0, 1.5, 0.5, 1.2, 0.6, 1.0],
        "link_com_pos_1": [0.5, 0.25, 0.75, 0.3, 0.6, 0.5],
        "link_com_pos_2": [0.5, 0.75, 0.25, 0.4, 0.7, 0.5],
        "link_moi": [1.0, 0.5, 1.5, 0.8, 1.2, 1.0],
    },
    "mountain_car": {
        "gravity": [0.0025, 0.00125, 0.00375, 0.002, 0.003, 0.0025],
        "force": [0.001, 0.0015, 0.0005, 0.0012, 0.0008, 0.001],
        "max_speed": [0.07, 0.035, 0.105, 0.05, 0.09, 0.07],
    },
    "mountain_car_continuous": {
        "power": [0.0015, 0.00225, 0.00075, 0.002, 0.001, 0.0015],
        "max_speed": [0.07, 0.035, 0.105, 0.05, 0.09, 0.07],
    },
}
DOMAINS = tuple(PHYSICS)
CLASSES = {
    "cartpole": "CartPole",
    "pendulum": "Pendulum",
    "acrobot": "Acrobot",
    "mountain_car": "MountainCar",
    "mountain_car_continuous": "MountainCarContinuous",
}
#: State fields that no step computes with sin, cos, remainder or a power.
EXACT = {
    "cartpole": (("task", "steps_beyond_terminated"), ("task", "has_steps_beyond_terminated")),
    "pendulum": ("torque", ("task", "has_last_u")),
    "acrobot": (),
    "mountain_car": (),
    "mountain_car_continuous": (("task", "state_is_float32"),),
}
BACKENDS = ["numpy", "torch"]


def _tol(backend, steps=1):
    """NumPy steps call the source's C functions and match exactly; Torch steps agree to rounding."""
    return 0.0 if backend == "numpy" else (1e-9 if steps == 1 else 1e-7)


def _actions(domain, steps, seed, dtype=None):
    rng = np.random.default_rng(seed)
    if domain == "cartpole":
        return torch.as_tensor(rng.integers(0, 2, (steps, LANES, 1)))
    if domain in ("acrobot", "mountain_car"):
        return torch.as_tensor(rng.integers(0, 3, (steps, LANES, 1)))
    bound = 4.5 if domain == "pendulum" else 1.5
    return torch.as_tensor(rng.uniform(-bound, bound, (steps, LANES, 1)), dtype=dtype or torch.float32)


def _pair(domain, backend, device="cpu", seed=7):
    reference = ClassicControlAdapter(domain, LANES, "cpu", seed)
    fast = BatchedClassicAdapter(domain, LANES, device, seed, backend)
    values = {name: np.asarray(rows, np.float64) for name, rows in PHYSICS[domain].items()}
    for item in (reference, fast):
        item.configure(values)
        item.reset(seed)
    return reference, fast


def _assert_agree(domain, actual, expected, atol=1e-9):
    actual, expected = actual.cpu(), expected.cpu()
    ours, theirs = actual["state"], expected["state"]
    assert set(ours.keys(True, True)) == set(theirs.keys(True, True))
    for key in theirs.keys(True, True):
        path = key if isinstance(key, tuple) else (key,)
        if key in EXACT[domain] or path[0] in ("rng", "physics", "lifecycle"):
            assert_same_state(ours[key], theirs[key])
        else:
            torch.testing.assert_close(ours[key], theirs[key], rtol=0, atol=atol, msg=lambda m, key=key: f"{key}: {m}")
        assert ours[key].dtype == CLASSIC_FIELDS[domain][key][1]
    for key in ("observation", "info_state"):
        assert actual[key].dtype == expected[key].dtype == torch.float32
        loose = 1e-6 if atol else 0.0
        torch.testing.assert_close(actual[key], expected[key], rtol=loose, atol=loose)
    assert actual["reward"].dtype == expected["reward"].dtype == torch.float64
    torch.testing.assert_close(actual["reward"], expected["reward"], rtol=0, atol=atol)
    assert torch.equal(actual["terminated"], expected["terminated"])
    assert torch.equal(actual["truncated"], expected["truncated"])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("domain", DOMAINS)
def test_each_step_matches_gymnasium_through_truncation(domain, backend):
    """Start every step from the Gymnasium State, so rounding differences cannot grow."""
    reference, fast = _pair(domain, backend)
    try:
        assert_same_state(fast.get_state(), reference.get_state())
        steps = MAX_EPISODE_STEPS[domain] + 5
        for t, action in enumerate(_actions(domain, steps, 3)):
            if t % 97 == 96:
                # The float64 state that MountainCarContinuous holds only after a reset.
                reference.reset(t)
                fast.reset(t)
            fast.set_state(reference.get_state())
            _assert_agree(domain, fast.step(action), reference.step(action), _tol(backend))
    finally:
        reference.close()
        fast.close()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("domain", DOMAINS)
def test_trajectories_match_gymnasium(domain, backend):
    reference, fast = _pair(domain, backend)
    try:
        for action in _actions(domain, 40, 5):
            _assert_agree(domain, fast.step(action), reference.step(action), _tol(backend, 40))
    finally:
        reference.close()
        fast.close()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("domain", ["pendulum", "mountain_car_continuous"])
def test_float64_actions_match_gymnasium(domain, backend):
    reference, fast = _pair(domain, backend)
    try:
        for t, action in enumerate(_actions(domain, 60, 9, torch.float64)):
            if t == 30:
                reference.reset(1)
                fast.reset(1)
            fast.set_state(reference.get_state())
            _assert_agree(domain, fast.step(action), reference.step(action), _tol(backend))
    finally:
        reference.close()
        fast.close()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_continuous_mountain_car_boundaries_match_gymnasium(backend, dtype):
    """Speed and position limits, the left wall and the goal, with float64 and float32 lane states."""
    reference, fast = _pair("mountain_car_continuous", backend)
    f32 = float(np.float32(-1.2))
    speed = np.asarray(PHYSICS["mountain_car_continuous"]["max_speed"])
    cases = [
        (-1.2, -0.01),
        (f32, -1e-9),
        (-1.19, -speed[2]),
        (0.45, 0.0),
        (float(np.float32(0.45)), 1e-9),
        (0.6, speed[3]),
        (0.59, float(np.float32(speed[4]))),
        (0.0, -float(np.float32(speed[5]))),
    ]
    forces = [-1.0, 1.0, -1.0000001, 1.0000001, 0.0, 2.0, -3.0, 0.5]
    try:
        for is32 in (False, True):
            for shift in range(len(cases)):
                state = reference.get_state()
                rows = [cases[(lane + shift) % len(cases)] for lane in range(LANES)]
                # Float64 values in a float32 lane: set_state rounds them as Gymnasium does.
                values = np.asarray(rows, np.float64)
                state["position"] = torch.as_tensor(values[:, :1])
                state["velocity"] = torch.as_tensor(values[:, 1:])
                state["task", "state_is_float32"].fill_(is32)
                reference.set_state(state)
                fast.set_state(state)
                action = torch.as_tensor(
                    [[forces[(lane + shift) % len(forces)]] for lane in range(LANES)], dtype=dtype
                )
                _assert_agree("mountain_car_continuous", fast.step(action), reference.step(action), _tol(backend))
    finally:
        reference.close()
        fast.close()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("domain", DOMAINS)
def test_inactive_lanes_hold_and_report_zero_outputs(domain, backend):
    reference, fast = _pair(domain, backend)
    try:
        for t, action in enumerate(_actions(domain, 30, 5)):
            active = torch.tensor([t % 2 == 0, True, t % 3 == 0, False, True, t % 5 == 0])
            expected = reference.step(action, active=active)
            _assert_agree(domain, fast.step(action, active=active), expected, _tol(backend, 30))
    finally:
        reference.close()
        fast.close()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("domain", DOMAINS)
def test_state_restores_across_the_two_adapters(domain, backend):
    reference, fast = _pair(domain, backend)
    try:
        actions = _actions(domain, 40, 11)
        for action in actions[:20]:
            fast.step(action)
        reference.set_state(fast.get_state())
        fast.set_state(reference.get_state())
        for action in actions[20:]:
            _assert_agree(domain, fast.step(action), reference.step(action), _tol(backend, 20))
        reference.reset(None)
        fast.reset(None)
        assert_same_state(fast.get_state(), reference.get_state())
    finally:
        reference.close()
        fast.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_configure_keeps_live_lanes_and_applies_new_physics(backend):
    reference, fast = _pair("cartpole", backend)
    try:
        for action in _actions("cartpole", 5, 2):
            fast.step(action)
            reference.step(action)
        values = {name: np.asarray(rows[::-1], np.float64) for name, rows in PHYSICS["cartpole"].items()}
        fast.configure(values)
        reference.configure(values)
        assert_same_state(fast.get_state(), reference.get_state())
        for action in _actions("cartpole", 5, 4):
            _assert_agree("cartpole", fast.step(action), reference.step(action), _tol(backend, 5))
    finally:
        reference.close()
        fast.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_invalid_discrete_actions_raise_like_gymnasium(backend):
    for domain, error in (("cartpole", AssertionError), ("mountain_car", AssertionError), ("acrobot", TypeError)):
        fast = BatchedClassicAdapter(domain, 2, "cpu", 0, backend)
        try:
            fast.reset(0)
            held = fast.get_state()
            with pytest.raises(error):
                fast.step(torch.zeros((2, 1)))
            with pytest.raises(AssertionError if domain != "acrobot" else IndexError):
                fast.step(torch.tensor([[0], [3]]))
            fast.step(torch.tensor([[0], [7]]), active=torch.tensor([True, False]))
            held["lifecycle", "elapsed_steps"][0] += 1
            assert torch.equal(fast.get_state()["lifecycle", "elapsed_steps"], held["lifecycle", "elapsed_steps"])
        finally:
            fast.close()
    fast = BatchedClassicAdapter("acrobot", 3, "cpu", 0, backend)
    reference = ClassicControlAdapter("acrobot", 3, "cpu", 0)
    try:
        fast.reset(0)
        reference.reset(0)
        # Python list indexing accepts negative actions.
        action = torch.tensor([[-1], [-2], [-3]])
        torch.testing.assert_close(fast.step(action)["state"], reference.step(action)["state"], rtol=0, atol=1e-9)
    finally:
        fast.close()
        reference.close()


def test_backend_selection():
    for domain, fewest in NUMPY_MIN_ENVS.items():
        for backend in (None, "auto"):
            assert classic_backend(domain, "cpu", fewest, backend) == "numpy"
            assert classic_backend(domain, "cpu", fewest - 1, backend) == "gym"
            assert classic_backend(domain, "cpu", 4096, backend) == "numpy"
        # By default CUDA lanes step on the GPU at every batch size; "auto" keeps small batches on the host.
        for lanes in (1, fewest, 4096):
            assert classic_backend(domain, "cuda", lanes) == "torch"
            assert classic_backend(domain, "cuda:0", lanes, "gym") == "gym"
        assert classic_backend(domain, "cuda", TORCH_MIN_ENVS, "auto") == "torch"
        assert classic_backend(domain, "cuda", 16, "auto", torch_min_envs=16) == "torch"
        assert classic_backend(domain, "cuda", 15, "auto", torch_min_envs=16) == "numpy"
        assert classic_backend(domain, "cuda", fewest - 1, "auto", torch_min_envs=16) == "gym"
    with pytest.raises(ValueError, match="torch_min_envs"):
        classic_backend("cartpole", "cuda", 2, "auto", torch_min_envs=0)
    with pytest.raises(ValueError, match="simulation_backend"):
        classic_backend("cartpole", "cpu", 2, "mjx")
    for name in CLASSES.values():
        with se.make(name, num_envs=8, device="cpu") as env:
            assert env.simulation_backend == "numpy" and env.execution_mode == "NATIVE_NUMPY"
            assert isinstance(env._simulator, BatchedClassicAdapter) and env._simulator.backend == "numpy"
        with se.make(name, num_envs=2, device="cpu", simulation_backend="gym") as env:
            assert env.simulation_backend == "gym" and env.execution_mode == "DELEGATED_CPU"
            assert type(env._simulator) is ClassicControlAdapter
        with se.make(name, num_envs=2, device="cpu", simulation_backend="torch") as env:
            assert env.simulation_backend == "torch" and env.execution_mode == "NATIVE_TORCH"
            assert env._simulator.backend == "torch"


@pytest.mark.parametrize("domain", DOMAINS)
def test_public_batched_env_matches_gymnasium_env(domain):
    name = CLASSES[domain]
    actions = _actions(domain, 12, 9)[:, :3].transpose(0, 1).contiguous()
    with (
        se.make(name, num_envs=3, device="cpu", simulation_backend="gym") as gym_env,
        se.make(name, num_envs=3, device="cpu", simulation_backend="numpy") as fast_env,
    ):
        gym_env.reset(seed=4)
        fast_env.reset(seed=4)
        a = gym_env.step(actions, frames=slice(None))
        b = fast_env.step(actions, frames=slice(None))
        for key in ("reward", "terminated", "truncated", "valid"):
            torch.testing.assert_close(b[key], a[key], rtol=0, atol=1e-9)
        for key in a["states"].keys(True, True):
            torch.testing.assert_close(b["states"][key], a["states"][key], rtol=0, atol=1e-9)
        # Frames draw the State; differences far below a pixel leave them equal.
        assert torch.equal(b["frames"], a["frames"])


@pytest.mark.parametrize("backend", ["gym", "numpy"])
@pytest.mark.parametrize("domain", DOMAINS)
def test_render_rows_follow_the_simulator_after_steps(domain, backend):
    """Steps leave the render rows to be read from the simulator; every reader sees the last step's State."""
    name = CLASSES[domain]
    with se.make(name, num_envs=3, device="cpu", simulation_backend=backend) as env:
        env.reset(seed=2)
        for action in _actions(domain, 4, 3)[:, :3]:
            state = env.step(action, frames=None)["state"]
        rows = env._pack_render_state(state)
        assert torch.equal(env._get_render_state(), rows)
        assert torch.equal(env._render(), env.render(state))
        env.step(_actions(domain, 1, 4)[0][:3], frames=None)
        assert torch.equal(env._render(), env.render())
        env._set_render_state(rows)
        assert torch.equal(env._get_render_state(), rows)


@pytest.mark.gpu
@pytest.mark.parametrize("domain", DOMAINS)
def test_cuda_steps_match_gymnasium(domain):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    for backend in BACKENDS:
        reference, fast = _pair(domain, backend, "cuda")
        try:
            for action in _actions(domain, MAX_EPISODE_STEPS[domain] + 5, 13):
                fast.set_state(reference.get_state())
                _assert_agree(domain, fast.step(action.cuda()), reference.step(action), _tol(backend))
        finally:
            reference.close()
            fast.close()
    with se.make(CLASSES[domain], num_envs=8, device="cuda") as env:
        assert env.simulation_backend == "torch" and env.execution_mode == "NATIVE_TORCH"
        assert env.get_state()["lifecycle", "elapsed_steps"].device.type == "cuda"
    with se.make(CLASSES[domain], num_envs=8, device="cuda", simulation_backend="auto", torch_min_envs=16) as env:
        assert env.simulation_backend == "numpy" and env.execution_mode == "NATIVE_NUMPY"
