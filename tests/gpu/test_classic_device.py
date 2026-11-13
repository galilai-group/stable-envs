"""Classic control on CUDA: State, steps and resets stay on the GPU and agree with Gymnasium."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("pygame")
torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

import stable_envs as se  # noqa: E402
from stable_envs.envs.classic_control.simulation.batched import MAX_EPISODE_STEPS  # noqa: E402
from stable_envs.envs.classic_control.simulation.device import DeviceClassicAdapter  # noqa: E402
from stable_envs.envs.classic_control.simulation.local import CLASSIC_FIELDS, ClassicControlAdapter  # noqa: E402
from tests.helpers.reference.semantics import assert_same_state  # noqa: E402
from tests.unit.test_classic_batched import CLASSES, DOMAINS, LANES, PHYSICS, _actions, _assert_agree  # noqa: E402

#: Torch agrees with Gymnasium to rounding: one step from the same State, and 40 steps of a trajectory.
STEP_ATOL, TRAJECTORY_ATOL = 1e-9, 1e-7


def _pair(domain, seed=7, fused=True):
    reference = ClassicControlAdapter(domain, LANES, "cpu", seed)
    fast = DeviceClassicAdapter(domain, LANES, "cuda", seed)
    # Without the Triton kernel, the graph replays the batched Torch kernel.
    fast._fused = fast._fused and fused
    values = {name: np.asarray(rows, np.float64) for name, rows in PHYSICS[domain].items()}
    for item in (reference, fast):
        item.configure(values)
        item.reset(seed)
    fast.set_state(reference.get_state())
    return reference, fast


@pytest.mark.parametrize("domain", DOMAINS)
def test_each_step_matches_gymnasium_through_truncation(domain):
    reference, fast = _pair(domain)
    try:
        assert_same_state(fast.get_state().cpu(), reference.get_state())
        for t, action in enumerate(_actions(domain, MAX_EPISODE_STEPS[domain] + 5, 3)):
            if t % 97 == 96:
                reference.reset(t)
            fast.set_state(reference.get_state())
            _assert_agree(domain, fast.step(action.cuda()), reference.step(action), STEP_ATOL)
    finally:
        reference.close()
        fast.close()


@pytest.mark.parametrize("fused", [True, False], ids=["triton", "torch"])
@pytest.mark.parametrize("domain", DOMAINS)
def test_trajectories_and_inactive_lanes_match_gymnasium(domain, fused):
    reference, fast = _pair(domain, fused=fused)
    try:
        for t, action in enumerate(_actions(domain, 40, 5)):
            active = None if t < 20 else torch.tensor([t % 2 == 0, True, t % 3 == 0, False, True, t % 5 == 0])
            expected = reference.step(action, active=active)
            actual = fast.step(action.cuda(), active=None if active is None else active.cuda())
            _assert_agree(domain, actual, expected, TRAJECTORY_ATOL)
    finally:
        reference.close()
        fast.close()


@pytest.mark.parametrize("domain", ["pendulum", "mountain_car_continuous"])
def test_float64_actions_match_gymnasium(domain):
    reference, fast = _pair(domain)
    try:
        for action in _actions(domain, 30, 9, torch.float64):
            fast.set_state(reference.get_state())
            _assert_agree(domain, fast.step(action.cuda()), reference.step(action), STEP_ATOL)
    finally:
        reference.close()
        fast.close()


@pytest.mark.parametrize("domain", DOMAINS)
def test_seeded_resets_repeat_and_follow_the_gymnasium_distribution(domain):
    n = 4096
    fast = DeviceClassicAdapter(domain, n, "cuda", 0)
    try:
        first = fast.reset(5)
        assert_same_state(fast.reset(5), first)
        again = fast.reset(None)
        assert set(first.keys(True, True)) == set(CLASSIC_FIELDS[domain])
        for key, (width, dtype) in CLASSIC_FIELDS[domain].items():
            assert first[key].shape == (n, width) and first[key].dtype == dtype and first[key].device.type == "cuda"
        assert torch.equal(first["lifecycle", "elapsed_steps"], torch.zeros_like(first["lifecycle", "elapsed_steps"]))
        reference = ClassicControlAdapter(domain, 256, "cpu", 0)
        try:
            gym = reference.reset(5)
        finally:
            reference.close()
        for key in first.keys(True, True):
            if key[0] in ("rng", "physics", "lifecycle"):
                continue
            ours, theirs = first[key].double().cpu(), gym[key].double()
            if theirs.min() == theirs.max():
                assert torch.equal(ours, torch.full_like(ours, float(theirs[0, 0])))
                continue
            low, high = float(theirs.min()), float(theirs.max())
            span = high - low
            assert ours.min() >= low - 0.01 * span and ours.max() <= high + 0.01 * span, key
            # A uniform on [low, high] has mean (low + high) / 2 and standard deviation span / sqrt(12).
            assert abs(float(ours.mean()) - (low + high) / 2) < 0.05 * span, key
            assert abs(float(ours.std()) - span / math.sqrt(12)) < 0.05 * span, key
            assert not torch.equal(again[key], first[key]), key
        if domain == "acrobot":
            theta = first["theta1"]
            assert torch.equal(theta, theta.float().double())
    finally:
        fast.close()


@pytest.mark.parametrize("domain", DOMAINS)
def test_cuda_default_steps_on_the_gpu_without_host_syncs(domain):
    with se.make(CLASSES[domain], num_envs=8, device="cuda") as env:
        assert env.simulation_backend == "torch" and env.execution_mode == "NATIVE_TORCH"
        assert isinstance(env._simulator, DeviceClassicAdapter)
        actions = _actions(domain, 3, 1)[:, :1].expand(3, 8, 1).cuda()
        env.reset(seed=3)
        env.step(actions[0], frames=None)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for action in actions[1:]:
                result = env.step(action, frames=None)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        state = result["state"]
        assert state["lifecycle", "elapsed_steps"].device.type == "cuda"
        assert torch.equal(state["lifecycle", "elapsed_steps"], torch.full((8, 1), 3, device="cuda"))
        # The step writes the render rows; they draw the returned State.
        sim = torch.cat([state[name].reshape(8, 1) for name in env.sim_state_names], 1)
        torch.testing.assert_close(env._sim, sim, rtol=0, atol=0)
        assert torch.equal(env.render(), env.render(state))


def test_auto_routes_small_cuda_batches_to_numpy():
    with se.make("CartPole", num_envs=8, device="cuda", simulation_backend="auto") as env:
        assert env.simulation_backend == "torch" and isinstance(env._simulator, DeviceClassicAdapter)
    with se.make("CartPole", num_envs=8, device="cuda", simulation_backend="auto", torch_min_envs=16) as env:
        assert env.simulation_backend == "numpy" and env.execution_mode == "NATIVE_NUMPY"


def _policy(domain, observation):
    """A scripted, mostly smooth controller for each task."""
    o = observation.double()
    if domain == "cartpole":
        return (o[:, 2] + 0.5 * o[:, 3] + 0.05 * o[:, 0] + 0.1 * o[:, 1] > 0).long()[:, None]
    if domain == "pendulum":
        cos, sin, speed = o[:, 0], o[:, 1], o[:, 2]
        energy = 0.5 * speed**2 + 10.0 * (cos - 1.0)
        pump = -0.5 * energy * torch.sign(speed * cos + 1e-9)
        hold = -10.0 * torch.atan2(sin, cos) - 2.0 * speed
        return torch.where(cos > 0.9, hold, pump).clamp(-2, 2).float()[:, None]
    if domain == "acrobot":
        return (torch.sign(o[:, 5]) + 1).long()[:, None]
    if domain == "mountain_car":
        return torch.where(o[:, 1] >= 0, 2, 0).long()[:, None]
    return torch.tanh(200.0 * o[:, 1]).float()[:, None]


#: Largest State deviation from Gymnasium allowed in a closed-loop rollout.
ROLLOUT_ATOL = 1e-6
#: Acrobot is chaotic: rounding differences of ``sin`` and ``cos`` grow until a bang-bang action flips, so
#: its deviation is bounded only over its first steps (Torch on the CPU also diverges after about 120).
DEVIATION_STEPS = {"acrobot": 100}


@pytest.mark.parametrize("domain", DOMAINS)
def test_closed_loop_rollouts_keep_the_gymnasium_outcome(domain):
    """300 steps of a scripted policy on each side: the same lanes reach the goal, with a bounded State deviation.

    Where the dynamics are not chaotic, each lane also ends at the same step.
    """
    reference, fast = _pair(domain, seed=11)
    keys = [key for key, (_, dtype) in CLASSIC_FIELDS[domain].items() if dtype == torch.float64]
    try:
        theirs = ours = None
        ended = {"gym": torch.full((LANES,), -1), "torch": torch.full((LANES,), -1)}
        deviation = 0.0
        for t in range(300):
            a_ref = _policy(domain, theirs["observation"]) if theirs is not None else _policy0(domain)
            a_fast = _policy(domain, ours["observation"].cpu()) if ours is not None else _policy0(domain)
            theirs, ours = reference.step(a_ref), fast.step(a_fast.cuda())
            for name, out in (("gym", theirs), ("torch", ours)):
                done = out["terminated"].cpu() & (ended[name] < 0)
                ended[name][done] = t
            if t < DEVIATION_STEPS.get(domain, 300):
                state = ours["state"].cpu()
                deviation = max(deviation, max(float((state[k] - theirs["state"][k]).abs().max()) for k in keys))
        assert torch.equal(ended["torch"] >= 0, ended["gym"] >= 0), ended
        if domain not in DEVIATION_STEPS:
            assert torch.equal(ended["torch"], ended["gym"]), ended
        assert deviation < ROLLOUT_ATOL, deviation
    finally:
        reference.close()
        fast.close()


def _policy0(domain):
    if domain in ("pendulum", "mountain_car_continuous"):
        return torch.zeros(LANES, 1)
    return torch.ones(LANES, 1, dtype=torch.long)
