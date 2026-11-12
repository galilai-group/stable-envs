"""The CUDA build of the Numba Rocket kernel against the CPU build, over closed-loop rollouts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("numba")

from stable_envs.envs.rocket.simulation import numba_physics as nk  # noqa: E402
from tests.helpers.rocket_policies import policy  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

KINDS = ["free", "burn", "smooth", "land", "pad", "firm", "hover", "climb"]
SEEDS = 4
MAX_STEPS = 1300
BODY = ("position", "quaternion", "linear_velocity", "angular_velocity")


def start(kinds, seed):
    """Start rows like the source's ``environment`` variation, or its default start for the landing policies."""
    rng = np.random.default_rng(seed)
    num = len(kinds)
    varied = np.array([kind not in ("land", "pad") for kind in kinds])[:, None]
    start_pos = np.column_stack([rng.uniform(-2, 2, (num, 2)), 500.0 * rng.uniform(0.7, 0.95, num)])
    start_pos = np.where(varied, start_pos, [0.0, 0.0, 450.0])
    return {
        "start_pos": start_pos,
        "start_orn": np.column_stack([rng.uniform(-0.2, 0.2, (num, 2)), np.zeros(num)]) * varied,
        "position": start_pos.copy(),
        "linear_velocity": np.tile([0.0, 0.0, -30.0], (num, 1)),
        "angular_velocity": np.zeros((num, 3)),
    }


def cpu_rollouts(kinds):
    """Closed-loop CPU kernel rollouts: actions, body states, outcomes and end-of-episode generator rows."""
    num = len(kinds)
    state = nk.new_state(num)
    state["rng"][:] = nk.rng_words([np.random.default_rng(100 + b) for b in range(num)])
    everyone = np.ones(num, bool)
    nk.start_state(state, everyone, start(kinds, 0))
    nk.settle(state, None, everyone)
    actions, bodies, live = [], [], everyone.copy()
    for t in range(MAX_STEPS):
        if not live.any():
            break
        view = [SimpleNamespace(**{k: state[k][b] for k in ("lin_pos", "ground_lin_vel", "ang_pos", "ang_vel")},
                                ignition_state=state["ignition_state"][b]) for b in range(num)]  # fmt: skip
        action = np.stack([policy(kinds[b], t, view[b]) for b in range(num)])
        nk.step(state, action, None, live)
        actions.append(action)
        bodies.append(np.concatenate([state[k] for k in BODY], 1))
        live &= ~(state["termination"] | state["truncation"])
    return np.stack(actions), np.stack(bodies), state


@pytest.fixture(scope="module")
def rollouts():
    from stable_envs.envs.rocket.simulation import numba_cuda_physics as gk

    kinds = [k for k in KINDS for _ in range(SEEDS)]
    actions, bodies, cpu = cpu_rollouts(kinds)
    num = len(kinds)
    state = gk.new_state(num)
    state["rng"][:] = torch.as_tensor(nk.rng_words([np.random.default_rng(100 + b) for b in range(num)]))
    everyone = torch.ones(num, dtype=torch.bool, device="cuda")
    gk.start_state(state, everyone, {k: torch.as_tensor(v) for k, v in start(kinds, 0).items()})
    gk.settle(state, None, everyone)
    gpu_bodies = []
    live = everyone.clone()
    for t in range(len(actions)):
        gk.step(state, torch.as_tensor(actions[t], device="cuda"), None, live)
        live &= ~(state["termination"] | state["truncation"])
        gpu_bodies.append(torch.cat([state[k] for k in BODY], 1).cpu().numpy())
    gpu = {k: v.cpu().numpy() for k, v in state.items()}
    return kinds, bodies, np.stack(gpu_bodies), cpu, gpu


def test_cuda_kernel_ends_every_episode_as_the_cpu_kernel(rollouts):
    kinds, _, _, cpu, gpu = rollouts
    for key in ("step_count", "termination", "truncation", "fatal_collision", "env_complete", "out_of_bounds"):
        np.testing.assert_array_equal(gpu[key], cpu[key], err_msg=key)
    # the same thrust noise draws, from the same per-environment streams
    np.testing.assert_array_equal(gpu["rng"], cpu["rng"])
    assert {kind for kind, done in zip(kinds, cpu["env_complete"], strict=True) if done} >= {"land", "pad"}


def test_cuda_kernel_follows_the_cpu_kernel(rollouts):
    _, bodies, gpu_bodies, cpu, gpu = rollouts
    # GPU code contracts multiply-adds; free flight agrees to about 1e-12, contacts stay within 1e-6
    np.testing.assert_allclose(gpu_bodies, bodies, rtol=0, atol=1e-6)
    for key in ("state", "fuel", "throttle", "fin", "gimbal", "reward"):
        np.testing.assert_allclose(gpu[key], cpu[key], rtol=0, atol=1e-6, err_msg=key)


def test_cuda_kernel_skips_inactive_and_ended_environments():
    from stable_envs.envs.rocket.simulation import numba_cuda_physics as gk

    state = gk.new_state(3)
    everyone = torch.ones(3, dtype=torch.bool, device="cuda")
    gk.start_state(state, everyone, {k: torch.as_tensor(v) for k, v in start(["free"] * 3, 1).items()})
    gk.settle(state, torch.zeros(3, nk.NOISE_PER_SETTLE, device="cuda", dtype=torch.float64), everyone)
    state["termination"][2] = True
    before = state["position"].clone()
    active = torch.tensor([True, False, True], device="cuda")
    reward, term, _, info = gk.step(state, torch.zeros(3, 7, device="cuda"), torch.zeros(3, 6, device="cuda"), active)
    assert torch.equal(state["step_count"].cpu(), torch.tensor([1, 0, 1]))
    assert torch.equal(info["noise_used"].cpu(), torch.tensor([nk.NOISE_PER_STEP, 0, 0]))
    assert not torch.equal(state["position"][0], before[0])
    assert torch.equal(state["position"][1:], before[1:])
    assert reward[2].item() == 0.0 and term[2].item()
