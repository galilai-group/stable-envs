"""MSHumanLocomotion against a direct CPU MuJoCo (float64) rollout of the pinned model."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

import stable_envs as se

LANES, STEPS, SEED = 3, 12, 7
#: The CPU env steps a native mujoco.batch; worst gaps over 3,000 random trajectories were under 1.5e-5, 3e-3,
#: 1.2e-7 and 2e-9 (docs/environments/mshuman.md), and this fixed rollout's are 2.4e-7, 7.6e-5, 6.0e-8 and 1.9e-9.
_ATOL = {"qpos": 1e-5, "qvel": 2e-3, "act": 1e-6, "time": 1e-8}


def _reference(actions):
    from stable_envs.envs.mshuman.assets import model

    m = model()
    lanes = [mujoco.MjData(m) for _ in range(LANES)]
    for d in lanes:
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
    out = []
    for action in actions:
        for d, ctrl in zip(lanes, np.asarray(action)):
            d.ctrl[:] = ctrl
            mujoco.mj_step(m, d)
        out.append(
            {
                key: torch.tensor(
                    np.stack([np.atleast_1d(getattr(d, key)).copy() for d in lanes]), dtype=torch.float32
                )
                for key in _ATOL
            }
        )
    return out


def test_matches_a_plain_mujoco_rollout():
    pytest.importorskip("mujoco_warp")
    rng = np.random.default_rng(SEED)
    env = se.make("MSHumanLocomotion", num_envs=LANES, device="cpu")
    try:
        actions = [env.action_spec.sample(LANES, rng) for _ in range(STEPS)]
        env.reset(seed=SEED)
        for action, want in zip(actions, _reference(actions)):
            state = env.step(action, frames=None)["state"]
            for key, atol in _ATOL.items():
                torch.testing.assert_close(state[key], want[key], rtol=0, atol=atol)
    finally:
        env.close()
