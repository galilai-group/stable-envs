"""MSHumanLocomotion against a direct MuJoCo Warp rollout of the pinned model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import stable_envs as se

LANES, STEPS, SEED = 3, 12, 7
_FIELDS = (("qpos", "qpos"), ("qvel", "qvel"), ("act", "act"), ("warmstart", "qacc_warmstart"))


def _reference(actions):
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    from stable_envs.envs.mshuman.assets import model

    m = model()
    data = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, data, 0)
    mujoco.mj_forward(m, data)
    mw = mjw.put_model(m)
    d = mjw.put_data(m, data, nworld=LANES, nconmax=LANES * m.nconmax, njmax=LANES * m.njmax)
    mjw.forward(mw, d)
    wp.synchronize()
    out = []
    for action in actions:
        wp.to_torch(d.ctrl).copy_(torch.as_tensor(action, dtype=wp.to_torch(d.ctrl).dtype))
        mjw.step(mw, d)
        wp.synchronize()
        out.append({key: wp.to_torch(getattr(d, src)).clone() for key, src in _FIELDS})
    return out


def test_matches_a_plain_mujoco_warp_rollout():
    pytest.importorskip("mujoco_warp")
    rng = np.random.default_rng(SEED)
    env = se.make("MSHumanLocomotion", num_envs=LANES, device="cpu")
    try:
        actions = [env.action_spec.sample(LANES, rng) for _ in range(STEPS)]
        env.reset(seed=SEED)
        expected = _reference(actions)
        for action, want in zip(actions, expected):
            state = env.step(action, frames=None)["state"]
            for key, got in ((k, state[k]) for k in want):
                torch.testing.assert_close(got, want[key].to(got), rtol=0, atol=0)
    finally:
        env.close()
