"""Fused MuJoCo Warp tree passes: subtree ranges, agreement with MuJoCo Warp's level loops, fallback and routing."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

pytest.importorskip("warp")

import mujoco

from stable_envs._execution import mjwarp_fused
from stable_envs._execution.mjbatch import WarpBatch

# Two branches of different depth under the world, a free body, and a branch inside the arm.
XML = """
<mujoco>
  <option timestep="0.004"/>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1"/>
    <body name="b1" pos="0 0 1.5">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.2"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04" mass="1"/>
      <body name="b2" pos="0.3 0 0">
        <joint type="hinge" axis="1 0 0" damping="0.2"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.7"/>
        <body name="b3" pos="0.3 0 0">
          <joint type="hinge" axis="0 0 1" damping="0.2"/>
          <geom type="box" size="0.05 0.04 0.03" mass="0.4"/>
          <body name="f1" pos="0.05 0.03 0">
            <joint type="slide" axis="0 1 0" range="0 0.04" damping="0.5"/>
            <geom type="box" size="0.02 0.01 0.02" mass="0.1"/>
          </body>
          <body name="f2" pos="0.05 -0.03 0">
            <joint type="slide" axis="0 -1 0" range="0 0.04" damping="0.5"/>
            <geom type="box" size="0.02 0.01 0.02" mass="0.1"/>
          </body>
        </body>
        <body name="side" pos="0.1 0.1 0">
          <joint type="ball" damping="0.1"/>
          <geom type="sphere" size="0.05" mass="0.2"/>
        </body>
      </body>
    </body>
    <body name="box" pos="0.6 0.5 0.2">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="0.3"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" gear="5"/>
  </actuator>
</mujoco>
"""
B = 8


def _cuda_problem() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA required"
    pytest.importorskip("mujoco_warp")
    from stable_envs._core.warp_cuda import warp_cuda_problem

    return warp_cuda_problem()


def cuda(fn):
    problem = _cuda_problem() if torch.cuda.is_available() else "CUDA required"
    return pytest.mark.gpu(pytest.mark.skipif(problem is not None, reason=str(problem))(fn))


def supported_cuda(fn):
    reason = "installed MuJoCo Warp is not a version the fused passes support"
    return cuda(pytest.mark.skipif(not mjwarp_fused.supported(), reason=reason)(fn))


def _model():
    return mujoco.MjModel.from_xml_string(XML)


def test_subtree_end_gives_contiguous_subtree_ranges():
    parent = _model().body_parentid
    end = mjwarp_fused.subtree_end(parent)
    assert end is not None
    for body in range(len(parent)):
        inside = set(range(body, int(end[body])))
        descendants = {k for k in range(len(parent)) if _is_ancestor(parent, body, k)}
        assert inside == descendants


def test_subtree_end_rejects_a_subtree_that_is_not_contiguous():
    # Body 4 is a child of body 2, but body 3, between them, is not in the subtree of body 2.
    assert mjwarp_fused.subtree_end(np.array([0, 0, 1, 0, 2])) is None
    assert mjwarp_fused.subtree_end(np.array([0, 2, 0])) is None


def _is_ancestor(parent, body, k):
    while True:
        if k == body:
            return True
        if k == 0:
            return False
        k = int(parent[k])


def test_unsupported_version_runs_mujoco_warp_unchanged(monkeypatch):
    smooth = pytest.importorskip("mujoco_warp._src.smooth")
    stock = (smooth.com_pos, smooth.crb, smooth._rne_cfrc_backward)
    monkeypatch.setattr(mjwarp_fused.metadata, "version", lambda name: "0.0.1")
    assert not mjwarp_fused.supported()
    fake = types.SimpleNamespace(body_parentid=types.SimpleNamespace(numpy=lambda: np.array([0, 0, 1])))
    with mjwarp_fused.tree_walks(fake) as on:
        assert on is None
        assert (smooth.com_pos, smooth.crb, smooth._rne_cfrc_backward) == stock


def test_non_contiguous_model_runs_mujoco_warp_unchanged(monkeypatch):
    smooth = pytest.importorskip("mujoco_warp._src.smooth")
    stock = (smooth.com_pos, smooth.crb, smooth._rne_cfrc_backward)
    monkeypatch.setattr(mjwarp_fused, "supported", lambda: True)
    fake = types.SimpleNamespace(body_parentid=types.SimpleNamespace(numpy=lambda: np.array([0, 0, 1, 0, 2])))
    with mjwarp_fused.tree_walks(fake) as on:
        assert on is None
        assert (smooth.com_pos, smooth.crb, smooth._rne_cfrc_backward) == stock
    with mjwarp_fused.tree_walks(fake, enabled=False) as on:
        assert on is None


@supported_cuda
def test_fused_passes_agree_with_mujoco_warp_on_random_states():
    import mujoco_warp as mjw
    import warp as wp
    from mujoco_warp._src import smooth

    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    stock = (smooth.com_pos, smooth.crb, smooth._rne_cfrc_backward)
    fields = ("subtree_com", "cinert", "cdof", "crb", "M", "qfrc_bias", "cfrc_int")
    rng = np.random.default_rng(0)
    with wp.ScopedDevice("cuda:0"):
        m = mjw.put_model(model)
        d = mjw.put_data(model, data, nworld=B)
        qpos = torch.as_tensor(np.tile(model.qpos0, (B, 1)), device="cuda", dtype=torch.float32)
        for _ in range(4):
            state = {
                "qpos": qpos + 0.3 * torch.as_tensor(rng.standard_normal(qpos.shape), device="cuda"),
                "qvel": torch.as_tensor(rng.standard_normal((B, model.nv)), device="cuda", dtype=torch.float32),
            }
            results = []
            for fused in (False, True):
                for name, value in state.items():
                    wp.to_torch(getattr(d, name)).copy_(value)
                with mjwarp_fused.tree_walks(m, fused) as on:
                    assert (on is not None) is fused
                    # Without the constraint solver the passes are compared alone.
                    smooth.kinematics(m, d)
                    smooth.com_pos(m, d)
                    smooth.crb(m, d)
                    smooth.com_vel(m, d)
                    smooth.rne(m, d)
                wp.synchronize()
                results.append({f: wp.to_torch(getattr(d, f)).double().clone() for f in fields})
            for f in fields:
                scale = results[0][f].abs().max().clamp_min(1e-6)
                gap = (results[0][f] - results[1][f]).abs().max() / scale
                assert gap < 2e-6, (f, float(gap))
    assert (smooth.com_pos, smooth.crb, smooth._rne_cfrc_backward) == stock


def _rollout(fused_tree_max_envs, steps=300):
    with WarpBatch(_model(), B, "cuda", frame_skip=4, fused_tree_max_envs=fused_tree_max_envs) as batch:
        fused = batch.fused_tree
        phase = torch.arange(B, device="cuda")[:, None] * 0.7
        for t in range(steps):
            batch.step(torch.sin(0.03 * t + phase))
        batch.check_buffers()
        return fused, batch.data("qpos").double().cpu(), batch.data("xpos").double().cpu()


@supported_cuda
def test_fused_tree_max_envs_routes_the_graph_and_rollouts_agree():
    stock, qpos_stock, xpos_stock = _rollout(B - 1)
    fused, qpos_fused, xpos_fused = _rollout(B)
    assert stock is False and fused is True
    assert (qpos_fused - qpos_stock).abs().max() < 1e-3
    assert (xpos_fused - xpos_stock).abs().max() < 1e-3


@cuda
def test_fused_tree_follows_the_installed_version(monkeypatch):
    monkeypatch.setattr(mjwarp_fused.metadata, "version", lambda name: "0.0.1")
    with WarpBatch(_model(), 2, "cuda") as batch:
        assert batch.fused_tree is False
