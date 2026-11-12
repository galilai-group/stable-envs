"""Fused kinematic-tree passes for MuJoCo Warp: one kernel launch per pass instead of one per tree level."""

from __future__ import annotations

import contextlib
from importlib import metadata

import numpy as np
import torch
import warp as wp

#: MuJoCo Warp versions whose ``com_pos``, ``crb`` and ``_rne_cfrc_backward`` these passes replace.
SUPPORTED_VERSIONS = ("3.12.0", "3.17.0")

wp.set_module_options({"enable_backward": False})


class _vec10(wp.types.vector(length=10, dtype=float)):
    pass


@wp.kernel
def _subtree_com_sum(
    body_subtree_end: wp.array[int],
    body_mass: wp.array2d[float],
    body_subtreemass: wp.array2d[float],
    xipos_in: wp.array2d[wp.vec3],
    subtree_com_out: wp.array2d[wp.vec3],
):
    worldid, bodyid = wp.tid()
    massid = worldid % body_mass.shape[0]
    com = wp.vec3(0.0)
    for k in range(bodyid, body_subtree_end[bodyid]):
        com += xipos_in[worldid, k] * body_mass[massid, k]
    mass = body_subtreemass[worldid % body_subtreemass.shape[0], bodyid]
    if mass != 0.0:
        com = com / mass
    subtree_com_out[worldid, bodyid] = com


@wp.kernel
def _crb_sum(
    body_subtree_end: wp.array[int],
    cinert_in: wp.array2d[_vec10],
    crb_out: wp.array2d[_vec10],
):
    worldid, bodyid = wp.tid()
    # Bodies attached to the world do not add to it.
    end = body_subtree_end[bodyid]
    if bodyid == 0:
        end = 1
    crb = _vec10()
    for k in range(bodyid, end):
        crb += cinert_in[worldid, k]
    crb_out[worldid, bodyid] = crb


@wp.kernel
def _cfrc_sum(
    body_subtree_end: wp.array[int],
    cfrc_in: wp.array2d[wp.spatial_vector],
    cfrc_int_out: wp.array2d[wp.spatial_vector],
):
    worldid, bodyid = wp.tid()
    frc = wp.spatial_vector()
    for k in range(bodyid, body_subtree_end[bodyid]):
        frc += cfrc_in[worldid, k]
    cfrc_int_out[worldid, bodyid] = frc


def subtree_end(parentid: np.ndarray) -> np.ndarray | None:
    """One past the last body of each body's subtree, or None if a subtree is not a contiguous range of ids."""
    parentid = np.asarray(parentid, dtype=np.int64)
    nbody = len(parentid)
    end, size = np.arange(1, nbody + 1), np.ones(nbody, dtype=np.int64)
    for body in range(nbody - 1, 0, -1):
        parent = parentid[body]
        if parent >= body:
            return None
        end[parent] = max(end[parent], end[body])
        size[parent] += size[body]
    return end if np.array_equal(end - np.arange(nbody), size) else None


class TreeWalks:
    """Device arrays that graphs captured under :func:`tree_walks` read: keep it as long as those graphs."""

    def __init__(self, body_subtree_end: wp.array):
        self.body_subtree_end = body_subtree_end
        self.buffers: dict[int, torch.Tensor] = {}


def _patches(smooth, model, owner: TreeWalks):
    stock = {name: getattr(smooth, name) for name in ("com_pos", "crb", "_rne_cfrc_backward")}
    body_subtree_end, scratch = owner.body_subtree_end, owner.buffers

    def com_pos(m, d):
        if m is not model:
            return stock["com_pos"](m, d)
        wp.launch(
            _subtree_com_sum,
            dim=(d.nworld, m.nbody),
            inputs=[body_subtree_end, m.body_mass, m.body_subtreemass, d.xipos],
            outputs=[d.subtree_com],
        )
        wp.launch(
            smooth._cinert,
            dim=(d.nworld, m.nbody),
            inputs=[m.body_rootid, m.body_mass, m.body_inertia, d.xipos, d.ximat, d.subtree_com],
            outputs=[d.cinert],
        )
        wp.launch(
            smooth._cdof,
            dim=(d.nworld, m.njnt),
            inputs=[m.body_rootid, m.jnt_type, m.jnt_dofadr, m.jnt_bodyid, d.xmat, d.xanchor, d.xaxis, d.subtree_com],
            outputs=[d.cdof],
        )

    def crb(m, d):
        if m is not model:
            return stock["crb"](m, d)
        wp.launch(_crb_sum, dim=(d.nworld, m.nbody), inputs=[body_subtree_end, d.cinert], outputs=[d.crb])
        d.M.zero_()
        wp.launch(
            smooth._M,
            dim=(d.nworld, m.nv),
            inputs=[m.dof_bodyid, m.dof_parentid, m.dof_armature, m.M_rownnz, m.M_rowadr, d.cdof, d.crb],
            outputs=[d.M],
        )

    def _rne_cfrc_backward(m, d):
        if m is not model:
            return stock["_rne_cfrc_backward"](m, d)
        buffer = scratch.get(d.cfrc_int.ptr)
        if buffer is None or tuple(buffer.shape[:2]) != tuple(d.cfrc_int.shape):
            # Torch memory, outside the Warp allocator that a caller may scope around a capture.
            buffer = torch.empty((*d.cfrc_int.shape, 6), dtype=torch.float32, device=str(d.cfrc_int.device))
            scratch[d.cfrc_int.ptr] = buffer
        cfrc = wp.from_torch(buffer, dtype=wp.spatial_vector)
        wp.copy(cfrc, d.cfrc_int)
        wp.launch(_cfrc_sum, dim=(d.nworld, m.nbody), inputs=[body_subtree_end, cfrc], outputs=[d.cfrc_int])

    return stock, {"com_pos": com_pos, "crb": crb, "_rne_cfrc_backward": _rne_cfrc_backward}


def supported() -> bool:
    """Whether the installed MuJoCo Warp is a version whose tree passes :func:`tree_walks` replaces."""
    try:
        return metadata.version("mujoco-warp") in SUPPORTED_VERSIONS
    except metadata.PackageNotFoundError:
        return False


@contextlib.contextmanager
def tree_walks(m, enabled: bool = True):
    """Within the block, MuJoCo Warp computes subtree centers of mass, composite inertias and RNE internal forces
    of the Warp model ``m`` with one launch per pass, instead of one launch per body tree level.

    Each thread sums one body's subtree, which MuJoCo stores as a contiguous range of body ids. The results are the
    same up to float32 summation order, and do not depend on atomic ordering. Use it around a graph capture: the
    captured graph keeps the fused kernels after the block ends. Without a supported MuJoCo Warp version, with
    ``enabled=False``, or for other models, MuJoCo Warp runs unchanged.

    Yields a :class:`TreeWalks` that owns the arrays the fused kernels read, or None when MuJoCo Warp runs
    unchanged. Keep it as long as the graphs captured in the block.
    """
    end = subtree_end(m.body_parentid.numpy()) if enabled and supported() else None
    if end is None:
        yield None
        return
    from mujoco_warp._src import smooth

    owner = TreeWalks(wp.array(end, dtype=int, device=m.body_parentid.device))
    stock, patches = _patches(smooth, m, owner)
    for name, fn in patches.items():
        setattr(smooth, name, fn)
    try:
        yield owner
    finally:
        for name, fn in stock.items():
            setattr(smooth, name, fn)
