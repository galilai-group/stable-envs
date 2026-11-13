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


#: Largest body count that the block-per-world passes take: one thread per body in a CUDA block.
MAX_BLOCK_BODIES = 1024


class TreeWalks:
    """Device arrays that graphs captured under :func:`tree_walks` read: keep it as long as those graphs."""

    def __init__(self, body_subtree_end: wp.array, body_depth: wp.array | None = None, nlevel: int = 0):
        self.body_subtree_end = body_subtree_end
        self.body_depth = body_depth
        self.nlevel = nlevel
        self.buffers: dict[int, torch.Tensor] = {}
        self.geom_movebody: wp.array | None = None
        # The Data whose com_pos and composite inertias the last kinematics launch computed.
        self.com_pos_done = self.crb_done = None


def _scratch(owner: TreeWalks, d) -> wp.array:
    buffer = owner.buffers.get(d.cfrc_int.ptr)
    if buffer is None or tuple(buffer.shape[:2]) != tuple(d.cfrc_int.shape):
        # Torch memory, outside the Warp allocator that a caller may scope around a capture.
        buffer = torch.empty((*d.cfrc_int.shape, 6), dtype=torch.float32, device=str(d.cfrc_int.device))
        owner.buffers[d.cfrc_int.ptr] = buffer
    return wp.from_torch(buffer, dtype=wp.spatial_vector)


def _block_patches(smooth, model, owner: TreeWalks, base: dict):
    """Kinematics (with ``com_pos`` and ``crb``), ``com_vel`` and RNE with one CUDA block per world.

    ``base`` maps a name to the function to run for other models; names it does not have run MuJoCo Warp's.
    """
    from mujoco_warp._src.types import DisableBit

    from stable_envs._execution import mjwarp_tree

    names = ("kinematics", "com_pos", "crb", "com_vel", "rne", "_rne_cacc_forward")
    stock = {name: base.get(name, getattr(smooth, name)) for name in names}
    depth, nlevel, end = owner.body_depth, owner.nlevel, owner.body_subtree_end
    nblock = -(-model.nbody // 32) * 32
    # The kinematics kernel spreads per-body, per-geom, per-site and per-joint work over a larger block.
    nitems = max(2 * model.nbody + model.ngeom + model.nsite, model.nbody + model.njnt)
    nblock_kin = max(nblock, min(256, -(-nitems // 32) * 32))
    bodyid, weldid = model.geom_bodyid.numpy(), model.body_weldid.numpy()
    rootid, mocapid = model.body_rootid.numpy(), model.body_mocapid.numpy()
    moves = (weldid[bodyid] != 0) | (mocapid[rootid[bodyid]] != -1)
    owner.geom_movebody = wp.array(np.where(moves, bodyid, -1), dtype=int, device=model.geom_bodyid.device)

    def launch(kernel, d, inputs, outputs, size=nblock):
        wp.launch(kernel, dim=d.nworld * size, inputs=inputs, outputs=outputs, block_dim=size)

    def kinematics(m, d):
        if m is not model:
            return stock["kinematics"](m, d)
        launch(
            mjwarp_tree._kinematics_tree,
            d,
            [
                nblock_kin,
                nlevel,
                m.nbody,
                m.njnt,
                m.ngeom,
                m.nsite,
                depth,
                end,
                owner.geom_movebody,
                m.qpos0,
                m.body_parentid,
                m.body_rootid,
                m.body_mocapid,
                m.body_jntnum,
                m.body_jntadr,
                m.body_pos,
                m.body_quat,
                m.body_ipos,
                m.body_iquat,
                m.body_mass,
                m.body_subtreemass,
                m.body_inertia,
                m.jnt_type,
                m.jnt_qposadr,
                m.jnt_dofadr,
                m.jnt_bodyid,
                m.jnt_pos,
                m.jnt_axis,
                m.geom_pos,
                m.geom_quat,
                m.site_bodyid,
                m.site_pos,
                m.site_quat,
                d.qpos,
                d.mocap_pos,
                d.mocap_quat,
            ],
            [
                d.xpos,
                d.xquat,
                d.xmat,
                d.xipos,
                d.ximat,
                d.xanchor,
                d.xaxis,
                d.geom_xpos,
                d.geom_xmat,
                d.site_xpos,
                d.site_xmat,
                d.subtree_com,
                d.cinert,
                d.cdof,
                d.crb,
            ],
            nblock_kin,
        )
        owner.com_pos_done = owner.crb_done = d

    def com_pos(m, d):
        if m is model and owner.com_pos_done is d:
            # The kinematics kernel right before computed it from the same poses.
            owner.com_pos_done = None
            return None
        return stock["com_pos"](m, d)

    def crb(m, d):
        if m is not model or owner.crb_done is not d:
            return stock["crb"](m, d)
        # The kinematics kernel computed the composite inertias; the mass matrix remains.
        owner.crb_done = None
        d.M.zero_()
        wp.launch(
            smooth._M,
            dim=(d.nworld, m.nv),
            inputs=[m.dof_bodyid, m.dof_parentid, m.dof_armature, m.M_rownnz, m.M_rowadr, d.cdof, d.crb],
            outputs=[d.M],
        )

    def com_vel(m, d):
        if m is not model:
            return stock["com_vel"](m, d)
        launch(
            mjwarp_tree._comvel_tree,
            d,
            [
                nblock,
                nlevel,
                m.nbody,
                depth,
                m.body_parentid,
                m.body_jntnum,
                m.body_jntadr,
                m.body_dofnum,
                m.body_dofadr,
                m.jnt_type,
                d.qvel,
                d.cdof,
            ],
            [d.cvel, d.cdof_dot],
        )

    def rne_tree(m, d, flg_acc, world_acc, full):
        launch(
            mjwarp_tree._rne_tree,
            d,
            [
                nblock,
                nlevel,
                m.nbody,
                m.nv,
                world_acc,
                full,
                depth,
                end,
                m.opt.gravity,
                m.body_parentid,
                m.body_dofnum,
                m.body_dofadr,
                m.dof_bodyid,
                d.qvel,
                d.qacc,
                d.cdof,
                d.cdof_dot,
                d.cinert,
                d.cvel,
                flg_acc,
                _scratch(owner, d),
            ],
            [d.cacc, d.cfrc_int, d.qfrc_bias],
        )

    def rne(m, d, flg_acc=False):
        if m is not model:
            return stock["rne"](m, d, flg_acc=flg_acc)
        rne_tree(m, d, flg_acc, 1 if m.opt.disableflags & DisableBit.GRAVITY else 2, True)

    def _rne_cacc_forward(m, d, flg_acc=False):
        if m is not model:
            return stock["_rne_cacc_forward"](m, d, flg_acc=flg_acc)
        rne_tree(m, d, flg_acc, 0, False)

    patches = {"kinematics": kinematics, "com_pos": com_pos, "crb": crb, "com_vel": com_vel, "rne": rne}
    patches["_rne_cacc_forward"] = _rne_cacc_forward
    return patches


def _patches(smooth, model, owner: TreeWalks):
    stock = {name: getattr(smooth, name) for name in ("com_pos", "crb", "_rne_cfrc_backward")}
    body_subtree_end = owner.body_subtree_end

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
        cfrc = _scratch(owner, d)
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
def tree_walks(m, enabled: bool = True, block: bool = True):
    """Within the block, MuJoCo Warp computes subtree centers of mass, composite inertias and RNE internal forces
    of the Warp model ``m`` with one launch per pass, instead of one launch per body tree level.

    Each thread sums one body's subtree, which MuJoCo stores as a contiguous range of body ids. The results are the
    same up to float32 summation order, and do not depend on atomic ordering. Use it around a graph capture: the
    captured graph keeps the fused kernels after the block ends. Without a supported MuJoCo Warp version, with
    ``enabled=False``, or for other models, MuJoCo Warp runs unchanged.

    With ``block``, on CUDA, models of up to :data:`MAX_BLOCK_BODIES` bodies run kinematics, ``com_vel`` and RNE
    with one thread block per world, one thread per body, and a block barrier between tree levels: one launch per
    pass. The kinematics launch also computes ``com_pos`` and the composite inertias of ``crb``; the calls of these
    two functions that follow it skip that work.

    Yields a :class:`TreeWalks` that owns the arrays the fused kernels read, or None when MuJoCo Warp runs
    unchanged. Keep it as long as the graphs captured in the block.
    """
    end = subtree_end(m.body_parentid.numpy()) if enabled and supported() else None
    if end is None:
        yield None
        return
    from mujoco_warp._src import smooth

    device = m.body_parentid.device
    parentid = m.body_parentid.numpy()
    depth = np.zeros(len(parentid), dtype=np.int64)
    for body in range(1, len(parentid)):
        depth[body] = depth[parentid[body]] + 1
    owner = TreeWalks(
        wp.array(end, dtype=int, device=device), wp.array(depth, dtype=int, device=device), int(depth.max())
    )
    stock, patches = _patches(smooth, m, owner)
    if block and wp.get_device(device).is_cuda and m.nbody <= MAX_BLOCK_BODIES:
        names = ("kinematics", "com_vel", "rne", "_rne_cacc_forward")
        stock.update({name: getattr(smooth, name) for name in names})
        patches.update(_block_patches(smooth, m, owner, dict(patches)))
    for name, fn in patches.items():
        setattr(smooth, name, fn)
    try:
        yield owner
    finally:
        for name, fn in stock.items():
            setattr(smooth, name, fn)
