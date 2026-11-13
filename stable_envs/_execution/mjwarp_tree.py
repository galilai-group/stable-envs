"""Block-per-world MuJoCo Warp tree kernels: one thread block walks one world's body tree level by level."""
# ruff: noqa: UP018  (Warp needs typed initial values for variables that change in branches and loops)

from __future__ import annotations

import warp as wp
from mujoco_warp._src import math
from mujoco_warp._src.types import JointType, vec10

wp.set_module_options({"enable_backward": False})

_SYNC = """
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
"""


@wp.func_native(_SYNC)
def _barrier(): ...


# Subtree sums over a contiguous range of body ids, four independent loads per round.


@wp.func
def _sum_com(
    xipos: wp.array2d[wp.vec3], mass: wp.array2d[float], worldid: int, massid: int, start: int, end: int
) -> wp.vec3:
    s0 = wp.vec3(0.0)
    s1 = wp.vec3(0.0)
    s2 = wp.vec3(0.0)
    s3 = wp.vec3(0.0)
    last = end - 1
    for k in range(start, end, 4):
        k1 = wp.min(k + 1, last)
        k2 = wp.min(k + 2, last)
        k3 = wp.min(k + 3, last)
        s0 += xipos[worldid, k] * mass[massid, k]
        s1 += wp.where(k + 1 <= last, xipos[worldid, k1] * mass[massid, k1], wp.vec3(0.0))
        s2 += wp.where(k + 2 <= last, xipos[worldid, k2] * mass[massid, k2], wp.vec3(0.0))
        s3 += wp.where(k + 3 <= last, xipos[worldid, k3] * mass[massid, k3], wp.vec3(0.0))
    return (s0 + s1) + (s2 + s3)


@wp.func
def _sum_vec10(values: wp.array2d[vec10], worldid: int, start: int, end: int) -> vec10:
    s0 = vec10()
    s1 = vec10()
    s2 = vec10()
    s3 = vec10()
    last = end - 1
    for k in range(start, end, 4):
        s0 += values[worldid, k]
        s1 += wp.where(k + 1 <= last, values[worldid, wp.min(k + 1, last)], vec10())
        s2 += wp.where(k + 2 <= last, values[worldid, wp.min(k + 2, last)], vec10())
        s3 += wp.where(k + 3 <= last, values[worldid, wp.min(k + 3, last)], vec10())
    return (s0 + s1) + (s2 + s3)


@wp.func
def _sum_spatial(values: wp.array2d[wp.spatial_vector], worldid: int, start: int, end: int) -> wp.spatial_vector:
    s0 = wp.spatial_vector()
    s1 = wp.spatial_vector()
    s2 = wp.spatial_vector()
    s3 = wp.spatial_vector()
    last = end - 1
    for k in range(start, end, 4):
        s0 += values[worldid, k]
        s1 += wp.where(k + 1 <= last, values[worldid, wp.min(k + 1, last)], wp.spatial_vector())
        s2 += wp.where(k + 2 <= last, values[worldid, wp.min(k + 2, last)], wp.spatial_vector())
        s3 += wp.where(k + 3 <= last, values[worldid, wp.min(k + 3, last)], wp.spatial_vector())
    return (s0 + s1) + (s2 + s3)


@wp.kernel
def _kinematics_tree(
    nblock: int,
    nlevel: int,
    nbody: int,
    njnt: int,
    ngeom: int,
    nsite: int,
    body_depth: wp.array[int],
    body_subtree_end: wp.array[int],
    geom_movebody: wp.array[int],
    # Model:
    qpos0: wp.array2d[float],
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_mocapid: wp.array[int],
    body_jntnum: wp.array[int],
    body_jntadr: wp.array[int],
    body_pos: wp.array2d[wp.vec3],
    body_quat: wp.array2d[wp.quat],
    body_ipos: wp.array2d[wp.vec3],
    body_iquat: wp.array2d[wp.quat],
    body_mass: wp.array2d[float],
    body_subtreemass: wp.array2d[float],
    body_inertia: wp.array2d[wp.vec3],
    jnt_type: wp.array[int],
    jnt_qposadr: wp.array[int],
    jnt_dofadr: wp.array[int],
    jnt_bodyid: wp.array[int],
    jnt_pos: wp.array2d[wp.vec3],
    jnt_axis: wp.array2d[wp.vec3],
    geom_pos: wp.array2d[wp.vec3],
    geom_quat: wp.array2d[wp.quat],
    site_bodyid: wp.array[int],
    site_pos: wp.array2d[wp.vec3],
    site_quat: wp.array2d[wp.quat],
    # Data in:
    qpos_in: wp.array2d[float],
    mocap_pos_in: wp.array2d[wp.vec3],
    mocap_quat_in: wp.array2d[wp.quat],
    # Data out:
    xpos_out: wp.array2d[wp.vec3],
    xquat_out: wp.array2d[wp.quat],
    xmat_out: wp.array2d[wp.mat33],
    xipos_out: wp.array2d[wp.vec3],
    ximat_out: wp.array2d[wp.mat33],
    xanchor_out: wp.array2d[wp.vec3],
    xaxis_out: wp.array2d[wp.vec3],
    geom_xpos_out: wp.array2d[wp.vec3],
    geom_xmat_out: wp.array2d[wp.mat33],
    site_xpos_out: wp.array2d[wp.vec3],
    site_xmat_out: wp.array2d[wp.mat33],
    subtree_com_out: wp.array2d[wp.vec3],
    cinert_out: wp.array2d[vec10],
    cdof_out: wp.array2d[wp.spatial_vector],
    crb_out: wp.array2d[vec10],
):
    tid = wp.tid()
    worldid = tid // nblock
    bodyid = tid % nblock
    active = bodyid < nbody

    # Body pose in the parent frame, joints applied: no data from other bodies, so all bodies at once.
    depth = int(0)
    free = bool(False)
    pid = int(-1)
    jntadr = int(0)
    jntnum = int(0)
    lpos = wp.vec3(0.0)
    lquat = wp.quat(0.0, 0.0, 0.0, 1.0)
    if active:
        depth = body_depth[bodyid]
        pid = body_parentid[bodyid]
        jntadr = body_jntadr[bodyid]
        jntnum = body_jntnum[bodyid]
        if jntnum == 1:
            if jnt_type[jntadr] == JointType.FREE:
                free = True
        if free:
            qadr = jnt_qposadr[jntadr]
            xpos = wp.vec3(qpos_in[worldid, qadr], qpos_in[worldid, qadr + 1], qpos_in[worldid, qadr + 2])
            xquat = wp.normalize(
                wp.quat(
                    qpos_in[worldid, qadr + 3],
                    qpos_in[worldid, qadr + 4],
                    qpos_in[worldid, qadr + 5],
                    qpos_in[worldid, qadr + 6],
                )
            )
            xpos_out[worldid, bodyid] = xpos
            xquat_out[worldid, bodyid] = xquat
            xanchor_out[worldid, jntadr] = xpos
            xaxis_out[worldid, jntadr] = jnt_axis[worldid % jnt_axis.shape[0], jntadr]
        else:
            mocapid = body_mocapid[bodyid]
            if mocapid >= 0:
                lpos = mocap_pos_in[worldid, mocapid]
                lquat = mocap_quat_in[worldid, mocapid]
            else:
                lpos = body_pos[worldid % body_pos.shape[0], bodyid]
                lquat = body_quat[worldid % body_quat.shape[0], bodyid]
            jnt_pos_id = worldid % jnt_pos.shape[0]
            for j in range(jntadr, jntadr + jntnum):
                qadr = jnt_qposadr[j]
                jtype = jnt_type[j]
                jpos = jnt_pos[jnt_pos_id, j]
                if jtype == JointType.BALL:
                    anchor = math.rot_vec_quat(jpos, lquat) + lpos
                    qloc = wp.normalize(
                        wp.quat(
                            qpos_in[worldid, qadr],
                            qpos_in[worldid, qadr + 1],
                            qpos_in[worldid, qadr + 2],
                            qpos_in[worldid, qadr + 3],
                        )
                    )
                    lquat = math.mul_quat(lquat, qloc)
                    lpos = anchor - math.rot_vec_quat(jpos, lquat)
                elif jtype == JointType.SLIDE:
                    axis = math.rot_vec_quat(jnt_axis[worldid % jnt_axis.shape[0], j], lquat)
                    lpos += axis * (qpos_in[worldid, qadr] - qpos0[worldid % qpos0.shape[0], qadr])
                elif jtype == JointType.HINGE:
                    anchor = math.rot_vec_quat(jpos, lquat) + lpos
                    angle = qpos_in[worldid, qadr] - qpos0[worldid % qpos0.shape[0], qadr]
                    lquat = math.mul_quat(
                        lquat, math.axis_angle_to_quat(jnt_axis[worldid % jnt_axis.shape[0], j], angle)
                    )
                    lpos = anchor - math.rot_vec_quat(jpos, lquat)
            if pid < 0:
                xpos_out[worldid, bodyid] = lpos
                xquat_out[worldid, bodyid] = wp.normalize(lquat)
    _barrier()

    # Compose with the parent, one tree level per round.
    for level in range(1, nlevel + 1):
        if active and not free and depth == level:
            ppos = xpos_out[worldid, pid]
            pquat = xquat_out[worldid, pid]
            xpos_out[worldid, bodyid] = math.rot_vec_quat(lpos, pquat) + ppos
            xquat_out[worldid, bodyid] = wp.normalize(math.mul_quat(pquat, lquat))
        _barrier()

    # Work that needs only the body poses, spread over the whole block: joint anchors and axes per body (from the
    # parent pose, as MuJoCo Warp computes them), body frames, geom and site poses.
    for item in range(bodyid, 2 * nbody + ngeom + nsite, nblock):
        if item < nbody:
            jadr = body_jntadr[item]
            jnum = body_jntnum[item]
            moving = jnum > 0
            if jnum == 1:
                if jnt_type[jadr] == JointType.FREE:
                    moving = False
            if moving:
                parent = body_parentid[item]
                ppos = xpos_out[worldid, parent]
                pquat = xquat_out[worldid, parent]
                xpos = math.rot_vec_quat(body_pos[worldid % body_pos.shape[0], item], pquat) + ppos
                xquat = math.mul_quat(pquat, body_quat[worldid % body_quat.shape[0], item])
                jnt_pos_id = worldid % jnt_pos.shape[0]
                for j in range(jadr, jadr + jnum):
                    qadr = jnt_qposadr[j]
                    jtype = jnt_type[j]
                    jaxis = jnt_axis[worldid % jnt_axis.shape[0], j]
                    jpos = jnt_pos[jnt_pos_id, j]
                    xanchor = math.rot_vec_quat(jpos, xquat) + xpos
                    xaxis = math.rot_vec_quat(jaxis, xquat)
                    if jtype == JointType.BALL:
                        qloc = wp.normalize(
                            wp.quat(
                                qpos_in[worldid, qadr],
                                qpos_in[worldid, qadr + 1],
                                qpos_in[worldid, qadr + 2],
                                qpos_in[worldid, qadr + 3],
                            )
                        )
                        xquat = math.mul_quat(xquat, qloc)
                        xpos = xanchor - math.rot_vec_quat(jpos, xquat)
                    elif jtype == JointType.SLIDE:
                        xpos += xaxis * (qpos_in[worldid, qadr] - qpos0[worldid % qpos0.shape[0], qadr])
                    elif jtype == JointType.HINGE:
                        angle = qpos_in[worldid, qadr] - qpos0[worldid % qpos0.shape[0], qadr]
                        xquat = math.mul_quat(xquat, math.axis_angle_to_quat(jaxis, angle))
                        xpos = xanchor - math.rot_vec_quat(jpos, xquat)
                    xanchor_out[worldid, j] = xanchor
                    xaxis_out[worldid, j] = xaxis
        elif item < 2 * nbody:
            b = item - nbody
            xpos = xpos_out[worldid, b]
            xquat = xquat_out[worldid, b]
            xmat_out[worldid, b] = math.quat_to_mat(xquat)
            xipos_out[worldid, b] = xpos + math.rot_vec_quat(body_ipos[worldid % body_ipos.shape[0], b], xquat)
            ximat_out[worldid, b] = math.quat_to_mat(
                math.mul_quat(xquat, body_iquat[worldid % body_iquat.shape[0], b])
            )
        elif item < 2 * nbody + ngeom:
            geomid = item - 2 * nbody
            # -1 for geoms welded to the world: they keep the pose that make_data gave them.
            gbody = geom_movebody[geomid]
            if gbody >= 0:
                xpos = xpos_out[worldid, gbody]
                xquat = xquat_out[worldid, gbody]
                geom_xpos_out[worldid, geomid] = xpos + math.rot_vec_quat(
                    geom_pos[worldid % geom_pos.shape[0], geomid], xquat
                )
                geom_xmat_out[worldid, geomid] = math.quat_to_mat(
                    math.mul_quat(xquat, geom_quat[worldid % geom_quat.shape[0], geomid])
                )
        else:
            siteid = item - 2 * nbody - ngeom
            sbody = site_bodyid[siteid]
            xpos = xpos_out[worldid, sbody]
            xquat = xquat_out[worldid, sbody]
            site_xpos_out[worldid, siteid] = xpos + math.rot_vec_quat(
                site_pos[worldid % site_pos.shape[0], siteid], xquat
            )
            site_xmat_out[worldid, siteid] = math.quat_to_mat(
                math.mul_quat(xquat, site_quat[worldid % site_quat.shape[0], siteid])
            )
    _barrier()

    # Subtree centers of mass: each body's subtree is a contiguous range of body ids.
    if active:
        com = _sum_com(xipos_out, body_mass, worldid, worldid % body_mass.shape[0], bodyid, body_subtree_end[bodyid])
        mass = body_subtreemass[worldid % body_subtreemass.shape[0], bodyid]
        if mass != 0.0:
            com = com / mass
        subtree_com_out[worldid, bodyid] = com
    _barrier()

    for item in range(bodyid, nbody + njnt, nblock):
        if item < nbody:
            b = item
            mat = ximat_out[worldid, b]
            inert = body_inertia[worldid % body_inertia.shape[0], b]
            mass = body_mass[worldid % body_mass.shape[0], b]
            dif = xipos_out[worldid, b] - subtree_com_out[worldid, body_rootid[b]]
            res = vec10()
            tmp = mat @ wp.diag(inert) @ wp.transpose(mat)
            res[0] = tmp[0, 0]
            res[1] = tmp[1, 1]
            res[2] = tmp[2, 2]
            res[3] = tmp[0, 1]
            res[4] = tmp[0, 2]
            res[5] = tmp[1, 2]
            res[0] += mass * (dif[1] * dif[1] + dif[2] * dif[2])
            res[1] += mass * (dif[0] * dif[0] + dif[2] * dif[2])
            res[2] += mass * (dif[0] * dif[0] + dif[1] * dif[1])
            res[3] -= mass * dif[0] * dif[1]
            res[4] -= mass * dif[0] * dif[2]
            res[5] -= mass * dif[1] * dif[2]
            res[6] = mass * dif[0]
            res[7] = mass * dif[1]
            res[8] = mass * dif[2]
            res[9] = mass
            cinert_out[worldid, b] = res
        else:
            jntid = item - nbody
            jbody = jnt_bodyid[jntid]
            dofid = jnt_dofadr[jntid]
            jtype = jnt_type[jntid]
            xaxis = xaxis_out[worldid, jntid]
            xmat = wp.transpose(xmat_out[worldid, jbody])
            offset = subtree_com_out[worldid, body_rootid[jbody]] - xanchor_out[worldid, jntid]
            if jtype == JointType.FREE:
                cdof_out[worldid, dofid + 0] = wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                cdof_out[worldid, dofid + 1] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
                cdof_out[worldid, dofid + 2] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
                cdof_out[worldid, dofid + 3] = wp.spatial_vector(xmat[0], wp.cross(xmat[0], offset))
                cdof_out[worldid, dofid + 4] = wp.spatial_vector(xmat[1], wp.cross(xmat[1], offset))
                cdof_out[worldid, dofid + 5] = wp.spatial_vector(xmat[2], wp.cross(xmat[2], offset))
            elif jtype == JointType.BALL:
                cdof_out[worldid, dofid + 0] = wp.spatial_vector(xmat[0], wp.cross(xmat[0], offset))
                cdof_out[worldid, dofid + 1] = wp.spatial_vector(xmat[1], wp.cross(xmat[1], offset))
                cdof_out[worldid, dofid + 2] = wp.spatial_vector(xmat[2], wp.cross(xmat[2], offset))
            elif jtype == JointType.SLIDE:
                cdof_out[worldid, dofid] = wp.spatial_vector(wp.vec3(0.0), xaxis)
            elif jtype == JointType.HINGE:
                cdof_out[worldid, dofid] = wp.spatial_vector(xaxis, wp.cross(xaxis, offset))
    _barrier()

    # Composite inertias; bodies attached to the world do not add to it.
    if active:
        end = body_subtree_end[bodyid]
        if bodyid == 0:
            end = 1
        crb_out[worldid, bodyid] = _sum_vec10(cinert_out, worldid, bodyid, end)


@wp.kernel
def _comvel_tree(
    nblock: int,
    nlevel: int,
    nbody: int,
    body_depth: wp.array[int],
    # Model:
    body_parentid: wp.array[int],
    body_jntnum: wp.array[int],
    body_jntadr: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    jnt_type: wp.array[int],
    # Data in:
    qvel_in: wp.array2d[float],
    cdof_in: wp.array2d[wp.spatial_vector],
    # Data out:
    cvel_out: wp.array2d[wp.spatial_vector],
    cdof_dot_out: wp.array2d[wp.spatial_vector],
):
    tid = wp.tid()
    worldid = tid // nblock
    bodyid = tid % nblock
    active = bodyid < nbody

    depth = int(0)
    pid = int(0)
    dofadr = int(0)
    delta = wp.spatial_vector()
    if active:
        depth = body_depth[bodyid]
        pid = body_parentid[bodyid]
        dofadr = body_dofadr[bodyid]
        for dofid in range(dofadr, dofadr + body_dofnum[bodyid]):
            delta += cdof_in[worldid, dofid] * qvel_in[worldid, dofid]
        if bodyid == 0:
            cvel_out[worldid, 0] = wp.spatial_vector()
    _barrier()

    cvel = wp.spatial_vector()
    for level in range(1, nlevel + 1):
        if active and depth == level:
            cvel = cvel_out[worldid, pid]
            cvel_out[worldid, bodyid] = cvel + delta
        _barrier()

    # cdof_dot from the velocity before each joint, as MuJoCo Warp's branch walk.
    if active and bodyid > 0:
        dofid = dofadr
        jntadr = body_jntadr[bodyid]
        for j in range(jntadr, jntadr + body_jntnum[bodyid]):
            jtype = jnt_type[j]
            if jtype == JointType.FREE:
                cvel += cdof_in[worldid, dofid + 0] * qvel_in[worldid, dofid + 0]
                cvel += cdof_in[worldid, dofid + 1] * qvel_in[worldid, dofid + 1]
                cvel += cdof_in[worldid, dofid + 2] * qvel_in[worldid, dofid + 2]
                cdof_dot_out[worldid, dofid + 0] = wp.spatial_vector()
                cdof_dot_out[worldid, dofid + 1] = wp.spatial_vector()
                cdof_dot_out[worldid, dofid + 2] = wp.spatial_vector()
                cdof_dot_out[worldid, dofid + 3] = math.motion_cross(cvel, cdof_in[worldid, dofid + 3])
                cdof_dot_out[worldid, dofid + 4] = math.motion_cross(cvel, cdof_in[worldid, dofid + 4])
                cdof_dot_out[worldid, dofid + 5] = math.motion_cross(cvel, cdof_in[worldid, dofid + 5])
                cvel += cdof_in[worldid, dofid + 3] * qvel_in[worldid, dofid + 3]
                cvel += cdof_in[worldid, dofid + 4] * qvel_in[worldid, dofid + 4]
                cvel += cdof_in[worldid, dofid + 5] * qvel_in[worldid, dofid + 5]
                dofid += 6
            elif jtype == JointType.BALL:
                cdof_dot_out[worldid, dofid + 0] = math.motion_cross(cvel, cdof_in[worldid, dofid + 0])
                cdof_dot_out[worldid, dofid + 1] = math.motion_cross(cvel, cdof_in[worldid, dofid + 1])
                cdof_dot_out[worldid, dofid + 2] = math.motion_cross(cvel, cdof_in[worldid, dofid + 2])
                cvel += cdof_in[worldid, dofid + 0] * qvel_in[worldid, dofid + 0]
                cvel += cdof_in[worldid, dofid + 1] * qvel_in[worldid, dofid + 1]
                cvel += cdof_in[worldid, dofid + 2] * qvel_in[worldid, dofid + 2]
                dofid += 3
            else:
                cdof_dot_out[worldid, dofid] = math.motion_cross(cvel, cdof_in[worldid, dofid])
                cvel += cdof_in[worldid, dofid] * qvel_in[worldid, dofid]
                dofid += 1


@wp.kernel
def _rne_tree(
    nblock: int,
    nlevel: int,
    nbody: int,
    nv: int,
    world_acc: int,
    full: bool,
    body_depth: wp.array[int],
    body_subtree_end: wp.array[int],
    gravity: wp.array[wp.vec3],
    # Model:
    body_parentid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    dof_bodyid: wp.array[int],
    # Data in:
    qvel_in: wp.array2d[float],
    qacc_in: wp.array2d[float],
    cdof_in: wp.array2d[wp.spatial_vector],
    cdof_dot_in: wp.array2d[wp.spatial_vector],
    cinert_in: wp.array2d[vec10],
    cvel_in: wp.array2d[wp.spatial_vector],
    # In:
    flg_acc: bool,
    cfrc_body: wp.array2d[wp.spatial_vector],
    # Data out:
    cacc_out: wp.array2d[wp.spatial_vector],
    cfrc_int_out: wp.array2d[wp.spatial_vector],
    qfrc_bias_out: wp.array2d[float],
):
    """``cacc`` forward pass; with ``full``, also ``cfrc_int`` and ``qfrc_bias`` as in ``smooth.rne``.

    ``world_acc``: 0 keeps ``cacc[:, 0]``, 1 sets it to zero, 2 sets it to minus gravity.
    """
    tid = wp.tid()
    worldid = tid // nblock
    bodyid = tid % nblock
    active = bodyid < nbody

    depth = int(0)
    pid = int(0)
    delta = wp.spatial_vector()
    if active:
        depth = body_depth[bodyid]
        pid = body_parentid[bodyid]
        dofadr = body_dofadr[bodyid]
        for dofid in range(dofadr, dofadr + body_dofnum[bodyid]):
            delta += cdof_dot_in[worldid, dofid] * qvel_in[worldid, dofid]
            if flg_acc:
                delta += cdof_in[worldid, dofid] * qacc_in[worldid, dofid]
        if bodyid == 0:
            if world_acc == 1:
                cacc_out[worldid, 0] = wp.spatial_vector()
            elif world_acc == 2:
                cacc_out[worldid, 0] = wp.spatial_vector(wp.vec3(0.0), -gravity[worldid % gravity.shape[0]])
    _barrier()

    for level in range(1, nlevel + 1):
        if active and depth == level:
            cacc_out[worldid, bodyid] = cacc_out[worldid, pid] + delta
        _barrier()

    if full:
        if active:
            frc = wp.spatial_vector()
            if bodyid > 0:
                cacc = cacc_out[worldid, bodyid]
                cinert = cinert_in[worldid, bodyid]
                cvel = cvel_in[worldid, bodyid]
                frc = math.inert_vec(cinert, cacc)
                frc += math.motion_cross_force(cvel, math.inert_vec(cinert, cvel))
            cfrc_body[worldid, bodyid] = frc
        _barrier()
        if active:
            cfrc_int_out[worldid, bodyid] = _sum_spatial(cfrc_body, worldid, bodyid, body_subtree_end[bodyid])
        _barrier()
        for dofid in range(bodyid, nv, nblock):
            qfrc_bias_out[worldid, dofid] = wp.dot(cdof_in[worldid, dofid], cfrc_int_out[worldid, dof_bodyid[dofid]])
