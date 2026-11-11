"""Private, whole-batch MuJoCo Warp execution for the DMC adapters.

This module has no task, episode or RNG policy. Its caller supplies native
models and physical checkpoints, and only passes lanes that should integrate.
"""

from contextlib import contextmanager
from dataclasses import fields
from functools import cache

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from ...._core.warp_cuda import require_warp_cuda
from .mjwarp_reset import WarpModelBatch

_PHYSICAL = {"qpos": "qpos", "qvel": "qvel", "time": "time", "warmstart": "qacc_warmstart"}
_RESULT_FIELDS = ("qpos", "qvel", "act", "time", "qacc_warmstart", "sensordata")
mjw = wp = None
_DIMENSIONS = ("nq", "nv", "nu", "na", "nbody", "njnt", "ngeom", "nsite", "ntendon", "neq", "nsensor", "nmocap")
_TOPOLOGY = (
    "body_parentid",
    "body_weldid",
    "body_rootid",
    "body_mocapid",
    "jnt_type",
    "jnt_qposadr",
    "jnt_dofadr",
    "geom_type",
    "geom_bodyid",
    "geom_contype",
    "geom_conaffinity",
    "pair_geom1",
    "pair_geom2",
    "site_bodyid",
    "actuator_trntype",
    "actuator_dyntype",
    "actuator_trnid",
    "eq_type",
    "eq_obj1id",
    "eq_obj2id",
    "sensor_type",
    "sensor_adr",
    "sensor_dim",
    "sensor_needstage",
    "wrap_type",
    "wrap_objid",
    "tendon_adr",
    "tendon_num",
)


def _batch_fields(cls):
    return tuple(f.name for f in fields(cls) if getattr(f.type, "shape", (None,))[0] == "*")


def _load_backend():
    global mjw, wp
    if mjw is None:
        try:
            import mujoco_warp
            import warp
        except ImportError as error:
            raise ImportError(
                "DMC MJWarp physics requires mujoco-warp and warp-lang; install stable-envs[dmcontrol-gpu]"
            ) from error
        if any(version.split(".")[:2] != ["3", "17"] for version in (mujoco.__version__, mujoco_warp.__version__)):
            raise RuntimeError(
                "DMC MJWarp physics requires MuJoCo and mujoco-warp 3.17; install stable-envs[dmcontrol-gpu]"
            )
        mjw, wp = mujoco_warp, warp


@cache
def _result_check_kernel():
    @wp.kernel
    def check(
        overflow: wp.array[int],
        nacon: wp.array[int],
        capacity_mask: int,
        naconmax: int,
        qpos: wp.array2d[float],
        qvel: wp.array2d[float],
        act: wp.array2d[float],
        time: wp.array[float],
        warmstart: wp.array2d[float],
        sensors: wp.array2d[float],
        error: wp.array[int],
    ):
        world = wp.tid()
        flags = int(0)  # noqa: UP018 - mutable Warp loop accumulator
        if (overflow[world] & capacity_mask) != 0 or nacon[0] > naconmax:
            flags = flags | 1
        for i in range(qpos.shape[1]):
            if not wp.isfinite(qpos[world, i]):
                flags = flags | 2
        for i in range(qvel.shape[1]):
            if not wp.isfinite(qvel[world, i]):
                flags = flags | 4
        for i in range(act.shape[1]):
            if not wp.isfinite(act[world, i]):
                flags = flags | 8
        if not wp.isfinite(time[world]):
            flags = flags | 16
        for i in range(warmstart.shape[1]):
            if not wp.isfinite(warmstart[world, i]):
                flags = flags | 32
        for i in range(sensors.shape[1]):
            if not wp.isfinite(sensors[world, i]):
                flags = flags | 64
        if flags != 0:
            wp.atomic_or(error, 0, flags)

    return check


@cache
def _plane_capsule_frame_kernel():
    # MuJoCo 3.17 passes the full capsule axis to mju_makeFrame, which tests
    # the tangent length before projection. Warp 3.17 tests the projected
    # length, changing the orientation of pyramidal friction constraints.
    # See MuJoCo 3.17 engine_collision_primitive.c and engine_util_spatial.c.
    # Construct this kernel only after the optional backend is loaded.
    @wp.kernel
    def repair(
        geom_type: wp.array[int],
        geom_xmat: wp.array2d[wp.mat33],
        nacon: wp.array[int],
        worldid: wp.array[int],
        geom: wp.array[wp.vec2i],
        frame: wp.array[wp.mat33],
    ):
        contact = wp.tid()
        if contact >= nacon[0]:
            return
        pair = geom[contact]
        if pair[0] < 0 or pair[1] < 0:
            return
        capsule = pair[0]
        if geom_type[pair[0]] == 0 and geom_type[pair[1]] == 3:
            capsule = pair[1]
        elif geom_type[pair[1]] != 0 or geom_type[pair[0]] != 3:
            return
        old = frame[contact]
        normal = wp.normalize(wp.vec3(old[0, 0], old[0, 1], old[0, 2]))
        mat = geom_xmat[worldid[contact], capsule]
        tangent = wp.vec3(mat[0, 2], mat[1, 2], mat[2, 2])
        if wp.dot(tangent, tangent) < 0.25:
            if -0.5 < normal[1] and normal[1] < 0.5:
                tangent = wp.vec3(0.0, 1.0, 0.0)
            else:
                tangent = wp.vec3(0.0, 0.0, 1.0)
        tangent = tangent - normal * wp.dot(normal, tangent)
        length = wp.length(tangent)
        # mju_normalize3's exact degenerate-vector rule; do not widen it to
        # suppress genuine near-parallel orientation differences in float32.
        if length < 1.0e-15:
            tangent = wp.vec3(1.0, 0.0, 0.0)
        else:
            tangent = tangent / length
        third = wp.cross(normal, tangent)
        frame[contact] = wp.mat33(
            old[0, 0],
            old[0, 1],
            old[0, 2],
            tangent[0],
            tangent[1],
            tangent[2],
            third[0],
            third[1],
            third[2],
        )

    return repair


@cache
def _capsule_slot_kernels():
    # A rotated near-parallel pair can have one upstream contact but two native
    # 3.17 contacts. Reserve the missing slot before correcting its geometry.
    Contact = wp.struct(mjw.Contact)

    @wp.kernel
    def index_contacts(nacon: wp.array[int], contact: Contact, pair_ids: wp.array2d[int], slots: wp.array3d[int]):
        i = wp.tid()
        if i >= nacon[0]:
            return
        geoms = contact.geom[i]
        if geoms[0] < 0 or geoms[1] < 0:
            return
        pair = pair_ids[geoms[0], geoms[1]]
        if pair >= 0:
            slots[contact.worldid[i], pair, contact.geomcollisionid[i]] = i

    @wp.kernel
    def expand_contacts(
        geom_xmat: wp.array2d[wp.mat33], slots: wp.array3d[int], contact: Contact, nacon: wp.array[int]
    ):
        world, pair = wp.tid()
        first = slots[world, pair, 0]
        second = slots[world, pair, 1]
        if (first < 0 and second < 0) or (first >= 0 and second >= 0):
            return
        source = wp.max(first, second)
        geoms = contact.geom[source]
        r1 = geom_xmat[world, geoms[0]]
        r2 = geom_xmat[world, geoms[1]]
        a = wp.vec3d(wp.float64(r1[0, 2]), wp.float64(r1[1, 2]), wp.float64(r1[2, 2]))
        b = wp.vec3d(wp.float64(r2[0, 2]), wp.float64(r2[1, 2]), wp.float64(r2[2, 2]))
        cross = wp.cross(a, b)
        if wp.dot(cross, cross) > wp.float64(1.0e-8) * wp.dot(a, a) * wp.dot(b, b):
            return
        dest = wp.atomic_add(nacon, 0, 1)
        if dest >= contact.type.shape[0]:
            return  # _check_result rejects the exceeded contact capacity.
        contact.dist[dest] = contact.dist[source]
        contact.pos[dest] = contact.pos[source]
        contact.frame[dest] = contact.frame[source]
        contact.includemargin[dest] = contact.includemargin[source]
        contact.friction[dest] = contact.friction[source]
        contact.solref[dest] = contact.solref[source]
        contact.solreffriction[dest] = contact.solreffriction[source]
        contact.solimp[dest] = contact.solimp[source]
        contact.dim[dest] = contact.dim[source]
        contact.geom[dest] = contact.geom[source]
        # Warp omits the flex/mesh arrays for primitive-only models.
        if contact.flex.shape[0]:
            contact.flex[dest] = contact.flex[source]
        if contact.elem.shape[0]:
            contact.elem[dest] = contact.elem[source]
        if contact.vert.shape[0]:
            contact.vert[dest] = contact.vert[source]
        contact.worldid[dest] = world
        contact.type[dest] = contact.type[source]
        contact.geomcollisionid[dest] = wp.where(first < 0, 0, 1)
        contact.adhesion[dest] = contact.adhesion[source]
        for j in range(contact.efc_address.shape[1]):
            contact.efc_address[dest, j] = -1

    return Contact, index_contacts, expand_contacts


@cache
def _capsule_contact_kernel():
    # MuJoCo 3.17 uses mjc_segWeights and a relative angular cutoff. Warp
    # still uses the older determinant/endpoint algorithm. Reconstruct the
    # native closest points in float64; integration remains float32.
    @wp.kernel
    def repair_capsule_contacts(
        geom_type: wp.array[int],
        geom_size: wp.array2d[wp.vec3],
        geom_xpos: wp.array2d[wp.vec3],
        geom_xmat: wp.array2d[wp.mat33],
        nacon: wp.array[int],
        worldid: wp.array[int],
        geom: wp.array[wp.vec2i],
        distance: wp.array[float],
        position: wp.array[wp.vec3],
        frame: wp.array[wp.mat33],
        collision_id: wp.array[int],
        contact_type: wp.array[int],
        includemargin: wp.array[float],
        pair_ids: wp.array2d[int],
        gaps: wp.array2d[float],
    ):
        index = wp.tid()
        if index >= nacon[0]:
            return
        pair = geom[index]
        if pair[0] < 0 or pair[1] < 0:
            return
        if geom_type[pair[0]] != 3 or geom_type[pair[1]] != 3:
            return
        world = worldid[index]
        p1f = geom_xpos[world, pair[0]]
        p2f = geom_xpos[world, pair[1]]
        p1 = wp.vec3d(wp.float64(p1f[0]), wp.float64(p1f[1]), wp.float64(p1f[2]))
        p2 = wp.vec3d(wp.float64(p2f[0]), wp.float64(p2f[1]), wp.float64(p2f[2]))
        r1 = geom_xmat[world, pair[0]]
        r2 = geom_xmat[world, pair[1]]
        z1 = wp.vec3d(wp.float64(r1[0, 2]), wp.float64(r1[1, 2]), wp.float64(r1[2, 2]))
        z2 = wp.vec3d(wp.float64(r2[0, 2]), wp.float64(r2[1, 2]), wp.float64(r2[2, 2]))
        size1 = geom_size[world, pair[0]]
        size2 = geom_size[world, pair[1]]
        axis1 = z1 * wp.float64(size1[1])
        axis2 = z2 * wp.float64(size2[1])
        a = p1 - axis1
        c = p2 - axis2
        ab = wp.float64(2.0) * axis1
        cd = wp.float64(2.0) * axis2
        ca = a - c
        dot1 = wp.dot(ab, ab)
        dot2 = wp.dot(cd, cd)
        dot3 = wp.dot(cd, ca)
        dot4 = wp.dot(ab, ca)
        dot5 = wp.dot(ab, cd)
        s = wp.float64(0.0)
        t = wp.float64(0.0)
        count = int(1)  # noqa: UP018 — mutable Warp local
        if dot1 <= 1.0e-15 and dot2 <= 1.0e-15:
            pass
        elif dot1 <= 1.0e-15:
            t = wp.clamp(dot3 / dot2, wp.float64(0.0), wp.float64(1.0))
        elif dot2 <= 1.0e-15:
            s = wp.clamp(-dot4 / dot1, wp.float64(0.0), wp.float64(1.0))
        else:
            cross = wp.cross(ab, cd)
            determinant = wp.dot(cross, cross)
            if determinant > wp.float64(1.0e-8) * dot1 * dot2:
                s = wp.clamp(wp.dot(cross, wp.cross(cd, ca)) / determinant, wp.float64(0.0), wp.float64(1.0))
            else:
                count = 2
                s = wp.float64(collision_id[index])
            t = (dot5 * s + dot3) / dot2
            if t < 0.0:
                s = wp.clamp(-dot4 / dot1, wp.float64(0.0), wp.float64(1.0))
                t = wp.float64(0.0)
            elif t > 1.0:
                s = wp.clamp((dot5 - dot4) / dot1, wp.float64(0.0), wp.float64(1.0))
                t = wp.float64(1.0)
        if collision_id[index] >= count:
            contact_type[index] = 0
            return
        closest1 = a + ab * s
        closest2 = c + cd * t
        separation = wp.length(closest2 - closest1)
        normal = closest2 - closest1
        # Native mjraw_SphereSphere uses the axis cross product at an
        # intersection, where nearest-point cancellation cannot give a normal.
        if separation < 1.0e-15:
            normal = wp.cross(z1, z2)
        length = wp.length(normal)
        if length < 1.0e-15:
            normal = wp.vec3d(1.0, 0.0, 0.0)
        else:
            normal = normal / length
        tangent = wp.vec3d(0.0, 0.0, 1.0)
        if -0.5 < normal[1] and normal[1] < 0.5:
            tangent = wp.vec3d(0.0, 1.0, 0.0)
        tangent = wp.normalize(tangent - normal * wp.dot(tangent, normal))
        third = wp.cross(normal, tangent)
        frame[index] = wp.mat33(
            wp.float32(normal[0]),
            wp.float32(normal[1]),
            wp.float32(normal[2]),
            wp.float32(tangent[0]),
            wp.float32(tangent[1]),
            wp.float32(tangent[2]),
            wp.float32(third[0]),
            wp.float32(third[1]),
            wp.float32(third[2]),
        )
        dist = separation - wp.float64(size1[0]) - wp.float64(size2[0])
        # A near-parallel pair can touch at only one endpoint. Correcting or
        # adding the other endpoint must not create a contact outside the
        # detection band. Preserve collision-sensor flags, if present.
        gap = gaps[world, pair_ids[pair[0], pair[1]]]
        if dist >= wp.float64(includemargin[index] + gap):
            contact_type[index] = contact_type[index] & ~1  # ContactType.CONSTRAINT
        center = closest1 + normal * (wp.float64(size1[0]) + wp.float64(0.5) * dist)
        distance[index] = wp.float32(dist)
        position[index] = wp.vec3(wp.float32(center[0]), wp.float32(center[1]), wp.float32(center[2]))

    return repair_capsule_contacts


@cache
def _constraint_order_kernels():
    @wp.struct
    class Rows:
        type: wp.array2d[int]
        id: wp.array2d[int]
        J: wp.array3d[float]
        pos: wp.array2d[float]
        margin: wp.array2d[float]
        D: wp.array2d[float]
        vel: wp.array2d[float]
        aref: wp.array2d[float]
        frictionloss: wp.array2d[float]
        force: wp.array2d[float]
        state: wp.array2d[int]
        island: wp.array2d[int]
        Jqvel: wp.array2d[float]

    @wp.kernel
    def order(
        ngeom: int,
        nobject: int,
        ncontact: int,
        nefc: wp.array[int],
        rows: Rows,
        geom: wp.array[wp.vec2i],
        point: wp.array[int],
        address: wp.array2d[int],
        keys: wp.array2d[wp.int64],
        permutation: wp.array2d[int],
        inverse: wp.array2d[int],
    ):
        world = wp.tid()
        capacity = permutation.shape[1]
        count = wp.min(nefc[world], capacity)
        # Cheetah and Humanoid have few active rows. Insertion sort visits only those rows,
        # with independent worlds remaining parallel and no global atomics.
        for row in range(count):
            kind = rows.type[world, row]
            obj = rows.id[world, row]
            contact_point = int(0)  # noqa: UP018 - Warp needs a dynamic kernel variable.
            component = row
            if kind >= 5:  # MuJoCo's three contact constraint types.
                contact_point = point[obj]
                component = row - address[obj, 0]
                pair = geom[obj]
                obj = pair[0] * ngeom + pair[1]
            key = (
                (wp.int64(kind) * wp.int64(nobject) + wp.int64(obj)) * wp.int64(ncontact) + wp.int64(contact_point)
            ) * wp.int64(capacity) + wp.int64(component)
            cursor = row
            while cursor > 0:
                if keys[world, cursor - 1] <= key:
                    break
                keys[world, cursor] = keys[world, cursor - 1]
                permutation[world, cursor] = permutation[world, cursor - 1]
                cursor -= 1
            keys[world, cursor] = key
            permutation[world, cursor] = row
        for row in range(count):
            inverse[world, permutation[world, row]] = row

    @wp.kernel
    def copy_rows(nefc: wp.array[int], permutation: wp.array2d[int], gather: bool, source: Rows, target: Rows):
        world, row = wp.tid()
        if row >= nefc[world]:
            return
        old = row
        if gather:
            old = permutation[world, row]
        target.type[world, row] = source.type[world, old]
        target.id[world, row] = source.id[world, old]
        target.pos[world, row] = source.pos[world, old]
        target.margin[world, row] = source.margin[world, old]
        target.D[world, row] = source.D[world, old]
        target.vel[world, row] = source.vel[world, old]
        target.aref[world, row] = source.aref[world, old]
        target.frictionloss[world, row] = source.frictionloss[world, old]
        target.force[world, row] = source.force[world, old]
        target.state[world, row] = source.state[world, old]
        target.island[world, row] = source.island[world, old]
        target.Jqvel[world, row] = source.Jqvel[world, old]
        for dof in range(source.J.shape[2]):
            target.J[world, row, dof] = source.J[world, old, dof]

    @wp.kernel
    def remap(nacon: wp.array[int], world: wp.array[int], inverse: wp.array2d[int], address: wp.array2d[int]):
        contact, component = wp.tid()
        if contact >= nacon[0]:
            return
        old = address[contact, component]
        if old >= 0 and old < inverse.shape[1]:
            address[contact, component] = inverse[world[contact], old]

    return Rows, order, copy_rows, remap


@cache
def _elliptic_hessian_kernel():
    from mujoco_warp._src import math, solver, types

    # MuJoCo Warp 3.17 adds elliptic contact curvature to H with floating-point
    # atomics. Give each (world, matrix entry) one writer in constraint-row order.
    # Reuse the upstream curvature formula and retain parallelism across worlds.
    @wp.kernel
    def accumulate(
        impratio: wp.array[float],
        tri_row: wp.array[int],
        tri_col: wp.array[int],
        nefc: wp.array[int],
        efc_type: wp.array2d[int],
        efc_id: wp.array2d[int],
        friction: wp.array[types.vec5],
        dimension: wp.array[int],
        address: wp.array2d[int],
        jacobian: wp.array3d[float],
        diagonal: wp.array2d[float],
        state: wp.array2d[int],
        jaref: wp.array2d[float],
        done: wp.array[bool],
        hessian: wp.array3d[float],
    ):
        world, element = wp.tid()
        if done[world]:
            return
        dof1, dof2 = tri_row[element], tri_col[element]
        value = hessian[world, dof1, dof2]
        for row in range(wp.min(nefc[world], state.shape[1])):
            if efc_type[world, row] != types.ConstraintType.CONTACT_ELLIPTIC:
                continue
            contact = efc_id[world, row]
            if address[contact, 0] != row or state[world, row] != types.ConstraintState.CONE:
                continue
            fri = friction[contact]
            mu = fri[0] * impratio[world % impratio.shape[0]]
            mu2 = mu * mu
            dm = math.safe_div(diagonal[world, row], mu2 * (1.0 + mu2))
            if dm == 0.0:
                continue
            n = jaref[world, row] * mu
            z01, z02 = mu * jacobian[world, row, dof1], mu * jacobian[world, row, dof2]
            tt = float(0.0)  # noqa: UP018 - Warp needs dynamic loop accumulators.
            projection1 = float(0.0)  # noqa: UP018
            projection2 = float(0.0)  # noqa: UP018
            tangent_dot = float(0.0)  # noqa: UP018
            for component in range(1, dimension[contact]):
                efcid = address[contact, component]
                if efcid >= 0:
                    scale = fri[component - 1]
                    u = jaref[world, efcid] * scale
                    z1, z2 = scale * jacobian[world, efcid, dof1], scale * jacobian[world, efcid, dof2]
                    tt += u * u
                    projection1 += u * z1
                    projection2 += u * z2
                    tangent_dot += z1 * z2
            t = wp.max(wp.sqrt(tt), types.MJ_MINVAL)
            ttt = wp.max(t * t * t, types.MJ_MINVAL)
            mu_tinv = math.safe_div(mu, t)
            value += solver._elliptic_hessian_entry_from_projections(
                dm,
                mu_tinv,
                mu * math.safe_div(n, ttt),
                mu2 - n * mu_tinv,
                z01,
                z02,
                projection1,
                projection2,
                tangent_dot,
            )
        hessian[world, dof1, dof2] = value

    return accumulate


class _DenseEllipticSolver:
    """The 3.17 dense Newton loop with ordered elliptic Hessian accumulation.

    Solver kernels, line search, convergence, warmstart and tolerances remain
    upstream's. Context and reduction buffers belong only to this runtime.
    """

    def __init__(self, model, data):
        from mujoco_warp._src import solver, types

        self._solver, self._types = solver, types
        self.m, self.d = model, data
        if (
            model.is_sparse
            or model.opt.solver != types.SolverType.NEWTON
            or model.opt.cone != types.ConeType.ELLIPTIC
            or model.opt.enableflags & types.EnableBit.SLEEP
            or model.nflex
            or not model.nv
            or not data.njmax
        ):
            raise ValueError("Ordered elliptic solving requires dense, awake rigid Newton worlds")
        self.ctx = solver._create_solver_context(model, data)
        self.nsolving = wp.empty(1, dtype=int)

    def gradient(self):
        m, d, ctx = self.m, self.d, self.ctx
        solver, types = self._solver, self._types
        wp.launch(
            solver._update_gradient_zero_grad_dot(False),
            dim=d.nworld,
            inputs=[d.nefc, ctx.alpha, ctx.done],
            outputs=[ctx.grad_dot, ctx.newton_decrement, ctx.grad_scale, ctx.search_unchanged],
        )
        wp.launch(
            solver._update_gradient_grad(False),
            dim=(d.nworld, m.nv),
            inputs=[d.qfrc_smooth, d.qfrc_constraint, d.efc.Ma, d.nefc, ctx.done],
            outputs=[ctx.grad, ctx.grad_dot],
        )
        wp.launch_tiled(
            solver._update_gradient_JTDAJ_dense_tiled(
                m.nv_pad, types.TILE_SIZE_JTDAJ_DENSE, d.njmax, m.M_colind.shape[0]
            ),
            dim=d.nworld,
            inputs=[m.M_colind, m.M_hinit_i, d.nefc, d.M, d.efc.J, d.efc.D, d.efc.state, ctx.done],
            outputs=[ctx.h],
            block_dim=m.block_dim.update_gradient_JTDAJ_dense,
        )
        wp.launch(
            _elliptic_hessian_kernel(),
            dim=(d.nworld, m.dof_tri_row.size),
            inputs=[
                m.opt.impratio_invsqrt,
                m.dof_tri_row,
                m.dof_tri_col,
                d.nefc,
                d.efc.type,
                d.efc.id,
                d.contact.friction,
                d.contact.dim,
                d.contact.efc_address,
                d.efc.J,
                d.efc.D,
                d.efc.state,
                ctx.Jaref,
                ctx.done,
            ],
            outputs=[ctx.h],
        )
        solver._cholesky_factorize_solve(m, d, ctx)

    def iteration(self):
        m, d, ctx = self.m, self.d, self.ctx
        solver = self._solver
        solver._linesearch(m, d, ctx)
        solver._update_constraint(m, d, ctx)
        self.gradient()
        wp.launch(
            solver._solve_done(int(m.opt.warn_overflow)),
            dim=d.nworld,
            inputs=[
                m.opt.tolerance,
                m.opt.iterations,
                ctx.inertia_scale,
                ctx.grad_dot,
                ctx.newton_decrement,
                ctx.improvement,
                ctx.alpha,
                ctx.done,
            ],
            outputs=[d.solver_niter, d.overflow, self.nsolving, ctx.done],
        )

    def solve(self):
        m, d, ctx = self.m, self.d, self.ctx
        solver, types = self._solver, self._types
        wp.launch(
            solver._solve_init_dof(not (m.opt.disableflags & types.DisableBit.WARMSTART), False),
            dim=(d.nworld, m.nv),
            inputs=[d.nefc, d.qacc_warmstart, d.qacc_smooth],
            outputs=[d.qacc, d.qfrc_constraint],
        )
        solver.init_context(m, d, ctx, grad=False)
        self.gradient()
        self.nsolving.fill_(d.nworld)
        if m.opt.iterations and m.opt.graph_conditional:
            wp.capture_while(self.nsolving, while_body=self.iteration)
        else:
            for _ in range(m.opt.iterations):
                self.iteration()


@cache
def _pyramidal_solver_kernels():
    @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
    def sort_changed_ids(changed_count: wp.array[int], done: wp.array[bool], changed_ids: wp.array2d[int]):
        world = wp.tid()
        if done[world]:
            return
        for index in range(1, changed_count[world]):
            value = changed_ids[world, index]
            previous = index - 1
            while previous >= 0:
                if changed_ids[world, previous] <= value:
                    break
                changed_ids[world, previous + 1] = changed_ids[world, previous]
                previous -= 1
            changed_ids[world, previous + 1] = value

    @wp.kernel(module="unique", module_options={"enable_backward": False, "fuse_fp": False}, grid_stride=False)
    def gradient_norm(
        nv: int,
        stable_fast: bool,
        state_changed_count: wp.array[int],
        done: wp.array[bool],
        grad: wp.array2d[float],
        grad_dot: wp.array[float],
    ):
        world = wp.tid()
        if done[world]:
            return
        if stable_fast:
            if state_changed_count[world] == 0:
                return
        # Preserve the upstream product rounding before each addition. Padded
        # DOFs are excluded; unchanged worlds retain their scaled stale norm.
        total = float(0.0)  # noqa: UP018 - Warp needs a dynamic loop accumulator.
        for dof in range(nv):
            value = grad[world, dof]
            total += value * value
        grad_dot[world] = total

    return sort_changed_ids, gradient_norm


class _DensePyramidalSolver:
    """The 3.17 incremental Newton loop with ordered row deltas and grad norms.

    Keep upstream gradient/Hessian construction, factorization, line search,
    convergence and stable-state fast paths. The only added work orders changed
    row IDs and replaces the atomic norm before convergence reads it.
    """

    def __init__(self, model, data):
        from mujoco_warp._src import solver, types

        if (
            model.is_sparse
            or model.nflex
            or not 0 < model.nv <= 50
            or not data.njmax
            or model.opt.solver != types.SolverType.NEWTON
            or model.opt.cone != types.ConeType.PYRAMIDAL
            or model.opt.integrator != types.IntegratorType.EULER
            or model.opt.enableflags & types.EnableBit.SLEEP
            or model.opt.noslip_iterations
            or model.opt.deterministic != types.DeterminismType.NONE
        ):
            raise ValueError(
                "Ordered pyramidal solving requires dense, awake rigid Euler/Newton worlds with 1-50 DOFs, "
                "pyramidal contacts, no noslip and no model-wide determinism"
            )
        self._solver, self._types = solver, types
        self.m, self.d = model, data
        self.ctx = solver._create_solver_context(model, data)
        self.nsolving = wp.empty(1, dtype=int)

    def gradient(self, incremental=False):
        m, d, ctx = self.m, self.d, self.ctx
        sort_changed_ids, gradient_norm = _pyramidal_solver_kernels()
        if incremental:
            wp.launch(
                sort_changed_ids,
                dim=d.nworld,
                inputs=[ctx.quad_changed_count, ctx.done],
                outputs=[ctx.quad_changed_ids],
            )
            self._solver._update_gradient_incremental(m, d, ctx, stable_fast=True)
        else:
            self._solver._update_gradient(m, d, ctx)
        # Cholesky reads grad and H, not grad_dot. Replace the atomic norm only
        # after the upstream update, before _solve_done can consume it.
        wp.launch(
            gradient_norm,
            dim=d.nworld,
            inputs=[m.nv, incremental, ctx.state_changed_count, ctx.done, ctx.grad],
            outputs=[ctx.grad_dot],
        )

    def iteration(self):
        m, d, ctx = self.m, self.d, self.ctx
        solver = self._solver
        solver._linesearch(m, d, ctx)
        wp.launch(
            solver._zero_change_counters, dim=d.nworld, outputs=[ctx.quad_changed_count, ctx.state_changed_count]
        )
        solver._update_constraint(m, d, ctx, track_changes=True, stable_fast=True)
        self.gradient(incremental=True)
        wp.launch(
            solver._solve_done(int(m.opt.warn_overflow)),
            dim=d.nworld,
            inputs=[
                m.opt.tolerance,
                m.opt.iterations,
                ctx.inertia_scale,
                ctx.grad_dot,
                ctx.newton_decrement,
                ctx.improvement,
                ctx.alpha,
                ctx.done,
            ],
            outputs=[d.solver_niter, d.overflow, self.nsolving, ctx.done],
        )

    def solve(self):
        m, d, ctx = self.m, self.d, self.ctx
        solver, types = self._solver, self._types
        wp.launch(
            solver._solve_init_dof(not (m.opt.disableflags & types.DisableBit.WARMSTART), False),
            dim=(d.nworld, m.nv),
            inputs=[d.nefc, d.qacc_warmstart, d.qacc_smooth],
            outputs=[d.qacc, d.qfrc_constraint],
        )
        solver.init_context(m, d, ctx, grad=False)
        self.gradient()
        # The previous solve's reusable ray is invalid after the full rebuild.
        ctx.search_unchanged.zero_()
        self.nsolving.fill_(d.nworld)
        if m.opt.iterations and m.opt.graph_conditional:
            wp.capture_while(self.nsolving, while_body=self.iteration)
        else:
            for _ in range(m.opt.iterations):
                self.iteration()
        # Stable-state iterations leave qfrc_constraint stale or zeroed for
        # converged worlds. Recover it exactly as the upstream Newton loop does.
        wp.launch(
            solver._qfrc_constraint_from_grad,
            dim=(d.nworld, m.nv),
            inputs=[d.qfrc_smooth, d.efc.Ma, ctx.grad, ctx.grad_scale],
            outputs=[d.qfrc_constraint],
        )


class WarpSnapshot:
    """Borrowed B-first views, valid until the runtime is advanced or loaded."""

    def __init__(self, runtime):
        self._runtime = runtime

    def data(self, name):
        value = self._runtime._data_view(name)
        # Match MuJoCo's named-data matrix layout used by the DMC task code.
        if name in ("xmat", "ximat", "geom_xmat", "site_xmat", "cam_xmat"):
            return value.flatten(-2)
        return value

    def model(self, name):
        runtime = self._runtime
        if name in runtime._native_model_views:
            return runtime._native_model_views[name]
        value = runtime._m
        for part in name.split("."):
            value = getattr(value, part)
        if isinstance(value, wp.array):
            view = wp.to_torch(value)
            # Non-batched indexing metadata is also useful to task functions.
            if name not in runtime._model_fields and not name.startswith(("opt.", "stat.")):
                return view.unsqueeze(0).expand(runtime.num_envs, *view.shape)
            return view.expand(runtime.num_envs, *view.shape[1:]) if view.shape[0] == 1 else view
        return torch.as_tensor(value, device=runtime.device).expand(runtime.num_envs)


class WarpRuntime:
    """Persistent same-topology worlds with source-compatible DMC stepping.

    CPU is supported for focused kernel tests. Public DMC routing requires CUDA.
    All model arrays which can vary between worlds are allocated before capture;
    subsequent loads copy values into stable buffers rather than replacing them.
    """

    @staticmethod
    def model_schema():
        _load_backend()
        return _batch_fields(mjw.Model), _batch_fields(mjw.Option)

    def __init__(
        self,
        models,
        device,
        nsubsteps,
        *,
        ordered_constraints=False,
        ordered_elliptic=False,
        ordered_pyramidal=False,
    ):
        _load_backend()
        if ordered_pyramidal and (not ordered_constraints or ordered_elliptic):
            raise ValueError("Ordered pyramidal solving requires canonical constraints and no elliptic solver")
        self._capacity_overflow = sum(
            int(flag)
            for flag in mjw.OverflowType
            if flag not in (mjw.OverflowType.ITERATIONS, mjw.OverflowType.LS_ITERATIONS)
        )
        self.models = models if isinstance(models, WarpModelBatch) else list(models)
        self.num_envs = len(self.models)
        if not self.num_envs:
            raise ValueError("WarpRuntime needs at least one model")
        if isinstance(nsubsteps, bool) or int(nsubsteps) != nsubsteps or nsubsteps < 1:
            raise ValueError("nsubsteps must be a positive integer")
        self.nsubsteps = int(nsubsteps)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("WarpRuntime requires CPU or CUDA")
        if self.device.type == "cuda":
            require_warp_cuda("DMC GPU physics")
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            self.device = torch.device("cuda", index)
        wp.init()
        self._wp_device = str(self.device)
        self._closed = False
        # One owned, nondefault stream is retained for the runtime lifetime.
        # Repeated stream_from_torch wrappers can unregister the same handle
        # while an outer scope still uses it, and the default stream cannot be
        # captured. Queue both libraries on this one persistent stream instead.
        self._warp_stream = wp.Stream(self._wp_device) if self.device.type == "cuda" else None
        self._torch_stream = wp.stream_to_torch(self._warp_stream) if self._warp_stream is not None else None
        self._graph = None
        self._constraint_buffers = None
        self._elliptic_solver = None
        self._pyramidal_solver = None
        self._model_fields = _batch_fields(mjw.Model)
        self._option_fields = _batch_fields(mjw.Option)
        self._reference = self.models.reference if isinstance(self.models, WarpModelBatch) else self.models[0]
        self._dimensions = {name: getattr(self._reference, name) for name in _DIMENSIONS}
        self._topology = {name: getattr(self._reference, name).copy() for name in _TOPOLOGY}
        self._scalar_options = {
            name: getattr(self._reference.opt, name)
            for name in ("integrator", "cone", "solver", "disableflags", "enableflags", "iterations", "ls_iterations")
        }
        self._check_models(self.models)
        self._integrator = int(self._reference.opt.integrator)
        self._physical = dict(_PHYSICAL)
        if self._reference.na:
            self._physical["activation"] = "act"
        sensor_indices = [
            index
            for sensor, stage in enumerate(self._reference.sensor_needstage)
            if stage == mujoco.mjtStage.mjSTAGE_ACC
            for index in range(
                self._reference.sensor_adr[sensor],
                self._reference.sensor_adr[sensor] + self._reference.sensor_dim[sensor],
            )
        ]
        self._acc_indices = torch.tensor(sensor_indices, dtype=torch.long, device=self.device)
        with self._scope():
            self._m = mjw.put_model(self._reference, batch_sizes=dict.fromkeys(self._model_fields, self.num_envs))
            for name in self._option_fields:
                current = getattr(self._m.opt, name)
                setattr(self._m.opt, name, wp.zeros(self.num_envs, dtype=current.dtype, device=self._wp_device))
            # Use declared capacities when present; otherwise conservative DMC
            # bounds. Overflow is an error, never permission to drop contacts.
            nconmax = max(128, int(self._reference.nconmax))
            njmax = max(512, int(self._reference.njmax))
            self._d = mjw.make_data(self._reference, nworld=self.num_envs, nconmax=nconmax, njmax=njmax)
            self._result_error = wp.zeros(1, dtype=int, device=self._wp_device)
            self._result_error_view = wp.to_torch(self._result_error)
            self._acc_values = torch.empty(
                (self.num_envs, len(sensor_indices)), dtype=torch.float32, device=self.device
            )
            capsule_count = np.count_nonzero(self._reference.geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE)
            self._plane_capsules = bool(
                capsule_count and np.any(self._reference.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
            )
            self._capsule_pairs = capsule_count > 1
            self._capsule_pair_ids = self._capsule_slots = self._capsule_gaps = self._contact_arrays = None
            if self._capsule_pairs:
                capsules = np.flatnonzero(self._reference.geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE)
                pair_ids = np.full((self._reference.ngeom, self._reference.ngeom), -1, dtype=np.int32)
                pairs = [(a, b) for i, a in enumerate(capsules) for b in capsules[i + 1 :]]
                for i, (a, b) in enumerate(pairs):
                    pair_ids[a, b] = pair_ids[b, a] = i
                self._capsule_geoms = np.asarray(pairs)
                self._capsule_gap_overrides = [
                    (pair_ids[a, b], i)
                    for i, (a, b) in enumerate(zip(self._reference.pair_geom1, self._reference.pair_geom2))
                    if pair_ids[a, b] >= 0
                ]
                self._capsule_pair_ids = wp.array(pair_ids, dtype=int, device=self._wp_device)
                self._capsule_slots = wp.full((self.num_envs, len(pairs), 2), -1, dtype=int, device=self._wp_device)
                self._capsule_gaps = wp.empty((self.num_envs, len(pairs)), dtype=float, device=self._wp_device)
                Contact, _, _ = _capsule_slot_kernels()
                self._contact_arrays = Contact()
                for field in fields(mjw.Contact):
                    setattr(self._contact_arrays, field.name, getattr(self._d.contact, field.name))
            if self._plane_capsules or self._capsule_pairs:
                self._m.callback.contactfilter = self._correct_contact_frames
            if ordered_constraints:
                if (
                    self._m.is_sparse
                    or self._m.nflex
                    or self._reference.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP
                    or self._reference.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_ACTUATION
                ):
                    raise ValueError("Ordered constraints require dense, awake rigid worlds with actuation enabled")
                rows, *_ = _constraint_order_kernels()
                source, scratch = rows(), rows()
                for name in rows.vars:
                    value = getattr(self._d.efc, name)
                    setattr(source, name, value)
                    setattr(scratch, name, wp.empty_like(value))
                shape = (self.num_envs, self._d.njmax)
                self._constraint_buffers = (
                    source,
                    scratch,
                    wp.empty(shape, dtype=wp.int64),
                    wp.empty(shape, dtype=int),
                    wp.empty(shape, dtype=int),
                )
                # Warp 3.17 invokes this hook after building constraints and
                # before solving, in both forward() and step1(). Keep ctrl intact.
                self._m.callback.control = self._order_constraints
            if ordered_elliptic:
                if self._integrator == mujoco.mjtIntegrator.mjINT_RK4:
                    raise ValueError("Ordered elliptic solving requires split-step integration")
                self._elliptic_solver = _DenseEllipticSolver(self._m, self._d)
            if ordered_pyramidal:
                self._pyramidal_solver = _DensePyramidalSolver(self._m, self._d)
            self._upload_models(self.models)

    def _order_constraints(self, model, data):
        source, scratch, keys, permutation, inverse = self._constraint_buffers
        _, order, copy_rows, remap = _constraint_order_kernels()
        wp.launch(
            order,
            dim=data.nworld,
            inputs=[
                model.ngeom,
                model.ngeom**2 + model.nv + model.neq,
                data.naconmax,
                data.nefc,
                source,
                data.contact.geom,
                data.contact.geomcollisionid,
                data.contact.efc_address,
                keys,
                permutation,
                inverse,
            ],
        )
        wp.launch(copy_rows, dim=permutation.shape, inputs=[data.nefc, permutation, True, source, scratch])
        wp.launch(copy_rows, dim=permutation.shape, inputs=[data.nefc, permutation, False, scratch, source])
        wp.launch(
            remap,
            dim=data.contact.efc_address.shape,
            inputs=[data.nacon, data.contact.worldid, inverse, data.contact.efc_address],
        )

    def _correct_contact_frames(self, model, data):
        if self._plane_capsules:
            wp.launch(
                _plane_capsule_frame_kernel(),
                dim=data.naconmax,
                inputs=[
                    model.geom_type,
                    data.geom_xmat,
                    data.nacon,
                    data.contact.worldid,
                    data.contact.geom,
                    data.contact.frame,
                ],
                device=self._wp_device,
            )
        if self._capsule_pairs:
            _, index_contacts, expand_contacts = _capsule_slot_kernels()
            self._capsule_slots.fill_(-1)
            wp.launch(
                index_contacts,
                dim=data.naconmax,
                inputs=[data.nacon, self._contact_arrays, self._capsule_pair_ids, self._capsule_slots],
                device=self._wp_device,
            )
            wp.launch(
                expand_contacts,
                dim=self._capsule_slots.shape[:2],
                inputs=[data.geom_xmat, self._capsule_slots, self._contact_arrays, data.nacon],
                device=self._wp_device,
            )
            wp.launch(
                _capsule_contact_kernel(),
                dim=data.naconmax,
                inputs=[
                    model.geom_type,
                    model.geom_size,
                    data.geom_xpos,
                    data.geom_xmat,
                    data.nacon,
                    data.contact.worldid,
                    data.contact.geom,
                    data.contact.dist,
                    data.contact.pos,
                    data.contact.frame,
                    data.contact.geomcollisionid,
                    data.contact.type,
                    data.contact.includemargin,
                    self._capsule_pair_ids,
                    self._capsule_gaps,
                ],
                device=self._wp_device,
            )

    def _check_open(self):
        if self._closed:
            raise RuntimeError("WarpRuntime is closed")

    def _check_models(self, models):
        if len(models) != self.num_envs:
            raise ValueError("Model batch does not match the runtime")
        # Compact batches share one immutable topology and scalar options.
        for model in [models.reference] if isinstance(models, WarpModelBatch) else models:
            if any(getattr(model, k) != v for k, v in self._dimensions.items()):
                raise ValueError("WarpRuntime models must have identical dimensions")
            if any(not np.array_equal(getattr(model, k), v) for k, v in self._topology.items()):
                raise ValueError("WarpRuntime models must have identical topology")
            for name, value in self._scalar_options.items():
                if getattr(model.opt, name) != value:
                    raise ValueError(f"WarpRuntime cannot batch option {name!r}")

    @staticmethod
    def _check_float32(values, name):
        values = np.asarray(values)
        if not np.isfinite(values).all() or np.any(np.abs(values) > np.finfo(np.float32).max):
            raise ValueError(f"Model field {name!r} must be finite in float32")

    def validate_models(self, models):
        """Preflight native models without modifying runtime buffers."""
        self._check_open()
        self._check_models(models)
        if isinstance(models, WarpModelBatch):
            for name in self._model_fields:
                self._check_float32(models.columns[name], name)
            for name, values in models.options.items():
                self._check_float32(values, f"opt.{name}")
            self._check_float32(models.columns["site_size"], "site_size")
            self._check_float32(models.fixed_positions, "fixed geom positions")
            self._check_float32(models.fixed_rotations, "fixed geom rotations")
            return
        for name in self._model_fields:
            self._check_float32([getattr(model, name) for model in models], name)
        for name in self._option_fields:
            source = "impratio" if name == "impratio_invsqrt" else name
            self._check_float32([getattr(model.opt, source) for model in models], f"opt.{source}")
        self._check_float32([model.site_size for model in models], "site_size")

    @contextmanager
    def _scope(self):
        self._check_open()
        if self.device.type == "cpu":
            with wp.ScopedDevice("cpu"):
                yield
            return
        caller = torch.cuda.current_stream(self.device)
        different_stream = caller.cuda_stream != self._torch_stream.cuda_stream
        if different_stream:
            self._torch_stream.wait_stream(caller)
        with (
            torch.cuda.stream(self._torch_stream),
            wp.ScopedDevice(self._wp_device),
            wp.ScopedStream(self._warp_stream),
        ):
            try:
                yield
            finally:
                if different_stream:
                    caller.wait_stream(self._torch_stream)

    def _data_view(self, name):
        return wp.to_torch(getattr(self._d, name))

    @staticmethod
    def _copy_native(array, values):
        view = wp.to_torch(array)
        value = torch.as_tensor(np.asarray(values), device=view.device, dtype=view.dtype)
        view.copy_(value.reshape(view.shape))

    def _upload_models(self, models):
        self.validate_models(models)
        compact = isinstance(models, WarpModelBatch)
        for name in self._model_fields:
            values = models.columns[name] if compact else [getattr(m, name) for m in models]
            self._copy_native(getattr(self._m, name), values)
        for name in self._option_fields:
            if compact:
                values = (
                    1 / np.sqrt(np.maximum(models.options["impratio"], mujoco.mjMINVAL))
                    if name == "impratio_invsqrt"
                    else models.options[name]
                )
            else:
                values = [
                    1 / np.sqrt(max(m.opt.impratio, mujoco.mjMINVAL))
                    if name == "impratio_invsqrt"
                    else getattr(m.opt, name)
                    for m in models
                ]
            self._copy_native(getattr(self._m.opt, name), values)
        if self._capsule_pairs:
            gaps = []
            for model in models:
                values = model.geom_gap[self._capsule_geoms].sum(axis=1)
                for pair, explicit in self._capsule_gap_overrides:
                    values[pair] = model.pair_gap[explicit]
                gaps.append(values)
            self._copy_native(self._capsule_gaps, gaps)
        # Warp deliberately skips geoms welded to the world during kinematics.
        # Native kinematics supplies their transforms after target/floor changes.
        model = models[0]
        body = model.geom_bodyid
        fixed = (model.body_weldid[body] == 0) & (model.body_mocapid[model.body_rootid[body]] == -1)
        self._fixed_indices = torch.as_tensor(np.flatnonzero(fixed), device=self.device)
        if compact:
            positions, rotations = models.fixed_positions, models.fixed_rotations
        else:
            positions, rotations = [], []
            for model in models:
                data = mujoco.MjData(model)
                mujoco.mj_kinematics(model, data)
                positions.append(data.geom_xpos[fixed].copy())
                rotations.append(data.geom_xmat[fixed].reshape(-1, 3, 3).copy())
        self._fixed_positions = torch.as_tensor(np.asarray(positions), device=self.device, dtype=torch.float32)
        self._fixed_rotations = torch.as_tensor(np.asarray(rotations), device=self.device, dtype=torch.float32)
        # site_size is task metadata absent from Warp's physics Model.
        self._native_model_views = {
            "site_size": torch.as_tensor(
                models.columns["site_size"] if compact else np.stack([m.site_size for m in models]),
                device=self.device,
                dtype=torch.float32,
            )
        }
        self.models = models if compact else list(models)

    def _load(self, state):
        mjw.reset_data(self._m, self._d)
        for key, name in self._physical.items():
            view = self._data_view(name)
            view.copy_(state[key].to(device=self.device, dtype=view.dtype).reshape(view.shape))
        self._data_view("geom_xpos").index_copy_(1, self._fixed_indices, self._fixed_positions)
        self._data_view("geom_xmat").index_copy_(1, self._fixed_indices, self._fixed_rotations)
        if self._elliptic_solver is None and self._pyramidal_solver is None:
            mjw.forward(self._m, self._d)
        else:
            mjw.step1(self._m, self._d)
            if self._pyramidal_solver is not None:
                self._pyramidal_forward()
            else:
                self._elliptic_forward()
        self._data_view("qacc_warmstart").copy_(state["warmstart"].to(self.device, torch.float32))

    def load(self, state, models=None):
        """Reconstruct derived data, retaining the checkpoint's solver warmstart."""
        self._check_open()
        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([self.num_envs]):
            raise ValueError("Physical State must have batch [num_envs]")
        converted = {}
        for key, name in self._physical.items():
            if key not in state or state[key].numel() != self._data_view(name).numel():
                raise ValueError(f"Invalid physical State field {key!r}")
            converted[key] = state[key].to(device=self.device, dtype=torch.float32)
            if not bool(torch.isfinite(converted[key]).all()):
                raise ValueError(f"Nonfinite float32 physical State field {key!r}")
        converted = TensorDict(converted, [self.num_envs], device=self.device)
        with self._scope():
            if models is not None:
                self._upload_models(models if isinstance(models, WarpModelBatch) else list(models))
            self._load(converted)
            self._check_result()

    def _elliptic_forward(self):
        mjw.fwd_actuation(self._m, self._d)
        mjw.fwd_acceleration(self._m, self._d)
        self._elliptic_solver.solve()
        mjw.sensor_acc(self._m, self._d)

    def _pyramidal_forward(self):
        mjw.fwd_actuation(self._m, self._d)
        mjw.fwd_acceleration(self._m, self._d)
        self._pyramidal_solver.solve()
        if self._m.opt.run_rne_postconstraint or (
            not (self._m.opt.disableflags & mjw.DisableBit.SENSOR) and self._m.sensor_rne_postconstraint
        ):
            mjw.rne_postconstraint(self._m, self._d)
        mjw.sensor_acc(self._m, self._d, skip_rne_postconstraint=True)

    def _advance(self):
        # dm_control Physics.legacy_step refreshes position/velocity after the
        # integration while retaining acceleration-stage sensors from that step.
        sensors = self._data_view("sensordata")
        # load() computed acceleration sensors already. Warp's step2 adds
        # touch forces without clearing them; native sensorAcc starts afresh.
        sensors.index_fill_(1, self._acc_indices, 0)
        if self._pyramidal_solver is not None:
            for substep in range(self.nsubsteps):
                self._pyramidal_forward()
                mjw.euler(self._m, self._d)
                if substep + 1 < self.nsubsteps:
                    mjw.step1(self._m, self._d)
            torch.index_select(sensors, 1, self._acc_indices, out=self._acc_values)
        elif self._elliptic_solver is not None:
            for substep in range(self.nsubsteps):
                self._elliptic_forward()
                if self._integrator in (mujoco.mjtIntegrator.mjINT_IMPLICIT, mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
                    mjw.implicit(self._m, self._d)
                else:
                    mjw.euler(self._m, self._d)
                if substep + 1 < self.nsubsteps:
                    mjw.step1(self._m, self._d)
            torch.index_select(sensors, 1, self._acc_indices, out=self._acc_values)
        elif self._integrator == mujoco.mjtIntegrator.mjINT_RK4:
            for _ in range(self.nsubsteps):
                mjw.forward(self._m, self._d)
                # Native RK4 skips sensors at its intermediate stages. Warp
                # recomputes them, so retain the initial-stage acceleration.
                torch.index_select(sensors, 1, self._acc_indices, out=self._acc_values)
                mjw.rungekutta4(self._m, self._d)
        else:
            mjw.step2(self._m, self._d)
            for _ in range(self.nsubsteps - 1):
                mjw.step(self._m, self._d)
            torch.index_select(sensors, 1, self._acc_indices, out=self._acc_values)
        mjw.step1(self._m, self._d)
        sensors.index_copy_(1, self._acc_indices, self._acc_values)

    def _record_result(self):
        # Warp also records finite solver-budget exits in this bitmask. Native
        # MuJoCo allows those exits; retain the source's iteration settings.
        self._result_error.zero_()
        wp.launch(
            _result_check_kernel(),
            dim=self.num_envs,
            inputs=[
                self._d.overflow,
                self._d.nacon,
                self._capacity_overflow,
                self._d.naconmax,
                *(getattr(self._d, name) for name in _RESULT_FIELDS),
            ],
            outputs=[self._result_error],
            device=self._wp_device,
        )

    def _check_result(self, *, recorded=False):
        if not recorded:
            self._record_result()
        # Read the captured bitmask once, including on failure. Keep the check
        # at each round's boundary so failed physics cannot commit a checkpoint
        # or run the next round's reset/RNG work.
        error = int(self._result_error_view[0])
        if error & 1:
            raise RuntimeError("DMC MJWarp contact/constraint capacity overflow")
        for index, name in enumerate(_RESULT_FIELDS, start=1):
            if error & (1 << index):
                raise RuntimeError(f"DMC MJWarp produced nonfinite {name}")

    def validate_controls(self, controls):
        """Validate scaled float32 controls before any physics or reset work."""
        self._check_open()
        controls = torch.as_tensor(controls, device=self.device, dtype=torch.float32)
        if controls.shape != (self.num_envs, self._reference.nu) or not bool(torch.isfinite(controls).all()):
            raise ValueError("Controls must be finite with shape [num_envs, nu]")
        return controls

    def advance(self, controls, *, controls_validated=False):
        """Advance one round; the adapter may reuse its validated controls."""
        self._check_open()
        if not controls_validated:
            controls = self.validate_controls(controls)
        with self._scope():
            self._data_view("ctrl").copy_(controls)
            if self.device.type == "cuda" and self._graph is None:
                checkpoint = self.physical_state()
                self._advance()
                wp.synchronize_device(self._wp_device)
                self._load(checkpoint)
                self._data_view("ctrl").copy_(controls)
                with wp.ScopedCapture() as capture:
                    self._advance()
                    self._record_result()
                self._graph = capture.graph
                self._load(checkpoint)
                self._data_view("ctrl").copy_(controls)
            if self._graph is None:
                self._advance()
                self._record_result()
            else:
                wp.capture_launch(self._graph)
            self._check_result(recorded=True)
        return self.snapshot()

    def copy_physical_state(self, destination, *, lanes=None, include_time=True):
        """Copy into owned checkpoint storage, converting directly to its dtype.

        ``lanes`` maps this runtime's rows into a larger checkpoint. The adapter
        excludes time because its public clock uses native double additions.
        """
        self._check_open()
        with self._scope():
            for key, name in self._physical.items():
                if key == "time" and not include_time:
                    continue
                source = self._data_view(name)
                if key == "time":
                    source = source.reshape(self.num_envs, 1)
                target = destination[key]
                if lanes is None:
                    target.copy_(source)
                else:
                    target.index_copy_(0, lanes, source.to(dtype=target.dtype))

    def physical_state(self):
        """Owned float32 physical fields; the adapter owns public State types."""
        self._check_open()
        with self._scope():
            values = {key: self._data_view(name).clone() for key, name in self._physical.items()}
            values["time"] = values["time"].reshape(self.num_envs, 1)
            return TensorDict(values, [self.num_envs], device=self.device)

    def snapshot(self):
        self._check_open()
        if self._torch_stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self._torch_stream)
        return WarpSnapshot(self)

    def close(self):
        if not self._closed:
            if self._torch_stream is not None:
                self._torch_stream.wait_stream(torch.cuda.current_stream(self.device))
                self._torch_stream.synchronize()
            self._graph = self._m = self._d = None
            self._result_error = self._result_error_view = None
            self._constraint_buffers = None
            self._elliptic_solver = None
            self._pyramidal_solver = None
            self._acc_indices = self._acc_values = None
            self._fixed_indices = self._fixed_positions = self._fixed_rotations = None
            self._capsule_pair_ids = self._capsule_slots = self._capsule_gaps = self._contact_arrays = None
            self._native_model_views.clear()
            self._torch_stream = self._warp_stream = None
            self._closed = True
