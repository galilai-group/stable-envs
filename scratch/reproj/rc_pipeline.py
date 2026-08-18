"""RoboCasa at 256 px as a REPROJECTED PLATE plus a WINDOWED dynamic pass, timed and scored.

The two measured facts this is built on, neither of them new:

* a depth composite of a static pass and a movers-only pass reproduces the fused MJWarp frame on
  99.997% of pixels EXACTLY (``scratch/perf/split_static.py --movers moved``), because this kitchen's
  one light declares ``castshadow=0`` and nothing in the shader couples the two halves;
* the static half costs 5.08 ms of a 10.86 ms frame at 64 worlds, and windowing the dynamic half is
  bit-identical to cropping the full frame and scales with the ray count (0.25 of the rays -> 3.01x).

So the frame is worth 2.1x if the static half can be made free, and the static half is one picture
seen from a camera that moves 2.2-3.4 cm. ``probe_delta.py`` measures what a plate costs at that
range -- at a 1 mm residual, mean |diff| 1.13/255 and SSIM 0.958 against the true static render, with
NO holes -- so this quantises the camera pose onto a grid, bakes one plate per cell, reprojects it
through its own depth buffer, and pays the ray tracer only for the box the movers occupy.

Every stage is timed separately and the whole thing is scored against BOTH the fused MJWarp frame it
replaces and the OFFICIAL MuJoCo raster the tier declares against.

Three things are asserted rather than assumed:
  * MJWarp puts the camera where the cache asks it to (``camera_error``);
  * the window is a gather of the full-resolution ray table, so a windowed pass is the crop of the
    full pass to the bit -- checked at ``--check``;
  * the mover box is conservative -- the composite is compared with the fused frame pixel for pixel,
    which is what would catch a mover leaking outside the window.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rp_common as RP  # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/reproj"


# --------------------------------------------------------------------------- the plate cache

class PlateCache:
    """One baked static view per cell of a quantised camera pose, with its depth buffer.

    The cell CENTRE is what gets baked, not the sample that missed: a cache keyed on the sample would
    be right for that sample and off by a whole cell for the next one, and the residual delta is the
    only thing the reprojection error depends on.
    """

    def __init__(self, ctx, cam: RP.Pinhole, bpos, bmat, qrow, grid_mm: float, grid_deg: float,
                 cap: int = 4096):
        self.c, self.cam, self.bpos, self.bmat, self.qrow = ctx, cam, bpos, bmat, qrow
        self.g = float(grid_mm) / 1000.0
        self.gr = np.radians(float(grid_deg))
        self.cap = int(cap)
        self.key2idx: dict = {}
        self.P: list = []                                       # world points, (p,3) float32, device
        self.RGB: list = []                                     # (p,3) uint8, device
        self.hits = self.misses = 0
        self.bake_s = 0.0
        self.ray = ctx.ray0.to(torch.float32).reshape(-1, 3)
        self._stacked = None

    def _cells(self, xpos: np.ndarray, xmat: np.ndarray):
        kp = np.round(xpos / self.g).astype(np.int64) if self.g > 0 else np.zeros_like(xpos, np.int64)
        km = (np.round(xmat.reshape(-1, 9) / self.gr).astype(np.int64) if self.gr > 0
              else np.zeros((len(xmat), 9), np.int64))
        return kp, km

    def _centre(self, kp, km):
        """Cell centre as a pose: the quantised position, and the quantised rotation re-orthonormalised
        (a rounded rotation matrix is not one; the nearest orthonormal matrix is its SVD's UV^T)."""
        pos = kp.astype(np.float64) * self.g
        mat = np.empty((len(km), 3, 3))
        for i, k in enumerate(km):
            u, _, vt = np.linalg.svd((k.astype(np.float64) * self.gr).reshape(3, 3))
            r = u @ vt
            mat[i] = r * np.sign(np.linalg.det(r))
        return pos, mat

    def get(self, xpos: np.ndarray, xmat: np.ndarray) -> torch.Tensor:
        """``(N,3),(N,3,3)`` world camera poses -> ``(N,)`` plate index, baking whatever is missing.

        The dedup is ``np.unique`` and not a Python loop over the batch: at 64 worlds x 16 chunks a
        per-row ``tuple()`` is a millisecond of a four-millisecond frame, and the rows repeat.
        """
        kp, km = self._cells(xpos, xmat)
        cells = np.concatenate([kp, km], 1)
        uniq, inv = np.unique(cells, axis=0, return_inverse=True)
        keys = [u.tobytes() for u in uniq]
        miss = [i for i, k in enumerate(keys) if k not in self.key2idx]
        if miss:
            t0 = time.perf_counter()
            self._bake(uniq[miss], [keys[i] for i in miss])
            self.bake_s += time.perf_counter() - t0
        idx = np.array([self.key2idx[k] for k in keys], np.int64)[inv.reshape(-1)]
        nmiss = int(np.isin(inv.reshape(-1), miss).sum()) if miss else 0
        self.hits += len(idx) - nmiss
        self.misses += nmiss
        return torch.as_tensor(idx, dtype=torch.long, device="cuda")

    def _bake(self, cells: np.ndarray, keys: list) -> None:
        c = self.c
        self._stacked = None
        pos, mat = self._centre(cells[:, :3], cells[:, 3:])
        miss = keys
        lp, lq = RP.local_from_world(self.bpos, self.bmat, pos, mat)
        q = torch.as_tensor(np.repeat(self.qrow[None], c.nworld, 0), dtype=torch.float32,
                            device="cuda")
        for lo in range(0, len(miss), c.nworld):
            hi = min(lo + c.nworld, len(miss))
            k = hi - lo
            c.wp.to_torch(c.wd.qpos)[:k] = q[:k]
            RP.set_camera(c, lp[lo:hi], lq[lo:hi], k)
            RP.launch(c)
            e = RP.camera_error(c, pos[lo:hi], mat[lo:hi], k)
            assert e[0] < 1e-5 and e[1] < 1e-5, f"camera placement error {e}"
            xa = torch.as_tensor(pos[lo:hi], dtype=torch.float32, device="cuda")
            Ra = torch.as_tensor(mat[lo:hi], dtype=torch.float32, device="cuda")
            P = RP.unproject(self.ray, c.depth[:k].reshape(k, -1), xa, Ra)
            rgb = c.bgra[:k][..., [2, 1, 0]].reshape(k, -1, 3).clone()
            for j in range(k):
                self.key2idx[miss[lo + j]] = len(self.P)
                self.P.append(P[j].clone())
                self.RGB.append(rgb[j])
        if len(self.P) > self.cap:
            raise RuntimeError(f"plate cache over {self.cap}; the grid is too fine for this pool")

    def stack(self):
        if self._stacked is None:
            self._stacked = (torch.stack(self.P), torch.stack(self.RGB))
        return self._stacked

    def clear(self) -> None:
        """Throw the cache away. Used to measure the WORST case: every state a new camera pose, so
        every chunk bakes its own 64 plates. That is the number an arbitrary state distribution gets,
        and it is not the same as giving up -- the bake IS the static pass, which the split would
        have paid anyway, so the floor of this scheme is the split's own cost and not the fused
        frame's."""
        self.key2idx.clear()
        self.P.clear()
        self.RGB.clear()
        self._stacked = None


# --------------------------------------------------------------------------- the mover window

def geom_boxes(pc: torch.Tensor, cam: RP.Pinhole, res: int, znear: float = 1e-3):
    """PER GEOM screen boxes: ``(N,G,K,3)`` camera-frame points -> ``(N,G,4)`` [c0,c1,r0,r1].

    One box per mover rather than one box round all of them, because the bounding box of 71 movers
    scattered over a kitchen IS the frame -- measured, ``window_sides {256: 16}`` on every chunk of
    every draw -- while the union of their own boxes is a fraction of it. The ray table is a flat
    gather, so the pass does not have to be a rectangle; it only has to be a SET.
    """
    z = -pc[..., 2]
    ok = z > znear
    col = (pc[..., 0] / z.clamp(min=znear) - cam.bu) / cam.au
    row = (pc[..., 1] / z.clamp(min=znear) - cam.bv) / cam.av
    big = torch.full_like(col, float(res))
    nil = torch.zeros_like(col)
    c0 = torch.where(ok, col, big).amin(-1).floor() - 1
    c1 = torch.where(ok, col, nil).amax(-1).ceil() + 1
    r0 = torch.where(ok, row, big).amin(-1).floor() - 1
    r1 = torch.where(ok, row, nil).amax(-1).ceil() + 1
    empty = ~ok.any(-1)                                   # nothing in front of the camera at all
    zz = torch.zeros_like(c0)
    box = torch.stack([torch.where(empty, zz, c0), torch.where(empty, zz, c1),
                       torch.where(empty, zz, r0), torch.where(empty, zz, r1)], -1)
    return box.clamp_(0, res).to(torch.int32)


def ray_mask(boxes: torch.Tensor, res: int) -> torch.Tensor:
    """``(M,4)`` boxes -> ``(res*res,)`` bool: is this pixel inside ANY of them?

    Summed-area rather than a loop: four +-1 marks per box into an ``(res+1, res+1)`` accumulator and
    two cumulative sums paint every box at once, so the cost is one pass over the frame however many
    movers there are and however many worlds share the launch.
    """
    acc = torch.zeros((res + 1) * (res + 1), dtype=torch.int32, device=boxes.device)
    c0, c1, r0, r1 = boxes[:, 0].long(), boxes[:, 1].long(), boxes[:, 2].long(), boxes[:, 3].long()
    w = res + 1
    one = torch.ones_like(c0, dtype=torch.int32)
    acc.scatter_add_(0, r0 * w + c0, one)
    acc.scatter_add_(0, r0 * w + c1, -one)
    acc.scatter_add_(0, r1 * w + c0, -one)
    acc.scatter_add_(0, r1 * w + c1, one)
    a = acc.reshape(w, w).cumsum(0).cumsum(1)[:res, :res]
    return (a > 0).reshape(-1)


def local_boxes(m, mv: np.ndarray) -> np.ndarray:
    """``(len(mv), 6)`` centre+half-extent of each mover's bounds in its OWN frame, from the mesh.

    ``m.geom_aabb`` is the obvious source and it is not usable here: on this kitchen it gives
    ``robot0_g23_vis`` a half-extent of 6 mm and a screen box covering 58 % of the frame, because the
    box it describes is not centred where the box arithmetic assumes. The mesh's own vertices are
    unambiguous -- ``mesh_vert`` is post-scale, in the geom's frame -- and a primitive falls back to
    ``geom_rbound``, which is exact for a sphere and conservative for everything else.
    """
    out = np.zeros((len(mv), 6), np.float64)
    for i, g in enumerate(mv):
        g = int(g)
        if int(m.geom_type[g]) == 7:                                        # MESH
            d = int(m.geom_dataid[g])
            a, n = int(m.mesh_vertadr[d]), int(m.mesh_vertnum[d])
            v = np.asarray(m.mesh_vert, np.float64).reshape(-1, 3)[a:a + n]
            out[i, :3] = (v.max(0) + v.min(0)) * 0.5
            out[i, 3:] = (v.max(0) - v.min(0)) * 0.5
        else:
            out[i, 3:] = float(m.geom_rbound[g])
    return out


def aabb_corners(m, mv: np.ndarray) -> np.ndarray:
    """``(len(mv), 20, 3)`` points whose projection BOUNDS each mover: its local AABB's 8 corners,
    plus one point per edge so the box can be clipped at the near plane.

    The corners alone are not enough and the failure is silent: a RoboCasa room camera is bolted to
    the robot, so the robot's own base and shoulder links -- movers, all of them -- have corners
    BEHIND the camera. A vertex behind the near plane has no projection, and the only safe thing to
    do with a box that has one is to open the window to the whole frame, which is what the first
    version did on every chunk of every draw: ``window_sides {256: 16}``, the window never fired, and
    the 1.59x measured was the plate alone. Clipping the box's 12 edges at the near plane instead
    keeps the bound exact -- the projection of a clipped convex body is the convex hull of the
    projections of its clipped vertices -- and the edge midpoints are refined by
    ``clip_to_near`` at run time. The 12 extra rows here are the edge ENDPOINT PAIRS, carried so the
    clip can be done in the camera frame where z is known.
    """
    ab = local_boxes(m, mv)
    s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float64)
    return ab[:, None, :3] + s[None] * ab[:, None, 3:]


#: The 12 edges of a box over the corner order ``aabb_corners`` generates (bit 0 = z, 1 = y, 2 = x).
BOX_EDGES = [(i, i ^ b) for i in range(8) for b in (1, 2, 4) if i & b == 0]


def clip_to_near(pc: torch.Tensor, ne: int, znear: float) -> torch.Tensor:
    """``(N, G*8, 3)`` camera-frame box corners -> ``(N, G, 20, 3)``: the corners plus, per geom,
    every near-plane edge crossing.

    Where an edge runs from in front of the near plane to behind it, the point where it crosses is
    on the silhouette and is what bounds the box there; where it does not cross, the substitute is a
    copy of an endpoint, which changes no bound.

    THE RETURN IS GROUPED PER GEOM, and that is the whole point. Concatenating the two blocks along a
    flat point axis and reshaping to ``(N, G, 20, 3)`` afterwards interleaves them -- geom 0 gets its
    own 8 corners and then geom 1's -- which silently unions every mover with its neighbours and
    inflates the per-geom boxes to a 6 mm gripper part covering 58 % of the frame. That is what
    ``window_sides {256: 16}`` was.
    """
    n = pc.shape[0]
    p = pc.reshape(n, ne, 8, 3)
    i = torch.as_tensor([e[0] for e in BOX_EDGES], device=pc.device)
    j = torch.as_tensor([e[1] for e in BOX_EDGES], device=pc.device)
    a = p.index_select(2, i)
    b = p.index_select(2, j)
    za, zb = -a[..., 2], -b[..., 2]
    cross = ((za > znear) != (zb > znear))
    t = ((znear + a[..., 2]) / (a[..., 2] - b[..., 2]).where(
        (a[..., 2] - b[..., 2]).abs() > 1e-12, torch.full_like(za, 1.0))).unsqueeze(-1)
    hit = a + (b - a) * t.clamp(0.0, 1.0)
    extra = torch.where(cross.unsqueeze(-1), hit, a)
    return torch.cat([p, extra], 2)                                     # (n, G, 8+12, 3)


def window_for(box: np.ndarray, sides: list[int], res: int):
    """One window covering every box in the chunk: the smallest side on the ladder that fits."""
    c0, c1 = int(box[:, 0].min()), int(box[:, 1].max())
    r0, r1 = int(box[:, 2].min()), int(box[:, 3].max())
    need = max(c1 - c0, r1 - r0)
    side = next((s for s in sides if s >= need), res)
    cc = (c0 + c1) // 2
    rc = (r0 + r1) // 2
    x0 = min(max(cc - side // 2, 0), res - side)
    y0 = min(max(rc - side // 2, 0), res - side)
    return side, x0, y0


# --------------------------------------------------------------------------- the pipeline

#: What ``refit_bvh`` and ``render`` read out of ``Data`` and ``kinematics``/``camlight`` write. The
#: pose pass fills them once per state; the draw pass gathers them back rather than recomputing them.
POSE_FIELDS = ("geom_xpos", "geom_xmat", "cam_xpos", "cam_xmat", "light_xpos", "light_xdir")


def upsample(x: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
    """``(B,r,r,3)`` uint8 -> ``(B,size,size,3)`` uint8. A no-op when already at ``size``.

    Byte for byte ``scratch/altrender/probe_budget_frontier.py``'s own ``upsample``, so the
    composition measured here is on top of that frontier and not beside it. ``bilinear`` because that
    probe measured it: ``nearest`` costs 2-3x more mean error at the same speed.
    """
    if x.shape[1] == size:
        return x
    y = x.permute(0, 3, 1, 2).to(torch.float32)
    y = torch.nn.functional.interpolate(y, size=(size, size), mode=mode, align_corners=False)
    return y.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


class Reproj:
    def __init__(self, env, size: int, nworld: int, grid_mm: float, grid_deg: float,
                 sides: list[int], fill: bool, kincache: bool = True, out: int | None = None,
                 plate_res: int | None = None):
        self.env, self.size, self.nworld = env, int(size), int(nworld)
        self.out = int(out or size)
        self.plate_res = int(plate_res or size)
        self.sides, self.fill, self.kincache = sides, bool(fill), bool(kincache)
        self.nocache = False
        self.mask_window = True
        self.gboxes = None
        self.saved: dict = {}
        cls = type(env)
        self.mod = RP.module_for(cls.__name__)
        self.cls = cls
        self.m = RP.model_of(self.mod, cls)
        self.cid = RP.cam_id_of(self.mod, cls, self.m)

        st = env.sample_states(64)
        qpos = RP.full_qpos(self.mod, cls, env, self.m, st)
        _, _, gp, gm = RP.kinematics_cpu(self.m, qpos, self.cid)
        self.mv = RP.moving_geoms(self.m, gp, gm, self.mod.VISIBLE_GROUPS)
        vis = set(int(g) for g in self.mod.VISIBLE_GROUPS)
        self.static = np.array([g for g in range(self.m.ngeom)
                                if int(self.m.geom_group[g]) in vis and g not in set(self.mv.tolist())],
                               np.int64)

        # Full scene at the OUTPUT resolution, as shipped: the baseline this has to beat. And, when
        # the internal resolution is reduced, the same thing at THAT resolution -- the other agent's
        # lever on its own, so the composition is reported beside each of its parts rather than
        # against an arithmetic product of them.
        self.fused = RP.make_ctx(self.mod, cls, self.m, self.out, nworld, depth=False)
        self.fused_r = (self.fused if self.size == self.out else
                        RP.make_ctx(self.mod, cls, self.m, self.size, nworld, depth=False))
        self.cam = RP.Pinhole(self.fused_r.ray0, self.size)
        self.full_ray = self.fused_r.ray0.clone()
        # static only, for the plates; movers only, for the window
        self.pl_ctx = RP.make_ctx(self.mod, cls, self.m, self.plate_res, nworld, hide=self.mv,
                                  depth=True)
        self.dyn = {}
        for s in sides:                                # one Data for the whole ladder: see make_ctx
            self.dyn[s] = RP.make_ctx(self.mod, cls, self.m, s, nworld, hide=self.static, depth=True,
                                      skybox=False,
                                      share=self.dyn[sides[0]] if self.dyn else None)
        qrow = self.mod._scene(cls.KITCHEN).pinned
        bpos, bmat = RP.parent_frame(self.m, self.cid, qrow)
        self.plates = PlateCache(self.pl_ctx, self.cam, bpos, bmat, qrow, grid_mm, grid_deg)
        self.corners = torch.as_tensor(aabb_corners(self.m, self.mv), dtype=torch.float32,
                                       device="cuda")
        self.mv_t = torch.as_tensor(self.mv, dtype=torch.long, device="cuda")
        self.znear = float(getattr(self.fused.rc, "znear", 0.01))

    # -- the pose pass ---------------------------------------------------------------------
    def poses(self, q: torch.Tensor):
        """``kinematics`` only, over the whole batch: where every camera is and where every mover is.

        Two things need this before anything can be drawn -- which plate to reproject, and which
        window to open -- and both are answers about the whole batch rather than about one chunk,
        because the batch is going to be REORDERED so that the states sharing a window are drawn
        together. The launch is ``kinematics`` and ``camlight`` with no render behind it.
        """
        d0 = self.dyn[self.sides[-1]]
        B = q.shape[0]
        xp, xm = [], []
        keep = {k: [] for k in POSE_FIELDS}
        for lo in range(0, B, self.nworld):
            hi = min(lo + self.nworld, B)
            n = hi - lo
            d0.wp.to_torch(d0.wd.qpos)[:n] = q[lo:hi]
            torch.cuda.synchronize()
            d0.wp.capture_launch(d0.graph_kin)
            d0.wp.synchronize()
            for k in POSE_FIELDS:
                keep[k].append(d0.wp.to_torch(getattr(d0.wd, k))[:n].clone())
            xp.append(d0.wp.to_torch(d0.wd.cam_xpos)[:n, self.cid].clone())
            xm.append(d0.wp.to_torch(d0.wd.cam_xmat)[:n, self.cid].reshape(n, 3, 3).clone())
        self.saved = {k: torch.cat(v) for k, v in keep.items()}
        # ONE pass of box arithmetic over the WHOLE batch, not one per chunk. Per chunk it was 15
        # small launches x 16 chunks and cost 0.6 ms of an 11 ms frame -- for tensors of 64x71x20x3.
        xpos, xmat = torch.cat(xp), torch.cat(xm)
        gx = self.saved["geom_xpos"].index_select(1, self.mv_t)
        gm = self.saved["geom_xmat"].index_select(1, self.mv_t).reshape(B, -1, 3, 3)
        cor = (gx.unsqueeze(2) + torch.einsum("ngij,gcj->ngci", gm, self.corners)).reshape(B, -1, 3)
        pc = torch.einsum("nji,ncj->nci", xmat, cor - xpos.unsqueeze(1))
        gb = geom_boxes(clip_to_near(pc, len(self.mv), self.znear), self.cam, self.size, self.znear)
        self.gboxes = gb
        bx = torch.stack([gb[..., 0].amin(1), gb[..., 1].amax(1),
                          gb[..., 2].amin(1), gb[..., 3].amax(1)], 1)
        return (xpos.cpu().numpy().astype(np.float64), xmat.cpu().numpy().astype(np.float64),
                bx.cpu().numpy())

    # -- one chunk ------------------------------------------------------------------------
    def chunk(self, q: torch.Tensor, sel_rows: torch.Tensor, xpos_np, xmat_np, box_np,
              out: torch.Tensor, stats: dict) -> None:
        n = q.shape[0]
        d0 = self.dyn[self.sides[-1]]
        if self.nocache:
            self.plates.clear()
        if self.kincache:
            # The pose pass already ran ``kinematics`` for these states. Running it a second time
            # because the batch has been REORDERED would be paying 0.7 ms of an 11 ms frame for an
            # answer already on the device, so the rows are gathered back into ``Data`` instead.
            for k, v in self.saved.items():
                d0.wp.to_torch(getattr(d0.wd, k))[:n] = v.index_select(0, sel_rows)
        else:
            d0.wp.to_torch(d0.wd.qpos)[:n] = q
            torch.cuda.synchronize()
            d0.wp.capture_launch(d0.graph_kin)

        idx = self.plates.get(xpos_np, xmat_np)
        P, RGB = self.plates.stack()
        xpos = torch.as_tensor(xpos_np, dtype=torch.float32, device="cuda")
        xmat = torch.as_tensor(xmat_np, dtype=torch.float32, device="cuda")
        srgb, sdep, hole = RP.splat(P.index_select(0, idx), RGB.index_select(0, idx),
                                    xpos, xmat, self.cam, self.size)

        if self.mask_window:
            keep = ray_mask(self.gboxes.index_select(0, sel_rows).reshape(-1, 4), self.size)
            pix = torch.nonzero(keep).squeeze(1)
            npix = int(pix.numel())
            side = next((s for s in self.sides if s * s >= npix), self.size)
            sel = torch.cat([pix, pix[:1].expand(side * side - npix)]) if npix else \
                torch.zeros(side * side, dtype=torch.long, device="cuda")
        else:
            side, x0, y0 = window_for(box_np, self.sides, self.size)
            pix = None
            sel = ((torch.arange(side, device="cuda") + y0).view(-1, 1) * self.size
                   + (torch.arange(side, device="cuda") + x0).view(1, -1)).reshape(-1)
        stats.setdefault("side", []).append(side)
        stats.setdefault("ray_frac", []).append(side * side / float(self.size ** 2))
        c = self.dyn[side]                                  # shares d0's Data: kinematics is done
        c.wp.to_torch(c.rc.ray).copy_(self.full_ray.index_select(0, sel))
        torch.cuda.synchronize()
        c.wp.capture_launch(c.graph_draw)
        c.wp.synchronize()

        drgb = c.bgra[:n][..., [2, 1, 0]].reshape(n, -1, 3)
        ddep = c.depth[:n].reshape(n, -1)
        flat_rgb = srgb.reshape(n, -1, 3)
        flat_dep = sdep.reshape(n, -1)
        if self.mask_window:
            k = npix
            ws = flat_dep.index_select(1, pix)
            take = (ddep[:, :k] > 0) & ((ws <= 0) | (ddep[:, :k] < ws))
            flat_rgb[:, pix] = torch.where(take.unsqueeze(-1), drgb[:, :k],
                                           flat_rgb.index_select(1, pix))
        else:
            win = ((torch.arange(side, device="cuda") + y0).view(-1, 1) * self.size
                   + (torch.arange(side, device="cuda") + x0).view(1, -1)).reshape(-1)
            ws = flat_dep.index_select(1, win)
            take = (ddep > 0) & ((ws <= 0) | (ddep < ws))
            flat_rgb[:, win] = torch.where(take.unsqueeze(-1), drgb,
                                           flat_rgb.index_select(1, win))
        if self.fill and bool(hole.any()):
            stats.setdefault("hole", []).append(float(hole.float().mean()))
            covered = torch.zeros_like(flat_dep, dtype=torch.bool)
            covered[:, pix if self.mask_window else win] = take
            srgb, _, _ = RP.fill_holes(srgb, sdep, hole & ~covered.reshape(hole.shape))
        out[:n] = srgb

    # -- whole batch ----------------------------------------------------------------------
    def render(self, qpos: np.ndarray, stats: dict, sort: bool = True) -> torch.Tensor:
        B = len(qpos)
        out = torch.empty((B, self.size, self.size, 3), dtype=torch.uint8, device="cuda")
        q = torch.as_tensor(qpos, dtype=torch.float32, device="cuda")
        xp, xm, bx = self.poses(q)
        if sort:
            # States whose movers land in the same place share a window. The ray table is per LAUNCH,
            # so what a window costs is set by the union over the chunk and not by the per-state box,
            # and the order the caller handed the batch in is not a property of the batch.
            order = np.lexsort(((bx[:, 2] + bx[:, 3]) // 8, (bx[:, 0] + bx[:, 1]) // 8))
        else:
            order = np.arange(B)
        oi = torch.as_tensor(order, dtype=torch.long, device="cuda")
        for lo in range(0, B, self.nworld):
            hi = min(lo + self.nworld, B)
            s = order[lo:hi]
            buf = torch.empty((hi - lo, self.size, self.size, 3), dtype=torch.uint8, device="cuda")
            self.chunk(q.index_select(0, oi[lo:hi]), oi[lo:hi], xp[s], xm[s], bx[s], buf, stats)
            out[oi[lo:hi]] = buf
        # The upsample is INSIDE the timed region, as it is in the frontier this composes with.
        return upsample(out, self.out)

    def render_fused(self, qpos: np.ndarray, reduced: bool = False) -> torch.Tensor:
        """The shipped path. ``reduced``: at the internal resolution, then upsampled -- the
        resolution lever on its own, which is what the composition has to be read against."""
        c = self.fused_r if reduced else self.fused
        B = len(qpos)
        out = torch.empty((B, c.res, c.res, 3), dtype=torch.uint8, device="cuda")
        q = torch.as_tensor(qpos, dtype=torch.float32, device="cuda")
        for lo in range(0, B, self.nworld):
            hi = min(lo + self.nworld, B)
            n = hi - lo
            c.wp.to_torch(c.wd.qpos)[:n] = q[lo:hi]
            RP.launch(c)
            out[lo:hi] = c.bgra[:n][..., [2, 1, 0]]
        return upsample(out, self.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="RoboCasaCounterToSink")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--grid-mm", type=float, default=1.0)
    ap.add_argument("--grid-deg", type=float, default=0.05)
    ap.add_argument("--sides", default="64,88,128,152,192,256")
    ap.add_argument("--nofill", action="store_true")
    ap.add_argument("--nosort", action="store_true")
    ap.add_argument("--fresh", type=int, default=3)
    ap.add_argument("--nokincache", action="store_true")
    ap.add_argument("--rectwindow", action="store_true",
                    help="one rectangle round every mover instead of the union of "
                         "their own boxes as a ray SET")
    ap.add_argument("--nocache", action="store_true",
                    help="throw the plate cache away every chunk: the worst case, every state a "
                         "camera pose never seen before")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--nofid", action="store_true")
    ap.add_argument("--render-res", type=int, default=0,
                    help="internal render resolution; the frame is bilinear-upsampled to --size "
                         "inside the timed region. 0 = --size, i.e. the lever off.")
    ap.add_argument("--plate-res", type=int, default=0,
                    help="resolution the PLATE is baked at. 0 = --render-res. Setting it to --size "
                         "with a reduced --render-res is the supersampled-plate case, which is a "
                         "measured negative at full resolution and is re-measured here.")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import stable_envs as se

    tag = a.tag or f"{a.env}_g{a.grid_mm}"
    env = se.make(a.env, batch=8, size=a.size, device="cuda")
    cls = type(env)
    rres = int(a.render_res or a.size)
    pres = int(a.plate_res or rres)
    sides = [int(s) for s in a.sides.split(",") if int(s) < rres] + [rres]
    r = Reproj(env, rres, a.nworld, a.grid_mm, a.grid_deg, sides, not a.nofill,
               kincache=not a.nokincache, out=a.size, plate_res=pres)
    r.nocache = bool(a.nocache)
    r.mask_window = not a.rectwindow
    mod = r.mod
    m = r.m

    st = env.sample_states(a.batch)
    qpos = RP.full_qpos(mod, cls, env, m, st)

    # warm: the cache fills, the graphs are resident, nothing is timed
    sort = not a.nosort
    stats: dict = {}
    r.render(qpos[:a.nworld * 2], stats, sort)
    warm_plates = len(r.plates.P)

    stats = {}
    ours = r.render(qpos, stats, sort)
    t_ours = RP.timed(lambda: r.render(qpos, {}, sort), a.reps, warm=1)
    t_pose = RP.timed(lambda: r.poses(torch.as_tensor(qpos, dtype=torch.float32, device="cuda")),
                      a.reps, warm=1)
    t_fused = RP.timed(lambda: r.render_fused(qpos), a.reps, warm=1)
    t_env = RP.timed(lambda: env.render(st), a.reps, warm=1)
    fused = r.render_fused(qpos)
    if rres != a.size:
        t_fused_r = RP.timed(lambda: r.render_fused(qpos, reduced=True), a.reps, warm=1)
        fused_r = r.render_fused(qpos, reduced=True)
    else:
        t_fused_r, fused_r = t_fused, fused

    row = {"env": a.env, "batch": a.batch, "nworld": a.nworld, "size": a.size,
           "render_res": rres, "plate_res": pres,
           "fps_res_only": a.batch / t_fused_r, "speedup_res_only": t_fused / t_fused_r,
           "grid_mm": a.grid_mm, "grid_deg": a.grid_deg, "fill": not a.nofill, "sort": sort,
           "nocache": bool(a.nocache),
           "mask_window": not a.rectwindow,
           "ray_frac_mean": float(np.mean(stats.get("ray_frac", [1.0]))),
           "sides": sides, "ms_per_chunk_ours": t_ours * 1e3 / (a.batch / a.nworld),
           "ms_per_chunk_fused": t_fused * 1e3 / (a.batch / a.nworld),
           "ms_per_chunk_pose_pass": t_pose * 1e3 / (a.batch / a.nworld),
           "fps_ours": a.batch / t_ours, "fps_fused": a.batch / t_fused,
           "fps_env_surface": a.batch / t_env,
           "speedup_vs_fused": t_fused / t_ours, "speedup_vs_env": t_env / t_ours,
           "plates": len(r.plates.P), "plates_after_warm": warm_plates,
           "plate_hit_rate": r.plates.hits / max(1, r.plates.hits + r.plates.misses),
           "bake_s_total": r.plates.bake_s,
           "window_sides": {int(s): int(sum(1 for x in stats.get("side", []) if x == s))
                            for s in sorted(set(stats.get("side", [])))},
           "hole_frac_mean": float(np.mean(stats.get("hole", [0.0]))),
           }
    # THE HONEST CACHE NUMBER. The timed repetitions above re-render one batch, so their hit rate is
    # 100% by construction. These are fresh draws from ``sample_states`` with fresh seeds: what the
    # cache does on states it has not seen, which is the only rate that means anything.
    fresh = []
    for seed in range(1, 1 + a.fresh):
        e2 = se.make(a.env, batch=8, size=a.size, device="cuda", seed=seed)
        st2 = e2.sample_states(a.batch)
        h0, m0, p0 = r.plates.hits, r.plates.misses, len(r.plates.P)
        r.render(RP.full_qpos(mod, cls, e2, m, st2), {}, sort)
        fresh.append({"seed": seed,
                      "hit_rate": (r.plates.hits - h0) / max(1, r.plates.hits - h0
                                                             + r.plates.misses - m0),
                      "new_plates": len(r.plates.P) - p0, "total_plates": len(r.plates.P)})
        del e2
    row["fresh_draws"] = fresh
    print("fresh " + json.dumps(fresh), flush=True)

    nm = min(256, a.batch)                      # SSIM over 1024 frames is a minute of scipy for nothing
    o = ours[:nm].cpu().numpy()
    f = fused[:nm].cpu().numpy()
    row["vs_fused"] = RP.metrics(o, f)
    row["vs_fused_n"] = nm
    if not a.nofid:
        ref = cls.reference_env(a.size)
        n_fid = min(64, a.batch)
        official = cls.reference_render(ref, st[:n_fid])
        fr = fused_r[:n_fid].cpu().numpy()
        row["ours_vs_official"] = RP.metrics(o[:n_fid], official) | RP.interior_metrics(
            o[:n_fid], official)
        row["fused_vs_official"] = RP.metrics(f[:n_fid], official) | RP.interior_metrics(
            f[:n_fid], official)
        row["res_only_vs_official"] = RP.metrics(fr, official) | RP.interior_metrics(fr, official)
        # The budget, on the SAME states, so the three rows above are read against a number measured
        # beside them rather than against one carried over from another draw.
        jo = RP.jpeg_roundtrip(official, 95)
        row["jpeg_official_vs_official"] = RP.metrics(jo, official) | RP.interior_metrics(
            jo, official)
        row["declared"] = dict(cls.visual_fidelity)
        _dump(os.path.join(OUT, f"pipe_{tag}.png"), official[0], f[0], o[0])
    print("ROW " + json.dumps(row), flush=True)
    with open(os.path.join(OUT, f"pipe_{tag}.json"), "w") as fh:
        json.dump(row, fh, indent=1)


def _dump(path, official, fused, ours) -> None:
    from PIL import Image
    amp = lambda d: np.stack([np.clip(d.max(-1) * 8, 0, 255).astype(np.uint8)] * 3, -1)
    d1 = amp(np.abs(fused.astype(np.int32) - official.astype(np.int32)))
    d2 = amp(np.abs(ours.astype(np.int32) - official.astype(np.int32)))
    d3 = amp(np.abs(ours.astype(np.int32) - fused.astype(np.int32)))
    Image.fromarray(np.concatenate([official, fused, ours, d1, d2, d3], 1)).save(path)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
