"""CarRacing-v3 — the env where two different pygame rasterizers meet in one frame.

Reference: ``gymnasium.envs.box2d.car_racing``. Measured against the official renderer on the states
this env reaches: **100.0000% of pixels exact, max deviation 0** at the reference's own 96x96, which
includes the anti-aliased rims, the car, the indicator bars and the score text. See
``docs/research/carracing.md`` for how each of those was recovered.

The reason this file is long is that a CarRacing frame is assembled by FOUR different pieces of
pygame, and three of them had to be recovered by measurement rather than read off the source:

  ``gfxdraw.aapolygon`` + ``filled_polygon``   background, 400 grass and N road quads
  ``pygame.draw.polygon``                      the car and the indicator bars — a DIFFERENT rule
  ``pygame.font`` glyphs                       the score
  ``pygame.transform.smoothscale``             1000x800 -> 96x96, on every frame

MEASURED CONSTANTS, each with how it was found (``scratch/carracing/probe_*.py``):

| what | the plausible reading | what is actually true | how it was found |
|---|---|---|---|
| ``smoothscale`` | a 16.16 fixed-point box filter | the unit is **2^14**: ``space=(0x4000*n)//m``, ``recip=(1<<30)//space``, taps ``(v*w)>>14``, emit ``(acc*recip)>>16`` | 2^16 reproduces 90.30% of samples, 2^14 reproduces **100.0000%**, max deviation 0, over 6 size pairs x 4 image families and 16 real 1000x800 CarRacing surfaces |
| its pass order | one 2-D filter | X then Y through an **8-bit** intermediate | two chained calls are bit-identical to one, on every trial |
| ``pygame.draw.polygon`` | the same scanline fill as gfxdraw | a different rasterizer entirely | they agree on **14 of 404** polygons, 17,167 differing pixels |
| its crossing rule | floor, or ceil, or round, of the exact crossing | ``floor`` on EVEN crossings and ``ceil`` on ODD ones, counting edges in vertex order per row | floor/ceil/round/trunc all top out at 476/800 polygons; the parity rule gives **800/800** and **225/225** on real car geometry |
| its vertices | rounded | truncated toward zero | apex x=100.99 renders identically to 100.00 |
| the score text | a per-digit glyph atlas | kerning makes the WIDTH depend on the digits (78..102 px), so the atlas is keyed by the whole string | ``render("1234")`` is 94 px where the four advances sum to 92 |
| ``hull.color`` | ``(0.8,0,0)*255 = 204`` | 204, but only because 0.8 rounds UP in float64 | read off the live body rather than transcribed |
| fixture vertices | ``HULL_POLY1..4`` as written | Box2D's ``b2PolygonShape::Set`` recomputes the hull and may rotate the order, and vertex order decides the parity rule | baked from a live body |
| Box2D's state | float64 | **float32** — every pose round-trips through float32 | a float64 round-trip assert fails by 2e-6 |

STATE IS THE RENDER STATE. 28 floats: the hull pose and its velocities, four wheel poses, four wheel
phases (the white stripe), four wheel angular velocities (the ABS bars), the accumulated reward (the
score text) and ``t`` (the zoom ramps for the first 50 steps). The steering bar is NOT a state
dimension: a Box2D revolute joint's angle is ``bodyB.angle - bodyA.angle``, so it is already
determined by wheel 0 and the hull — carrying it would be a dimension ``reference_set_state`` could
not honour, and the round-trip assert in ``reference_render`` checks that identity holds.

THE VISITED-TILE MASK IS NOT APPEARANCE, and it is the one place this env's contract is unusual.
``FrictionDetector._contact`` runs ``tile.color[:] = self.env.road_color`` — an in-place write to the
array ``road_poly`` holds — so driving over a tile permanently repaints it from ``road_color + c`` to
``road_color``. Measured: 12 of 394 quads after 120 steps, and restoring the complete geometric state
and re-rendering still leaves 1,056 pixels different at max |d| exactly 5/255. So the frame is NOT a
function of the car's pose and the score, and a port that claimed otherwise would be wrong. The mask
lives in the style tensor because it is per-episode-SHAPED, but unlike every other style in this repo
it changes DURING an episode; a dataset has to record it, which is exactly what "style is state"
already requires.

Per-episode geometry (the track) is a baked table, not state: 394 road quads and their colours, from
one fixed reset. That is the same shape as a maze layout.

THROUGHPUT AND ITS CEILING. The frame is assembled by ``render/gfxdraw_fused.py`` -- one kernel that
walks the polygon list per PIXEL -- rather than by one batched torch op per primitive, which is worth
203 -> 21,820 frames/s at batch 1024 with the parity above unchanged. It stops there, and the reason
is this env's contract rather than the rasterizer: ``smoothscale`` reads every one of the 1000x800
surface's pixels exactly once, so 98.8% of what is rasterized is thrown away, and writing that
surface and resampling it while drawing NOTHING already costs 11.1 ms per 1,024 frames. After binning
a pixel walks a mean of 4.10 primitives, which is about as few as a playfield quad plus a grass quad
plus a road quad can be, and the kernel evaluates them at 168.5 G primitive-pixel evaluations/s. The
remaining host-side dispatch is worth ~14 ms and would take it to ~31,000 f/s; nothing takes it to
the 75,146 f/s at which eight cores decode the 3.4 kB JPEG this all produces. See
``docs/research/carracing.md``.

NO COLOUR VARIATIONS, deliberately, and the reason is the resample. ``check_colors_apply_exactly``
asks whether a requested colour is literally present in the frame, and after a 10.4x box downscale it
never is: a flat region of value v comes out at v-2 (102 -> 101 -> 100 through the two passes, which
is exactly what the official frames show). Declaring RGB knobs here would advertise something the
output cannot express.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..env import Env
from ..registry import register
from ..render import gfxdraw as G
from ..render import gfxdraw_fused as F
from ..spaces import Box, Dict, Discrete


def _check_car_racing_works() -> None:
    """Fail at IMPORT if the reference env cannot be built, not at construction.

    ``render`` is pure torch, but a STATE cannot be produced without Box2D and pygame: ``sample_states``
    rolls the official env out, because the track geometry and the car's fixture poses have no closed
    form. An env that renders but cannot say what to render is not usable, so the dependency is hard and
    is declared as one -- ``ImportError`` here means ``stable_envs/envs/__init__.py`` records it in
    ``import_errors()`` and the env is cleanly ABSENT.

    Registering anyway is not a harmless extra: this file did exactly that and produced 34 errors in the
    smoke subset alone, failing inside every fixture at construction time with ``DependencyNotInstalled``
    rather than being reported once as a missing dependency.

    The probe is the FAILING PATH -- make, reset, render one frame -- not ``import Box2D``. gymnasium
    raises ``DependencyNotInstalled`` from inside ``envs/box2d/car_racing.py``, and pygame is a separate
    requirement that only ``render`` reaches, so a top-level import check passes and the fixture still
    dies. ``lunarlander.py`` carries the same guard for the same reason.
    """
    import gymnasium
    try:
        env = gymnasium.make("CarRacing-v3", render_mode="rgb_array")
        env.reset(seed=0)
        env.unwrapped.render()
        env.close()
    except Exception as e:                          # noqa: BLE001
        raise ImportError(
            f"CarRacing-v3 cannot be reset and rendered: {type(e).__name__}: {e}. This env needs BOTH "
            f"`box2d-py` (for the simulator that produces its states) and `pygame` (for the reference "
            f'renderer parity is measured against); `pip install swig` then `pip install '
            f'"gymnasium[box2d]"`.') from e


_check_car_racing_works()

W, H = 1000, 800
SCALE = 6.0
ZOOM = 2.7
PLAYFIELD = 2000 / SCALE
GRASS_DIM = PLAYFIELD / 20.0
TRACK_WIDTH = 40 / SCALE
TRACK_DETAIL_STEP = 21 / SCALE
MAX_SHAPE_DIM = max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
TWO_PI = 2.0 * math.pi
#: static bound on the Wu walk. Every edge is clipped to the surface first, so no walk exceeds the
#: longer side; a per-edge bound would need a host sync (``render/gfxdraw`` rule 6).
KMAX = 1024
BAKE_SEED = 0
TEXT_LO, TEXT_HI = -100, 1000


# =============================================================================== bakes
_BAKE: dict = {}
_TEXT: dict = {}


def _bake_track(seed: int) -> dict:
    """One reset of the official env, reduced to tables. Everything here is a per-episode constant."""
    key = int(seed)
    if key in _BAKE:
        return _BAKE[key]
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import gymnasium
    from gymnasium.envs.box2d.car_dynamics import SIZE, WHEEL_R, WHEEL_W

    env = gymnasium.make("CarRacing-v3", render_mode="state_pixels")
    env.reset(seed=key)
    u = env.unwrapped
    road_v = np.array([[list(p) for p in poly] for poly, _ in u.road_poly], np.float64)
    road_c = np.array([[int(c) for c in col] for _, col in u.road_poly], np.float32)
    # road_poly interleaves TILES (whose colour is the mutable ``t.color`` array) with the red/white
    # border quads (a plain tuple, never mutated). Only the tiles carry a visited bit.
    is_tile = np.array([isinstance(col, np.ndarray) for _, col in u.road_poly], bool)
    tile_of = np.full(len(is_tile), -1, np.int64)
    tile_of[is_tile] = np.arange(int(is_tile.sum()))

    wl = [[(float(v[0]), float(v[1])) for v in w.fixtures[0].shape.vertices] for w in u.car.wheels]
    assert all(w == wl[0] for w in wl), "the four wheels no longer share one local polygon"
    hull_l = [[(float(v[0]), float(v[1])) for v in f.shape.vertices] for f in u.car.hull.fixtures]
    assert all(len(h) != 3 for h in hull_l) and len(wl[0]) != 3, (
        "a car fixture became a triangle; pygame.draw.polygon dispatches those to draw_filltri, "
        "which is a different rule from the one implemented here")
    g = GRASS_DIM
    grass = [[(g * x + g, g * y), (g * x, g * y), (g * x, g * y + g), (g * x + g, g * y + g)]
             for x in range(-20, 20, 2) for y in range(-20, 20, 2)]
    b = PLAYFIELD
    out = dict(
        road_v=road_v, road_c=road_c, is_tile=is_tile, tile_of=tile_of,
        n_tiles=int(is_tile.sum()),
        grass_v=np.array(grass, np.float64),
        field_v=np.array([[(b, b), (b, -b), (-b, -b), (-b, b)]], np.float64),
        wheel_local=np.array(wl[0], np.float64),
        hull_local=[np.array(h, np.float64) for h in hull_l],
        hull_color=[int(c * 255) for c in u.car.hull.color],
        wheel_color=[int(c * 255) for c in u.car.wheels[0].color],
        wheel_white=[77, 77, 77],
        wheel_w=WHEEL_W * SIZE, wheel_r=WHEEL_R * SIZE,
        road_color=np.array([int(c) for c in u.road_color], np.float32),
        bg_color=np.array([int(c) for c in u.bg_color], np.float32),
        grass_color=np.array([int(c) for c in u.grass_color], np.float32),
        track=np.array([[t[1], t[2], t[3]] for t in u.track], np.float64),   # (beta, x, y)
    )
    env.close()
    _BAKE[key] = out
    return out


def _bake_text() -> tuple:
    """Every string ``"%04i" % reward`` can produce, as a padded grey atlas plus its widths."""
    if "a" in _TEXT:
        return _TEXT["a"], _TEXT["w"], _TEXT["h"]
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.font.init()
    f = pygame.font.Font(pygame.font.get_default_font(), 42)
    surfs = [f.render("%04i" % v, True, (255, 255, 255), (0, 0, 0))
             for v in range(TEXT_LO, TEXT_HI + 1)]
    ws = [s.get_size()[0] for s in surfs]
    hs = {s.get_size()[1] for s in surfs}
    assert len(hs) == 1, f"the score text is not a fixed height: {hs}"
    th = hs.pop()
    atlas = np.zeros((len(surfs), th, max(ws)), np.uint8)
    for i, s in enumerate(surfs):
        buf = pygame.Surface(s.get_size(), depth=32)
        buf.fill((0, 0, 0))
        buf.blit(s, (0, 0))
        a = np.transpose(np.array(pygame.surfarray.pixels3d(buf)), (1, 0, 2))
        atlas[i, :, :a.shape[1]] = a[..., 0]      # grey: R == G == B, measured
    _TEXT["a"], _TEXT["w"], _TEXT["h"] = atlas, np.array(ws, np.int64), th
    return atlas, _TEXT["w"], th


# =============================================================================== smoothscale
def _shrink_plan(n: int, m: int):
    """``(idx, weight, recip)`` for one axis of ``pygame.transform.smoothscale``'s shrink.

    The unit is 2^14, which is the whole difference between exact and 90% (see the module docstring).
    Every output sample is a fixed set of at most 12 input samples with fixed weights, so on the GPU
    this is a gather, not a filter.
    """
    U = 1 << 14
    space = (U * n) // m
    recip = (1 << 30) // space
    rows, cur, counter = [], [], space
    for x in range(n):
        if counter > U:
            cur.append((x, U))
            counter -= U
        else:
            frac = U - counter
            rows.append(cur + [(x, counter)])
            cur = [(x, frac)] if frac else []
            counter = space - frac
        if len(rows) == m:
            break
    assert len(rows) == m and all(sum(w for _, w in r) == space for r in rows)
    K = max(len(r) for r in rows)
    idx = np.zeros((m, K), np.int64)
    wgt = np.zeros((m, K), np.int64)
    for j, r in enumerate(rows):
        for k, (i, w) in enumerate(r):
            idx[j, k], wgt[j, k] = i, w
    return idx, wgt, recip


class _Shrink:
    """One axis of the resample. ``(v*w)>>14`` for a partial tap and ``v`` for a whole one — the two
    are not the same number, which is why the weight 2^14 is kept and branched on.

    ``__call__`` is the readable torch form and is what ``render/gfxdraw_fused.shrink`` is diffed
    against; the env uses the kernel. Not because the torch form is slow in itself -- it is two
    gathers -- but because it widens the surface to int64 to do them, and 1000x800x3 int64 per
    environment is 19 GB at batch 1024 for an output of 27 KB.
    """

    def __init__(self, n, m, device):
        idx, wgt, recip = _shrink_plan(n, m)
        self.idx = torch.as_tensor(idx, device=device).reshape(-1)
        self.w = torch.as_tensor(wgt, device=device)
        self.full = self.w == (1 << 14)
        self.recip, self.m, self.K = int(recip), m, idx.shape[1]
        #: the same two tables as int32, laid out flat for the kernel
        self.idx32 = self.idx.to(torch.int32).contiguous()
        self.w32 = self.w.reshape(-1).to(torch.int32).contiguous()

    def __call__(self, a: torch.Tensor, axis: int) -> torch.Tensor:
        a = a.movedim(axis, 0).to(torch.int64)
        g = a[self.idx].reshape(self.m, self.K, *a.shape[1:])
        sh = (self.m, self.K) + (1,) * (a.dim() - 1)
        acc = torch.where(self.full.reshape(sh), g, (g * self.w.reshape(sh)) >> 14).sum(1)
        return (((acc & 0xFFFF) * self.recip) >> 16).clamp(0, 255).to(torch.uint8).movedim(0, axis)


# =============================================================================== geometry
def _rotate_rad(v, cos, sin):
    """``pygame.math.Vector2.rotate_rad``. cos/sin come from the CALLER because the reference
    normalises the angle into [0, 2pi) with ``fmod`` first, and sin/cos of the normalised angle is
    not the same double as sin/cos of the raw one."""
    return torch.stack([v[..., 0] * cos - v[..., 1] * sin,
                        v[..., 0] * sin + v[..., 1] * cos], -1)


def _to_screen(verts, cos, sin, zoom, trans):
    """``(B,n,2)`` world -> screen. cos/sin are viewed ``(B,1)`` and NOT ``(B,1,1)``: ``_rotate_rad``
    indexes ``v[..., 0]``, which has dropped the coordinate axis, so a ``(B,1,1)`` factor broadcasts
    to ``(B,B,n)`` and silently draws B copies of the wrong geometry."""
    r = _rotate_rad(verts, cos.view(-1, 1), sin.view(-1, 1))
    return r * zoom.view(-1, 1, 1) + trans.view(-1, 1, 2)


def _body_to_world(local, px, py, ang):
    """``f.body.transform * v``. Box2D's transform is about the body ORIGIN, not its centre of mass."""
    c, s = torch.cos(ang).view(-1, 1), torch.sin(ang).view(-1, 1)
    x, y = local[..., 0], local[..., 1]
    return torch.stack([px.view(-1, 1) + x * c - y * s, py.view(-1, 1) + x * s + y * c], -1)


def _pad_verts(v: torch.Tensor, vmax: int) -> torch.Tensor:
    """``(B,n,2)`` or ``(B,P,n,2)`` padded to ``vmax`` vertices by repeating the last one.

    The padding is never READ -- every rule below loops the primitive's own ``nv`` -- so what it
    contains does not matter; repeating the last vertex only keeps the array in range for the
    bounding-box reduction, which masks it anyway.
    """
    n = v.shape[-2]
    if n >= vmax:
        return v
    return torch.cat([v, v[..., -1:, :].expand(*v.shape[:-2], vmax - n, 2)], dim=-2)


def _visible(scr):
    """The reference's own cull: any vertex inside the window grown by ``MAX_SHAPE_DIM``."""
    x, y = scr[..., 0], scr[..., 1]
    return ((-MAX_SHAPE_DIM <= x) & (x <= W + MAX_SHAPE_DIM)
            & (-MAX_SHAPE_DIM <= y) & (y <= H + MAX_SHAPE_DIM)).any(-1)


def _draw_poly_mask(scr, ys, xs):
    """``pygame.draw.polygon``'s ``draw_fillpoly`` -> ``(B,H,W)`` bool.

    The load-bearing line is the parity of the per-ROW crossing counter, which walks edges in VERTEX
    order: even crossings are floored and odd ones ceiled. A given geometric edge therefore changes
    rounding when the number of lower-indexed ACTIVE edges changes — i.e. mid-edge, as ``y`` passes
    some other vertex — which is why no rule written per edge or per side reproduces it, and why a
    sweep over floor/ceil/round/truncate tops out at 476/800 polygons where this gives 800/800.
    The division is float32, as the reference's ``(float)`` cast.
    """
    v = torch.trunc(scr)
    vx = v[..., 0].to(torch.int64)
    vy = v[..., 1].to(torch.int64)
    B, n = vx.shape
    ay, by = vy.roll(1, 1), vy
    ax, bx = vx.roll(1, 1), vx
    up = ay < by
    y1 = torch.where(up, ay, by).unsqueeze(-1)
    y2 = torch.where(up, by, ay).unsqueeze(-1)
    x1 = torch.where(up, ax, bx).unsqueeze(-1)
    x2 = torch.where(up, bx, ax).unsqueeze(-1)
    live = (ay != by).unsqueeze(-1)
    maxy = vy.max(1, keepdim=True).values.unsqueeze(-1)
    miny = vy.min(1, keepdim=True).values.unsqueeze(-1)

    y = ys.view(1, 1, -1)
    act = live & (((y1 <= y) & (y < y2)) | ((y == maxy) & (y2 == maxy)))
    ai = act.to(torch.int32)
    rank = torch.cumsum(ai, dim=1) - ai
    q = ((y - y1) * (x2 - x1)).to(torch.float32) / (y2 - y1).clamp_min(1).to(torch.float32)
    xc = torch.where(rank % 2 == 0, torch.floor(q), torch.ceil(q)).to(torch.int64) + x1

    BIG = 1 << 40
    srt = torch.where(act, xc, torch.full_like(xc, BIG)).sort(dim=1).values
    cnt = act.sum(1)
    xg = xs.view(1, 1, -1)
    mask = torch.zeros((B, ys.numel(), xs.numel()), dtype=torch.bool, device=vx.device)
    for k in range(0, n - 1, 2):
        mask |= ((cnt >= k + 2).unsqueeze(-1) & (xg >= srt[:, k, :].unsqueeze(-1))
                 & (xg <= srt[:, k + 1, :].unsqueeze(-1)))

    flat = (miny == maxy).reshape(-1)
    if bool(flat.any()):                       # a quad one row high: min x to max x
        row = ys.view(1, -1) == vy.min(1, keepdim=True).values
        mask |= (flat.view(-1, 1, 1) & row.unsqueeze(-1)
                 & (xg >= vx.min(1).values.view(-1, 1, 1))
                 & (xg <= vx.max(1).values.view(-1, 1, 1)))

    lo_y, hi_y = miny.reshape(-1, 1), maxy.reshape(-1, 1)
    for e in range(n):                          # horizontal edges strictly inside get one span each
        sel = (vy.roll(1, 1)[:, e] == vy[:, e]) & (vy[:, e] > lo_y[:, 0]) & (vy[:, e] < hi_y[:, 0])
        if not bool(sel.any()):
            continue
        row = ys.view(1, -1) == vy[:, e:e + 1]
        a = torch.minimum(vx[:, e], vx.roll(1, 1)[:, e]).view(-1, 1, 1)
        b = torch.maximum(vx[:, e], vx.roll(1, 1)[:, e]).view(-1, 1, 1)
        mask |= sel.view(-1, 1, 1) & row.unsqueeze(-1) & (xg >= a) & (xg <= b)
    return mask


# =============================================================================== the env
@register
class CarRacing(Env):
    """CarRacing-v3 at its native 96x96. 100.0000% of pixels exact against the official renderer."""

    name = "car-racing"
    state_dim = 28
    state_names = (
        "hull_x", "hull_y", "hull_angle", "hull_vx", "hull_vy", "hull_omega",
        "w0_x", "w0_y", "w0_angle", "w1_x", "w1_y", "w1_angle",
        "w2_x", "w2_y", "w2_angle", "w3_x", "w3_y", "w3_angle",
        "w0_phase", "w1_phase", "w2_phase", "w3_phase",
        "w0_omega", "w1_omega", "w2_omega", "w3_omega",
        "reward", "t")
    #: Step-9 state (contract §3). Five keys over the unchanged 28 columns, grouped by what draws
    #: them: the hull's pose and twist, the four wheels' poses, their rolling phases (the tread
    #: pattern) and their spin rates (the skid marks), and the two scalars the HUD prints. Every
    #: column is covered and none moved, so the frozen layout and its digests are untouched.
    #: ``reward`` and ``time`` are STATE here and not a step result -- they are two numbers this
    #: renderer paints onto the score bar, which is why they are in the row at all.
    state_layout = ((("hull",), 6, 0),
                    (("wheels", "pose"), 12, 6),
                    (("wheels", "phase"), 4, 18),
                    (("wheels", "omega"), 4, 22),
                    (("reward",), 1, 26),
                    (("time",), 1, 27))
    default_size = 96
    reference_id = "CarRacing-v3"

    #: wheel local anchors, for placing the wheels when sampling a pose
    WHEELPOS = ((-55, 80), (55, 80), (-55, -82), (55, -82))
    SIZE = 0.02

    def __init__(self, *a, **kw):
        # The track is baked BEFORE ``super().__init__``: that constructor calls
        # ``make_variation_space()``, and the visited-tile leaf's width is the baked tile count.
        d = self.bk = _bake_track(BAKE_SEED)
        self.n_tiles = d["n_tiles"]
        super().__init__(*a, **kw)
        dev = self.device
        f = dict(device=dev, dtype=torch.float64)
        self.road_v = torch.as_tensor(d["road_v"], **f)
        self.road_c = torch.as_tensor(d["road_c"], device=dev)
        self.tile_of = torch.as_tensor(d["tile_of"], device=dev)
        self.grass_v = torch.as_tensor(d["grass_v"], **f)
        self.field_v = torch.as_tensor(d["field_v"], **f)
        self.wheel_local = torch.as_tensor(d["wheel_local"], **f)
        self.hull_local = [torch.as_tensor(h, **f) for h in d["hull_local"]]
        self.road_color = torch.as_tensor(d["road_color"], device=dev)
        at, wid, th = _bake_text()
        self.atlas = torch.as_tensor(at, device=dev)
        self.widths = torch.as_tensor(wid, device=dev)
        self.text_h = th
        self.gfx = G.Gfx(H, W, dev)
        self.sx = _Shrink(W, 96, dev)
        self.sy = _Shrink(H, 96, dev)

    # --- contract -----------------------------------------------------------------------
    def make_variation_space(self) -> Dict:
        """Two knobs, and neither is a colour — see the module docstring for why the resample makes
        an exact-colour knob unverifiable here.

        ``track.visited`` is the per-tile repaint the FrictionDetector performs in place. It is
        declared as a variation because it is per-episode-SHAPED and a dataset must record it, not
        because it is appearance.
        """
        return Dict({
            "track": Dict({"visited": Box(low=0.0, high=1.0, shape=(self.n_tiles,),
                                          init_value=np.zeros(self.n_tiles, np.float32))}),
            "rendering": Dict({"antialias": Discrete(2, init_value=1)}),
        })

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """The car on a track tile, driving, spread around the whole loop.

        Spreading matters for more than realism: ``check_state_dims_matter`` calls a dimension alive
        if ANY frame in the batch moves, and the visited-tile bits are only visible through the tiles
        a given camera can see, so a pool taken from one stretch of track would report most of the
        mask dead — a property of the pool, not of the renderer. Each pose sees 29-81 road quads.

        ``t`` is sampled from 0.85 upward rather than from 0, and that is a COST bound, not a
        correctness one: the renderer is exact for any ``t``, but ``zoom`` collapses towards
        0.6 px/m as t -> 0, the reference's cull then admits all 794 primitives instead of ~100, and
        because the draw is serial one such state in a batch multiplies every render in the suite by
        7. The range still straddles 1.0, so ``zoom`` genuinely varies and ``t`` is not a dead
        dimension.
        """
        rng = rng or self.rng
        tr = self.bk["track"]
        out = np.zeros((batch, self.state_dim), np.float64)
        i = rng.integers(0, len(tr), batch)
        beta, x, y = tr[i, 0], tr[i, 1], tr[i, 2]
        ang = beta + rng.uniform(-0.5, 0.5, batch)
        sp = rng.uniform(0.5, 25.0, batch)
        out[:, 0], out[:, 1], out[:, 2] = x, y, ang
        out[:, 3], out[:, 4] = sp * np.cos(ang), sp * np.sin(ang)
        out[:, 5] = rng.uniform(-1.5, 1.5, batch)
        for k, (wx, wy) in enumerate(self.WHEELPOS):
            lx, ly = wx * self.SIZE, wy * self.SIZE
            out[:, 6 + 3 * k] = x + lx * np.cos(ang) - ly * np.sin(ang)
            out[:, 7 + 3 * k] = y + lx * np.sin(ang) + ly * np.cos(ang)
            out[:, 8 + 3 * k] = ang + (rng.uniform(-0.4, 0.4, batch) if k < 2 else 0.0)
        out[:, 18:22] = rng.uniform(-10, 10, (batch, 4))
        out[:, 22:26] = rng.uniform(-5, 90, (batch, 4))
        out[:, 26] = rng.uniform(TEXT_LO, TEXT_HI - 1, batch)
        out[:, 27] = rng.uniform(0.85, 3.0, batch)
        return out.astype(np.float32)

    # --- render -------------------------------------------------------------------------
    @torch.no_grad()
    def render(self, states=None, size: int | None = None, supersample: int | None = None,
               variation: dict | None = None, style=None, out=None) -> torch.Tensor:
        want = int(self.size if size is None else size)
        if want != 96:
            raise ValueError(
                f"CarRacing renders at the reference's own 96x96 and nothing else (got size={want}). "
                f"The observation is a smoothscale of a FIXED 1000x800 surface, so another output "
                f"size is a different filter, not a different resolution.")
        if supersample not in (None, 1):
            raise ValueError("CarRacing: supersample is not available; both of its rasterizers are "
                             "integer, so there is no coverage to supersample.")
        s = self._state if states is None else torch.as_tensor(states, device=self.device,
                                                               dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim).to(torch.float64)
        B = s.shape[0]
        sty = self.resolve_style(style, variation, batch=B)
        img = self._draw(s, sty, B)
        if out is not None:
            out.reshape(-1, 96, 96, 3).copy_(img)
            img = out.reshape(-1, 96, 96, 3)
        return img.view(*lead, 96, 96, 3) if lead else img

    def _draw(self, s, sty, B):
        """The whole frame, as ONE list of polygons handed to one kernel, then the resample.

        The reference draws strictly in order and each ``aapolygon`` blends against whatever is
        already on the canvas, so this cannot batch ACROSS polygons -- and for a year that read as
        "so it must be one python step per polygon", which cost 3,575 ms a frame at batch 64 and the
        same 3,575 ms at batch 256. The order is a property of the composite, not of the dispatch:
        every rule here is a closed form in the pixel, so a pixel can walk the list itself.
        ``render/gfxdraw_fused.py`` is that walk; everything below only assembles the list.
        """
        dev = self.device
        on = torch.ones(B, dtype=torch.bool, device=dev)
        ar = torch.arange(B, device=dev)
        gfx_v, gfx_c, gfx_on = [], [], []          # gfxdraw quads, pre-flip
        pre_v, pre_c, pre_on = [], [], []          # pygame.draw polygons, pre-flip
        post_v, post_c, post_on = [], [], []       # pygame.draw polygons, post-flip

        angle = -s[:, 2]
        a = torch.fmod(angle, TWO_PI)
        a = torch.where(a < 0, a + TWO_PI, a)
        cos, sin = torch.cos(a), torch.sin(a)
        t = s[:, 27]
        zoom = 0.1 * SCALE * torch.clamp(1 - t, min=0) + ZOOM * SCALE * torch.clamp(t, max=1)
        scroll = torch.stack([-s[:, 0] * zoom, -s[:, 1] * zoom], -1)
        tr = _rotate_rad(scroll, cos, sin)
        trans = torch.stack([W / 2 + tr[..., 0], H / 4 + tr[..., 1]], -1)

        def const(c):
            return torch.as_tensor(c, dtype=torch.float32, device=dev).view(1, 3).expand(B, 3)

        # 1 background: clip=False, so it is drawn unconditionally
        gfx_v.append(_to_screen(self.field_v[0].unsqueeze(0).expand(B, 4, 2),
                                cos, sin, zoom, trans).unsqueeze(1))
        gfx_c.append(const(self.bk["bg_color"]).unsqueeze(1))
        gfx_on.append(on.unsqueeze(1))

        # 2 grass and 3 road. Both are culled by the reference per POLYGON and per ENVIRONMENT, so
        # the batch cannot share a python `if`; instead the visible ones are gathered into K slots in
        # index order (a stable argsort on the negated predicate), which keeps the draw order and
        # turns 794 primitives into the ~100 the cull actually leaves.
        def culled(all_verts, n_all):
            scr = _to_screen(all_verts.reshape(1, -1, 2).expand(B, n_all * 4, 2),
                             cos, sin, zoom, trans).reshape(B, n_all, 4, 2)
            vis = _visible(scr)
            order = torch.argsort((~vis).to(torch.uint8), dim=1, stable=True)
            K = int(vis.sum(1).max().item())
            return scr, vis, order[:, :K], K

        gscr, gvis, gord, gK = culled(self.grass_v, self.grass_v.shape[0])
        gfx_v.append(gscr.gather(1, gord.view(B, gK, 1, 1).expand(B, gK, 4, 2)))
        gfx_c.append(const(self.bk["grass_color"]).unsqueeze(1).expand(B, gK, 3))
        gfx_on.append(gvis.gather(1, gord))

        rscr, rvis, rord, rK = culled(self.road_v, self.road_v.shape[0])
        visited = sty["track.visited"] > 0.5                       # (B, n_tiles)
        base = self.road_c[rord]                                   # (B,rK,3) unvisited colour
        ti = self.tile_of[rord]                                    # (B,rK)
        was = (ti >= 0) & visited.gather(1, ti.clamp_min(0))
        gfx_v.append(rscr.gather(1, rord.view(B, rK, 1, 1).expand(B, rK, 4, 2)))
        gfx_c.append(torch.where(was.unsqueeze(-1), self.road_color.view(1, 1, 3), base))
        gfx_on.append(rvis.gather(1, rord))

        # 4 the car, through the OTHER rasterizer: wheels (each followed by its stripe), then hull
        wc, ww = const(self.bk["wheel_color"]), const(self.bk["wheel_white"])
        hc = const(self.bk["hull_color"])
        for k in range(4):
            px, py, wa = s[:, 6 + 3 * k], s[:, 7 + 3 * k], s[:, 8 + 3 * k]
            wv = _body_to_world(self.wheel_local.unsqueeze(0).expand(B, 4, 2), px, py, wa)
            pre_v.append(_to_screen(wv, cos, sin, zoom, trans))
            pre_c.append(wc)
            pre_on.append(on)
            sv, sg = self._stripe(px, py, wa, s[:, 18 + k], B)
            pre_v.append(_to_screen(sv, cos, sin, zoom, trans))
            pre_c.append(ww)
            pre_on.append(sg)
        for hl in self.hull_local:
            n = hl.shape[0]
            hv = _body_to_world(hl.unsqueeze(0).expand(B, n, 2), s[:, 0], s[:, 1], s[:, 2])
            pre_v.append(_to_screen(hv, cos, sin, zoom, trans))
            pre_c.append(hc)
            pre_on.append(on)

        # 5 indicators, which the reference draws AFTER ``transform.flip`` and therefore in flipped
        # coordinates. The kernel is told which frame a primitive is in rather than the surface being
        # moved between the two.
        speed = torch.sqrt(s[:, 3] ** 2 + s[:, 4] ** 2)
        steer = s[:, 8] - s[:, 2]                      # revolute joint angle, not a state dimension
        for verts, col, gate in self._indicators(speed, s[:, 22:26], steer, s[:, 5], B):
            post_v.append(verts)
            post_c.append(const(col))
            post_on.append(gate)

        buf = self._raster(B, gfx_v, gfx_c, gfx_on, pre_v, pre_c, pre_on, post_v, post_c, post_on,
                           sty["rendering.antialias"] > 0.5)

        # 6 score, then 7 the resample
        self._blit_text(buf, s[:, 26])
        x = F.shrink(buf.reshape(B * H, W, 3), self.sx)
        return F.shrink(x.reshape(B, H, 96 * 3), self.sy).view(B, 96, 96, 3)

    def _raster(self, B, gfx_v, gfx_c, gfx_on, pre_v, pre_c, pre_on, post_v, post_c, post_on, aa):
        """Pack the three polygon groups into the kernel's flat tables and rasterize.

        A primitive's bounding box is stored in the OUTPUT (flipped) frame whatever frame it was
        built in, so binning and the per-tile reject are one comparison rather than two cases. The
        box is grown by a pixel because an ``aapolygon``'s fringe steps one off its own edge.
        """
        dev = self.device
        VMAX = max([v.shape[1] for v in pre_v] + [4])
        Pg = sum(v.shape[1] for v in gfx_v)
        groups = [(torch.cat(gfx_v, 1), torch.cat(gfx_c, 1), torch.cat(gfx_on, 1))]
        for vs, cs, os_ in ((pre_v, pre_c, pre_on), (post_v, post_c, post_on)):
            if not vs:                      # an empty group is legal; a bisect over the list uses it
                continue
            groups.append((torch.stack([_pad_verts(v, VMAX) for v in vs], 1),
                           torch.stack(cs, 1), torch.stack(os_, 1)))
        nvs = ([4] * Pg + [v.shape[1] for v in pre_v] + [4] * len(post_v))
        Ppost = Pg + len(pre_v)
        P = Ppost + len(post_v)

        vert = torch.cat([_pad_verts(g[0].reshape(B, -1, g[0].shape[2], 2), VMAX) for g in groups], 1)
        vert = torch.trunc(vert).to(torch.int32)                       # (B,P,VMAX,2)
        col = torch.cat([g[1] for g in groups], 1).to(torch.int32)     # (B,P,3)
        gate = torch.cat([g[2] for g in groups], 1)                    # (B,P)
        nv = torch.tensor(nvs, dtype=torch.int32, device=dev).view(1, P)

        vx, vy = vert[..., 0], vert[..., 1]
        live = torch.arange(VMAX, device=dev).view(1, 1, VMAX) < nv.unsqueeze(-1)
        BIG = torch.iinfo(torch.int32).max // 4
        x0 = torch.where(live, vx, torch.full_like(vx, BIG)).amin(-1) - 1
        x1 = torch.where(live, vx, torch.full_like(vx, -BIG)).amax(-1) + 1
        ylo = torch.where(live, vy, torch.full_like(vy, BIG)).amin(-1) - 1
        yhi = torch.where(live, vy, torch.full_like(vy, -BIG)).amax(-1) + 1
        post = torch.arange(P, device=dev).view(1, P) >= Ppost
        y0 = torch.where(post, ylo, H - 1 - yhi)
        y1 = torch.where(post, yhi, H - 1 - ylo)
        off = ~gate
        meta = torch.stack([nv.expand(B, P), col[..., 0], col[..., 1], col[..., 2],
                            torch.where(off, torch.ones_like(x0), x0),
                            torch.where(off, torch.ones_like(y0), y0),
                            torch.where(off, torch.zeros_like(x1), x1),
                            torch.where(off, torch.zeros_like(y1), y1)], -1).contiguous()

        edge = F.edge_table(vx[:, :Pg, :4].reshape(-1, 4).to(torch.int64),
                            vy[:, :Pg, :4].reshape(-1, 4).to(torch.int64), H, W)
        return F.render_surface(vert.contiguous(), meta, edge.reshape(B, Pg, 4, F.EG_W),
                                aa.to(torch.int32).reshape(B).contiguous(), H, W, Pg, Ppost)

    def _stripe(self, px, py, ang, phase, B):
        """The wheel's white stripe: a quad whose two heights are ``cos(phase)`` and
        ``cos(phase+1.2)``, each replaced by its SIGN when the matching sine is positive, and the
        whole quad skipped when both sines are."""
        a1, a2 = phase, phase + 1.2
        s1, s2 = torch.sin(a1), torch.sin(a2)
        c1 = torch.where(s1 > 0, torch.sign(torch.cos(a1)), torch.cos(a1))
        c2 = torch.where(s2 > 0, torch.sign(torch.cos(a2)), torch.cos(a2))
        z = torch.zeros_like(c1)
        wf, wr = self.bk["wheel_w"], self.bk["wheel_r"]
        loc = torch.stack([torch.stack([z - wf, wr * c1], -1),
                           torch.stack([z + wf, wr * c1], -1),
                           torch.stack([z + wf, wr * c2], -1),
                           torch.stack([z - wf, wr * c2], -1)], 1)
        return _body_to_world(loc, px, py, ang), ~((s1 > 0) & (s2 > 0))

    def _indicators(self, speed, omegas, steer, gyro, B):
        """``_render_indicators``, in POST-flip coordinates, with its ``render_if_min`` gate."""
        dev = self.device
        sx, hh = W / 40.0, H / 40.0
        one = torch.ones(B, dtype=torch.bool, device=dev)

        def vert(place, val):
            top, bot = H - (hh + hh * val), torch.full_like(val, H - hh)
            f = lambda c: torch.full_like(val, c)                                   # noqa: E731
            return torch.stack([torch.stack([f(place * sx), top], -1),
                                torch.stack([f((place + 1) * sx), top], -1),
                                torch.stack([f((place + 1) * sx), bot], -1),
                                torch.stack([f(place * sx), bot], -1)], 1)

        def horiz(place, val):
            x0, x1 = torch.full_like(val, place * sx), (place + val) * sx
            y0, y1 = torch.full_like(val, H - 4 * hh), torch.full_like(val, H - 2 * hh)
            return torch.stack([torch.stack([x0, y0], -1), torch.stack([x1, y0], -1),
                                torch.stack([x1, y1], -1), torch.stack([x0, y1], -1)], 1)

        panel = torch.tensor([[W, H], [W, H - 5 * hh], [0, H - 5 * hh], [0, H]],
                             dtype=torch.float64, device=dev).view(1, 4, 2).expand(B, 4, 2)
        out = [(panel, (0, 0, 0), one),
               (vert(5, 0.02 * speed), (255, 255, 255), speed.abs() > 1e-4)]
        for k, place, col in ((0, 7, (0, 0, 255)), (1, 8, (0, 0, 255)),
                              (2, 9, (51, 0, 255)), (3, 10, (51, 0, 255))):
            out.append((vert(place, 0.01 * omegas[:, k]), col, omegas[:, k].abs() > 1e-4))
        out.append((horiz(20, -10.0 * steer), (0, 255, 0), steer.abs() > 1e-4))
        out.append((horiz(30, -0.8 * gyro), (255, 0, 0), gyro.abs() > 1e-4))
        return out

    def _blit_text(self, buf, reward):
        """The score, blitted opaquely. ``"%04i"`` truncates a float toward zero, and the blit x
        follows from ``text_rect.center = (60, 750)``, so it moves with the string's width."""
        dev = self.device
        idx = (torch.trunc(reward).to(torch.int64) - TEXT_LO).clamp(0, self.atlas.shape[0] - 1)
        w = self.widths[idx]
        th = self.text_h
        y0 = 750 - th // 2
        win_w = int(self.widths.max())
        win_lo = 60 - win_w // 2
        x0 = 60 - torch.div(w, 2, rounding_mode="floor")
        j = torch.arange(win_w, device=dev).view(1, -1)
        src = win_lo + j - x0.view(-1, 1)
        ok = (src >= 0) & (src < w.view(-1, 1))
        g = self.atlas[idx.view(-1, 1, 1), torch.arange(th, device=dev).view(1, -1, 1),
                       src.clamp(0, self.atlas.shape[2] - 1).view(-1, 1, win_w)]
        dst = buf[:, y0:y0 + th, win_lo:win_lo + win_w, :]
        dst.copy_(torch.where(ok.view(-1, 1, win_w, 1), g.unsqueeze(-1).expand(-1, -1, -1, 3), dst))

    # --- reference bridge ---------------------------------------------------------------
    @staticmethod
    def reference_env(size: int | None = None):
        """The official env, reset to the SAME track this class baked. ``size`` is ignored: the
        observation is a fixed 96x96 and the surface behind it a fixed 1000x800."""
        import os
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import gymnasium
        env = gymnasium.make("CarRacing-v3", render_mode="state_pixels").unwrapped
        env.reset(seed=BAKE_SEED)
        return env

    @staticmethod
    def reference_set_state(env, st) -> None:
        """Place the official env at ``st``. Nothing here steps the world, which matters: a step
        would run the contact listener, and the contact listener repaints tiles in place."""
        st = np.asarray(st, np.float32)
        env.car.hull.position = (float(st[0]), float(st[1]))
        env.car.hull.angle = float(st[2])
        env.car.hull.linearVelocity = (float(st[3]), float(st[4]))
        env.car.hull.angularVelocity = float(st[5])
        for k, w in enumerate(env.car.wheels):
            w.position = (float(st[6 + 3 * k]), float(st[7 + 3 * k]))
            w.angle = float(st[8 + 3 * k])
            w.phase = float(st[18 + k])
            w.omega = float(st[22 + k])
        env.reward = float(st[26])
        env.t = float(st[27])

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        out = []
        for st in np.asarray(states, np.float32):
            cls.reference_set_state(env, st)
            # Box2D stores every pose as float32, so this round-trips exactly only for a float32
            # state -- which is the repo's own dtype. Asserting at float64 tolerance fails by ~2e-6
            # and reads as a renderer bug.
            got = np.array([*env.car.hull.position, env.car.hull.angle,
                            *env.car.hull.linearVelocity, env.car.hull.angularVelocity], np.float64)
            want = np.asarray(st[:6], np.float64)[[0, 1, 2, 3, 4, 5]]
            if np.abs(got - want).max() > 1e-5:
                raise AssertionError(
                    f"the reference env did not reach the requested state: {got} vs {want}")
            j = float(env.car.wheels[0].joint.angle)
            if abs(j - (float(st[8]) - float(st[2]))) > 1e-4:
                raise AssertionError(
                    f"wheels[0].joint.angle is {j}, not wheel0.angle - hull.angle "
                    f"({float(st[8]) - float(st[2])}) -- the steering bar would be drawn from a "
                    f"quantity this state cannot reproduce")
            out.append(env.render().copy())
        return np.stack(out)
