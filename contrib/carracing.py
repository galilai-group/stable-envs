from __future__ import annotations

import math

import numpy as np
import torch

from stable_envs.env import Env
from stable_envs.registry import register
from stable_envs.render import gfxdraw as G
from stable_envs.render.gfxdraw import fused as F
from stable_envs.spaces import Box, Dict, Discrete


def _check_car_racing_works() -> None:
    """Require a working Box2D reset and pygame render before registration.

    Import checks alone miss dependencies used only during reset or rendering."""
    import gymnasium

    try:
        env = gymnasium.make("CarRacing-v3", render_mode="rgb_array")
        env.reset(seed=0)
        env.unwrapped.render()
        env.close()
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            f"CarRacing-v3 cannot be reset and rendered: {type(e).__name__}: {e}. This env needs BOTH "
            f"`box2d-py` (for the simulator that produces its states) and `pygame` (for the reference "
            f"renderer parity is measured against); `pip install swig` then `pip install "
            f'"gymnasium[box2d]"`.'
        ) from e


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
# Clipping bounds each Wu walk by the surface’s longer side; avoid per-edge host sync.
KMAX = 1024
BAKE_SEED = 0
TEXT_LO, TEXT_HI = -100, 1000


_BAKE: dict = {}
_TEXT: dict = {}


def _bake_track(seed: int) -> dict:
    """Cache source track geometry, colours and Box2D fixture vertex order for one seed."""
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
        "which is a different rule from the one implemented here"
    )
    g = GRASS_DIM
    grass = [
        [(g * x + g, g * y), (g * x, g * y), (g * x, g * y + g), (g * x + g, g * y + g)]
        for x in range(-20, 20, 2)
        for y in range(-20, 20, 2)
    ]
    b = PLAYFIELD
    out = dict(
        road_v=road_v,
        road_c=road_c,
        is_tile=is_tile,
        tile_of=tile_of,
        n_tiles=int(is_tile.sum()),
        grass_v=np.array(grass, np.float64),
        field_v=np.array([[(b, b), (b, -b), (-b, -b), (-b, b)]], np.float64),
        wheel_local=np.array(wl[0], np.float64),
        hull_local=[np.array(h, np.float64) for h in hull_l],
        hull_color=[int(c * 255) for c in u.car.hull.color],
        wheel_color=[int(c * 255) for c in u.car.wheels[0].color],
        wheel_white=[77, 77, 77],
        wheel_w=WHEEL_W * SIZE,
        wheel_r=WHEEL_R * SIZE,
        road_color=np.array([int(c) for c in u.road_color], np.float32),
        bg_color=np.array([int(c) for c in u.bg_color], np.float32),
        grass_color=np.array([int(c) for c in u.grass_color], np.float32),
        track=np.array([[t[1], t[2], t[3]] for t in u.track], np.float64),  # (beta, x, y)
    )
    env.close()
    _BAKE[key] = out
    return out


def _bake_text() -> tuple:
    """Cache whole score strings in a padded grey atlas over TEXT_LO..TEXT_HI.

    Kerning changes string width, so a per-digit atlas would place text differently."""
    if "a" in _TEXT:
        return _TEXT["a"], _TEXT["w"], _TEXT["h"]
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.font.init()
    f = pygame.font.Font(pygame.font.get_default_font(), 42)
    surfs = [f.render("%04i" % v, True, (255, 255, 255), (0, 0, 0)) for v in range(TEXT_LO, TEXT_HI + 1)]
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
        atlas[i, :, : a.shape[1]] = a[..., 0]  # grey: R == G == B, measured
    _TEXT["a"], _TEXT["w"], _TEXT["h"] = atlas, np.array(ws, np.int64), th
    return atlas, _TEXT["w"], th


def _shrink_plan(n: int, m: int):
    """Return gather indices, 2¹⁴ fixed-point weights and reciprocal for one shrink axis."""
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
    """Source-compatible shrink arithmetic and kernel tables for one axis.

    The Torch form is a readable differential reference; rendering uses the fused
    kernel to avoid widening the full surface to int64."""

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


def _rotate_rad(v, cos, sin):
    """Rotate using caller-supplied sin/cos of the angle normalized by fmod to [0, 2π)."""
    return torch.stack([v[..., 0] * cos - v[..., 1] * sin, v[..., 0] * sin + v[..., 1] * cos], -1)


def _to_screen(verts, cos, sin, zoom, trans):
    """Transform ``(B, N, 2)`` vertices using ``(B, 1)`` trigonometric factors.

    An extra factor dimension would broadcast across environments."""
    r = _rotate_rad(verts, cos.view(-1, 1), sin.view(-1, 1))
    return r * zoom.view(-1, 1, 1) + trans.view(-1, 1, 2)


def _body_to_world(local, px, py, ang):
    """Transform fixture vertices about the Box2D body origin, not its centre of mass."""
    c, s = torch.cos(ang).view(-1, 1), torch.sin(ang).view(-1, 1)
    x, y = local[..., 0], local[..., 1]
    return torch.stack([px.view(-1, 1) + x * c - y * s, py.view(-1, 1) + x * s + y * c], -1)


def _pad_verts(v: torch.Tensor, vmax: int) -> torch.Tensor:
    """Repeat the last vertex to pad the array; rasterization uses each polygon’s true count."""
    n = v.shape[-2]
    if n >= vmax:
        return v
    return torch.cat([v, v[..., -1:, :].expand(*v.shape[:-2], vmax - n, 2)], dim=-2)


def _visible(scr):
    """Apply the source vertex-in-window cull, expanded by MAX_SHAPE_DIM."""
    x, y = scr[..., 0], scr[..., 1]
    return ((-MAX_SHAPE_DIM <= x) & (x <= W + MAX_SHAPE_DIM) & (-MAX_SHAPE_DIM <= y) & (y <= H + MAX_SHAPE_DIM)).any(
        -1
    )


def _draw_poly_mask(scr, ys, xs):
    """Return pygame.draw polygon coverage as a Boolean ``(B, H, W)`` mask.

    Count active crossings in vertex order per scanline: floor even crossings
    and ceil odd ones, using float32 division. Sorting edges first changes coverage."""
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
        mask |= (
            (cnt >= k + 2).unsqueeze(-1) & (xg >= srt[:, k, :].unsqueeze(-1)) & (xg <= srt[:, k + 1, :].unsqueeze(-1))
        )

    flat = (miny == maxy).reshape(-1)
    if bool(flat.any()):  # a quad one row high: min x to max x
        row = ys.view(1, -1) == vy.min(1, keepdim=True).values
        mask |= (
            flat.view(-1, 1, 1)
            & row.unsqueeze(-1)
            & (xg >= vx.min(1).values.view(-1, 1, 1))
            & (xg <= vx.max(1).values.view(-1, 1, 1))
        )

    lo_y, hi_y = miny.reshape(-1, 1), maxy.reshape(-1, 1)
    for e in range(n):  # horizontal edges strictly inside get one span each
        sel = (vy.roll(1, 1)[:, e] == vy[:, e]) & (vy[:, e] > lo_y[:, 0]) & (vy[:, e] < hi_y[:, 0])
        if not bool(sel.any()):
            continue
        row = ys.view(1, -1) == vy[:, e : e + 1]
        a = torch.minimum(vx[:, e], vx.roll(1, 1)[:, e]).view(-1, 1, 1)
        b = torch.maximum(vx[:, e], vx.roll(1, 1)[:, e]).view(-1, 1, 1)
        mask |= sel.view(-1, 1, 1) & row.unsqueeze(-1) & (xg >= a) & (xg <= b)
    return mask


@register
class CarRacing(Env):
    """Render a fixed CarRacing track at native 96×96 resolution.

    The 28-column render pose includes hull/wheel motion, score and zoom time;
    it is not a full Box2D checkpoint. Record ``track.visited`` alongside it:
    the source repaints tiles during driving, so pose alone does not fix pixels.
    Rendering preserves separate gfxdraw/draw polygon rules, then shrinks the
    1000×800 surface horizontally and vertically through a uint8 intermediate."""

    family = "carracing"
    domain = "carracing"

    name = "car-racing"
    state_dim = 28
    state_names = (
        "hull_x",
        "hull_y",
        "hull_angle",
        "hull_vx",
        "hull_vy",
        "hull_omega",
        "w0_x",
        "w0_y",
        "w0_angle",
        "w1_x",
        "w1_y",
        "w1_angle",
        "w2_x",
        "w2_y",
        "w2_angle",
        "w3_x",
        "w3_y",
        "w3_angle",
        "w0_phase",
        "w1_phase",
        "w2_phase",
        "w3_phase",
        "w0_omega",
        "w1_omega",
        "w2_omega",
        "w3_omega",
        "reward",
        "t",
    )
    # Render fields include HUD inputs; they do not imply simulated rewards or episode time.
    state_layout = (
        (("hull",), 6, 0),
        (("wheels", "pose"), 12, 6),
        (("wheels", "phase"), 4, 18),
        (("wheels", "omega"), 4, 22),
        (("reward",), 1, 26),
        (("time",), 1, 27),
    )
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

    def make_variation_space(self) -> Dict:
        """Expose visited-tile repainting and antialiasing.

        The visited mask is stored in style because its width depends on track geometry,
        but it changes during an episode and must be saved with render poses."""
        return Dict(
            {
                "track": Dict(
                    {
                        "visited": Box(
                            low=0.0, high=1.0, shape=(self.n_tiles,), init_value=np.zeros(self.n_tiles, np.float32)
                        )
                    }
                ),
                "rendering": Dict({"antialias": Discrete(2, init_value=1)}),
            }
        )

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample synthetic car poses across the fixed track, not source rollouts.

        Times start at 0.85 to avoid the wide initial camera admitting every polygon;
        the range still spans the zoom ramp. Track coverage exercises visited-tile bits."""
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

    @torch.no_grad()
    def _render(
        self,
        states=None,
        size: int | None = None,
        supersample: int | None = None,
        variation: dict | None = None,
        style=None,
        out=None,
    ) -> torch.Tensor:
        want = int(self.size if size is None else size)
        if want != 96:
            raise ValueError(
                f"CarRacing renders at the reference's own 96x96 and nothing else (got size={want}). "
                f"The observation is a smoothscale of a FIXED 1000x800 surface, so another output "
                f"size is a different filter, not a different resolution."
            )
        if supersample not in (None, 1):
            raise ValueError(
                "CarRacing: supersample is not available; both of its rasterizers are "
                "integer, so there is no coverage to supersample."
            )
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
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
        """Assemble ordered polygon groups, rasterize once, draw text, then shrink.

        Antialiased edges blend against earlier polygons, so painter order must
        survive culling and batching."""
        dev = self.device
        on = torch.ones(B, dtype=torch.bool, device=dev)
        gfx_v, gfx_c, gfx_on = [], [], []  # gfxdraw quads, pre-flip
        pre_v, pre_c, pre_on = [], [], []  # pygame.draw polygons, pre-flip
        post_v, post_c, post_on = [], [], []  # pygame.draw polygons, post-flip

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
        gfx_v.append(_to_screen(self.field_v[0].unsqueeze(0).expand(B, 4, 2), cos, sin, zoom, trans).unsqueeze(1))
        gfx_c.append(const(self.bk["bg_color"]).unsqueeze(1))
        gfx_on.append(on.unsqueeze(1))

        # Cull per polygon and row, using stable ordering so antialiased painter order survives.
        def culled(all_verts, n_all):
            scr = _to_screen(all_verts.reshape(1, -1, 2).expand(B, n_all * 4, 2), cos, sin, zoom, trans).reshape(
                B, n_all, 4, 2
            )
            vis = _visible(scr)
            order = torch.argsort((~vis).to(torch.uint8), dim=1, stable=True)
            K = int(vis.sum(1).max().item())
            return scr, vis, order[:, :K], K

        gscr, gvis, gord, gK = culled(self.grass_v, self.grass_v.shape[0])
        gfx_v.append(gscr.gather(1, gord.view(B, gK, 1, 1).expand(B, gK, 4, 2)))
        gfx_c.append(const(self.bk["grass_color"]).unsqueeze(1).expand(B, gK, 3))
        gfx_on.append(gvis.gather(1, gord))

        rscr, rvis, rord, rK = culled(self.road_v, self.road_v.shape[0])
        visited = sty["track.visited"] > 0.5  # (B, n_tiles)
        base = self.road_c[rord]  # (B,rK,3) unvisited colour
        ti = self.tile_of[rord]  # (B,rK)
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

        # Indicators use post-flip coordinates; pass the coordinate frame to the kernel.
        speed = torch.sqrt(s[:, 3] ** 2 + s[:, 4] ** 2)
        steer = s[:, 8] - s[:, 2]  # revolute joint angle, not a state dimension
        for verts, col, gate in self._indicators(speed, s[:, 22:26], steer, s[:, 5], B):
            post_v.append(verts)
            post_c.append(const(col))
            post_on.append(gate)

        buf = self._raster(
            B, gfx_v, gfx_c, gfx_on, pre_v, pre_c, pre_on, post_v, post_c, post_on, sty["rendering.antialias"] > 0.5
        )

        # 6 score, then 7 the resample
        self._blit_text(buf, s[:, 26])
        x = F.shrink(buf.reshape(B * H, W, 3), self.sx)
        return F.shrink(x.reshape(B, H, 96 * 3), self.sy).view(B, 96, 96, 3)

    def _raster(self, B, gfx_v, gfx_c, gfx_on, pre_v, pre_c, pre_on, post_v, post_c, post_on, aa):
        """Pack polygon groups into kernel tables with output-frame bounding boxes.

        Grow bounds by one pixel to include the antialiased fringe."""
        dev = self.device
        VMAX = max([v.shape[1] for v in pre_v] + [4])
        Pg = sum(v.shape[1] for v in gfx_v)
        groups = [(torch.cat(gfx_v, 1), torch.cat(gfx_c, 1), torch.cat(gfx_on, 1))]
        for vs, cs, os_ in ((pre_v, pre_c, pre_on), (post_v, post_c, post_on)):
            if not vs:  # an empty group is legal; a bisect over the list uses it
                continue
            groups.append((torch.stack([_pad_verts(v, VMAX) for v in vs], 1), torch.stack(cs, 1), torch.stack(os_, 1)))
        nvs = [4] * Pg + [v.shape[1] for v in pre_v] + [4] * len(post_v)
        Ppost = Pg + len(pre_v)
        P = Ppost + len(post_v)

        vert = torch.cat([_pad_verts(g[0].reshape(B, -1, g[0].shape[2], 2), VMAX) for g in groups], 1)
        vert = torch.trunc(vert).to(torch.int32)  # (B,P,VMAX,2)
        col = torch.cat([g[1] for g in groups], 1).to(torch.int32)  # (B,P,3)
        gate = torch.cat([g[2] for g in groups], 1)  # (B,P)
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
        meta = torch.stack(
            [
                nv.expand(B, P),
                col[..., 0],
                col[..., 1],
                col[..., 2],
                torch.where(off, torch.ones_like(x0), x0),
                torch.where(off, torch.ones_like(y0), y0),
                torch.where(off, torch.zeros_like(x1), x1),
                torch.where(off, torch.zeros_like(y1), y1),
            ],
            -1,
        ).contiguous()

        edge = F.edge_table(
            vx[:, :Pg, :4].reshape(-1, 4).to(torch.int64), vy[:, :Pg, :4].reshape(-1, 4).to(torch.int64), H, W
        )
        return F.render_surface(
            vert.contiguous(),
            meta,
            edge.reshape(B, Pg, 4, F.EG_W),
            aa.to(torch.int32).reshape(B).contiguous(),
            H,
            W,
            Pg,
            Ppost,
        )

    def _stripe(self, px, py, ang, phase, B):
        """Build the wheel stripe quad and its source phase-dependent visibility mask."""
        a1, a2 = phase, phase + 1.2
        s1, s2 = torch.sin(a1), torch.sin(a2)
        c1 = torch.where(s1 > 0, torch.sign(torch.cos(a1)), torch.cos(a1))
        c2 = torch.where(s2 > 0, torch.sign(torch.cos(a2)), torch.cos(a2))
        z = torch.zeros_like(c1)
        wf, wr = self.bk["wheel_w"], self.bk["wheel_r"]
        loc = torch.stack(
            [
                torch.stack([z - wf, wr * c1], -1),
                torch.stack([z + wf, wr * c1], -1),
                torch.stack([z + wf, wr * c2], -1),
                torch.stack([z - wf, wr * c2], -1),
            ],
            1,
        )
        return _body_to_world(loc, px, py, ang), ~((s1 > 0) & (s2 > 0))

    def _indicators(self, speed, omegas, steer, gyro, B):
        """Build indicator polygons in post-flip coordinates with source visibility thresholds."""
        dev = self.device
        sx, hh = W / 40.0, H / 40.0
        one = torch.ones(B, dtype=torch.bool, device=dev)

        def vert(place, val):
            top, bot = H - (hh + hh * val), torch.full_like(val, H - hh)
            f = lambda c: torch.full_like(val, c)  # noqa: E731
            return torch.stack(
                [
                    torch.stack([f(place * sx), top], -1),
                    torch.stack([f((place + 1) * sx), top], -1),
                    torch.stack([f((place + 1) * sx), bot], -1),
                    torch.stack([f(place * sx), bot], -1),
                ],
                1,
            )

        def horiz(place, val):
            x0, x1 = torch.full_like(val, place * sx), (place + val) * sx
            y0, y1 = torch.full_like(val, H - 4 * hh), torch.full_like(val, H - 2 * hh)
            return torch.stack(
                [
                    torch.stack([x0, y0], -1),
                    torch.stack([x1, y0], -1),
                    torch.stack([x1, y1], -1),
                    torch.stack([x0, y1], -1),
                ],
                1,
            )

        panel = (
            torch.tensor([[W, H], [W, H - 5 * hh], [0, H - 5 * hh], [0, H]], dtype=torch.float64, device=dev)
            .view(1, 4, 2)
            .expand(B, 4, 2)
        )
        out = [(panel, (0, 0, 0), one), (vert(5, 0.02 * speed), (255, 255, 255), speed.abs() > 1e-4)]
        for k, place, col in ((0, 7, (0, 0, 255)), (1, 8, (0, 0, 255)), (2, 9, (51, 0, 255)), (3, 10, (51, 0, 255))):
            out.append((vert(place, 0.01 * omegas[:, k]), col, omegas[:, k].abs() > 1e-4))
        out.append((horiz(20, -10.0 * steer), (0, 255, 0), steer.abs() > 1e-4))
        out.append((horiz(30, -0.8 * gyro), (255, 0, 0), gyro.abs() > 1e-4))
        return out

    def _blit_text(self, buf, reward):
        """Draw score text centered at (60, 750), truncating and clamping to the atlas range."""
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
        g = self.atlas[
            idx.view(-1, 1, 1),
            torch.arange(th, device=dev).view(1, -1, 1),
            src.clamp(0, self.atlas.shape[2] - 1).view(-1, 1, win_w),
        ]
        dst = buf[:, y0 : y0 + th, win_lo : win_lo + win_w, :]
        dst.copy_(torch.where(ok.view(-1, 1, win_w, 1), g.unsqueeze(-1).expand(-1, -1, -1, 3), dst))

    @staticmethod
    def reference_env(size: int | None = None):
        """Reset the source to the baked track seed; its 96×96 observation ignores ``size``."""
        import os

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import gymnasium

        env = gymnasium.make("CarRacing-v3", render_mode="state_pixels").unwrapped
        env.reset(seed=BAKE_SEED)
        return env

    @staticmethod
    def reference_set_state(env, st) -> None:
        """Install float32 body/wheel poses and HUD values without stepping.

        Stepping would trigger contacts that repaint road tiles."""
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
            # Box2D stores poses in float32; compare against the float32 input after restoration.
            got = np.array(
                [
                    *env.car.hull.position,
                    env.car.hull.angle,
                    *env.car.hull.linearVelocity,
                    env.car.hull.angularVelocity,
                ],
                np.float64,
            )
            want = np.asarray(st[:6], np.float64)[[0, 1, 2, 3, 4, 5]]
            if np.abs(got - want).max() > 1e-5:
                raise AssertionError(f"the reference env did not reach the requested state: {got} vs {want}")
            j = float(env.car.wheels[0].joint.angle)
            if abs(j - (float(st[8]) - float(st[2]))) > 1e-4:
                raise AssertionError(
                    f"wheels[0].joint.angle is {j}, not wheel0.angle - hull.angle "
                    f"({float(st[8]) - float(st[2])}) -- the steering bar would be drawn from a "
                    f"quantity this state cannot reproduce"
                )
            out.append(env.render().copy())
        return np.stack(out)
