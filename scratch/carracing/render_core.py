"""GPU reimplementation of CarRacing's 1000x800 surface. Layer by layer, so each can be diffed alone.

Layers, in the reference's own order (``CarRacing._render``):

  1 background quad      gfxdraw.aapolygon + filled_polygon, clip=False
  2 400 grass quads      same, culled
  3 N road quads         same, culled, per-quad colour
  4 car                  pygame.draw.polygon  (4 wheels each +stripe, then 4 hull fixtures)
  - vertical flip
  5 indicators           pygame.draw.polygon, in POST-flip coordinates
  6 score text           opaque blit of a baked glyph string
  7 resample             the recovered smoothscale, 2^14 fixed point

Everything is float64 until the truncation to Sint16, per ``render/gfxdraw.py`` rule 1, except the
``pygame.draw.polygon`` crossing, which the reference computes in float32 and which therefore has to
be float32 here.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from stable_envs.render import gfxdraw as G

W, H = 1000, 800
SCALE = 6.0
ZOOM = 2.7
PLAYFIELD = 2000 / SCALE
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = max(GRASS_DIM, 40 / SCALE, 21 / SCALE) * math.sqrt(2) * ZOOM * SCALE
TWO_PI = 2.0 * math.pi
KMAX = 1024


# --------------------------------------------------------------------------- canvas
def canvas(gfx: G.Gfx, B: int, bg=(0, 0, 0)) -> G.StyleCanvas:
    """``render/gfxdraw.StyleCanvas`` already carries everything this env needs: a per-environment
    colour (road tiles differ by index AND by whether they have been driven over), a per-environment
    ``gate`` (the reference's cull is a per-environment decision, so it cannot be a python ``if``),
    and a per-environment ``aa`` gate. Nothing here re-implements it."""
    col = torch.zeros(B, 3, dtype=torch.float32, device=gfx.device)
    for c in range(3):
        col[:, c] = bg[c]
    return G.StyleCanvas(gfx, B, col)


def flip_in_place(cv) -> None:
    """The reference's ``pygame.transform.flip(surf, False, True)``, applied MID-render: the
    indicators and the score are drawn after it and in its coordinates."""
    cv.buf.copy_(torch.flip(cv.buf, dims=(1,)))


def poly(cv, verts, color, gate, bidx, kmax=None) -> None:
    """``_draw_colored_polygon``: ``aapolygon`` then ``filled_polygon`` on the same float64 vertices."""
    G.polygon(cv.use(color).gated(gate), verts, None, None, bidx, KMAX if kmax is None else kmax, True)


# --------------------------------------------------------------------------- transform
def rotate_rad(v: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``pygame.math.Vector2.rotate_rad``: ``[[cos,-sin],[sin,cos]]``, cos/sin supplied by the caller
    because the reference normalises the angle into [0, 2pi) with ``fmod`` FIRST, and sin/cos of the
    normalised angle is not bit-identical to sin/cos of the raw one."""
    return torch.stack([v[..., 0] * cos - v[..., 1] * sin,
                        v[..., 0] * sin + v[..., 1] * cos], -1)


def angle_cos_sin(angle: torch.Tensor):
    """``_vector2_rotate_helper``'s normalisation, then cos/sin. float64 throughout."""
    a = torch.fmod(angle, TWO_PI)
    a = torch.where(a < 0, a + TWO_PI, a)
    return torch.cos(a), torch.sin(a)


def view_params(hx, hy, hang, t):
    """``_render``'s camera: angle, zoom and translation, exactly as the reference computes them."""
    angle = -hang
    zoom = 0.1 * SCALE * torch.clamp(1 - t, min=0) + ZOOM * SCALE * torch.clamp(t, max=1)
    cos, sin = angle_cos_sin(angle)
    scroll = torch.stack([-hx * zoom, -hy * zoom], -1)
    tr = rotate_rad(scroll, cos, sin)
    return cos, sin, zoom, torch.stack([W / 2 + tr[..., 0], H / 4 + tr[..., 1]], -1)


def to_screen(verts, cos, sin, zoom, trans):
    """``(B,n,2)`` world -> screen.

    ``cos``/``sin`` are viewed as ``(B,1)`` and NOT ``(B,1,1)``: ``rotate_rad`` indexes ``v[..., 0]``,
    which has already dropped the coordinate axis, so a ``(B,1,1)`` factor broadcasts against
    ``(B,n)`` to ``(B,B,n)`` and silently renders B copies of the wrong geometry.
    """
    r = rotate_rad(verts, cos.view(-1, 1), sin.view(-1, 1))
    return r * zoom.view(-1, 1, 1) + trans.view(-1, 1, 2)


def visible(scr: torch.Tensor) -> torch.Tensor:
    """The reference's own cull: any vertex inside the window grown by ``MAX_SHAPE_DIM``."""
    x, y = scr[..., 0], scr[..., 1]
    return (((-MAX_SHAPE_DIM <= x) & (x <= W + MAX_SHAPE_DIM)
             & (-MAX_SHAPE_DIM <= y) & (y <= H + MAX_SHAPE_DIM)).any(-1))


# --------------------------------------------------------------------------- pygame.draw.polygon
def draw_poly_mask(scr: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """``pygame.draw.polygon``'s ``draw_fillpoly`` -> ``(B,H,W)`` bool. Measured 800/800 and 225/225.

    The load-bearing line is the parity of the per-ROW crossing counter, which walks edges in VERTEX
    order: even-numbered crossings are floored and odd ones ceiled, so a given geometric edge changes
    rounding when the number of lower-indexed active edges changes -- i.e. mid-edge, as ``y`` passes
    some other vertex. No per-edge rule reproduces it. ``polygon()`` also dispatches 3-vertex inputs
    to a different function; nothing CarRacing draws is a triangle (hull fixtures are 4/4/8/4,
    wheels, stripes and indicator quads are 4), which is asserted by the caller.
    """
    v = torch.trunc(scr)
    vx = v[..., 0].to(torch.int64)
    vy = v[..., 1].to(torch.int64)
    B, n = vx.shape
    ay, by = vy.roll(1, 1), vy                       # edge (i-1, i), i.e. vertex order
    ax, bx = vx.roll(1, 1), vx
    up = ay < by
    y1 = torch.where(up, ay, by).unsqueeze(-1)       # (B,n,1)
    y2 = torch.where(up, by, ay).unsqueeze(-1)
    x1 = torch.where(up, ax, bx).unsqueeze(-1)
    x2 = torch.where(up, bx, ax).unsqueeze(-1)
    live = (ay != by).unsqueeze(-1)                  # horizontal edges: trailing pass, not here
    maxy = vy.max(1, keepdim=True).values.unsqueeze(-1)
    miny = vy.min(1, keepdim=True).values.unsqueeze(-1)

    y = ys.view(1, 1, -1)
    act = live & (((y1 <= y) & (y < y2)) | ((y == maxy) & (y2 == maxy)))
    rank = torch.cumsum(act.to(torch.int32), dim=1) - act.to(torch.int32)
    num = ((y - y1) * (x2 - x1)).to(torch.float32)
    den = (y2 - y1).clamp_min(1).to(torch.float32)
    q = num / den                                    # float32, as the reference
    xc = torch.where(rank % 2 == 0, torch.floor(q), torch.ceil(q)).to(torch.int64) + x1

    BIG = 1 << 40
    xs_sorted = torch.where(act, xc, torch.full_like(xc, BIG)).sort(dim=1).values   # (B,n,H)
    cnt = act.sum(1)                                                               # (B,H)
    mask = torch.zeros((B, ys.numel(), xs.numel()), dtype=torch.bool, device=vx.device)
    xg = xs.view(1, 1, -1)
    for k in range(0, n - 1, 2):                     # consecutive PAIRS of sorted crossings
        lo = xs_sorted[:, k, :].transpose(0, 1) if False else xs_sorted[:, k, :]
        hi = xs_sorted[:, k + 1, :]
        ok = cnt >= k + 2
        mask |= ok.unsqueeze(-1) & (xg >= lo.unsqueeze(-1)) & (xg <= hi.unsqueeze(-1))

    # miny == maxy: the whole polygon is one row, drawn from min x to max x
    flat = (miny == maxy).squeeze(-1).squeeze(-1)                                  # (B,)
    if bool(flat.any()):
        row = (ys.view(1, -1) == vy.min(1, keepdim=True).values)                   # (B,H)
        lo = vx.min(1).values.view(-1, 1, 1)
        hi = vx.max(1).values.view(-1, 1, 1)
        mask |= (flat.view(-1, 1, 1) & row.unsqueeze(-1)
                 & (xg >= lo) & (xg <= hi))

    # trailing pass: horizontal edges strictly between miny and maxy get one span each
    hy = vy
    hprev = vy.roll(1, 1)
    hx, hxp = vx, vx.roll(1, 1)
    for e in range(n):
        sel = (hprev[:, e] == hy[:, e]) & (hy[:, e] > miny.squeeze(-1).squeeze(-1)) \
            & (hy[:, e] < maxy.squeeze(-1).squeeze(-1))
        if not bool(sel.any()):
            continue
        row = (ys.view(1, -1) == hy[:, e:e + 1])
        lo = torch.minimum(hx[:, e], hxp[:, e]).view(-1, 1, 1)
        hi = torch.maximum(hx[:, e], hxp[:, e]).view(-1, 1, 1)
        mask |= sel.view(-1, 1, 1) & row.unsqueeze(-1) & (xg >= lo) & (xg <= hi)
    return mask


# --------------------------------------------------------------------------- car + indicators
def draw_fill(cv, scr, color, gate, window=None) -> None:
    """One ``pygame.draw.polygon``. No anti-aliasing: this rasterizer has none."""
    mask = draw_poly_mask(scr, cv.ys, cv.xs)
    cv.use(color).gated(gate).paint(mask, window=window)


def car_local(u):
    """The car's LOCAL fixture vertices, read off Box2D rather than off the source.

    Box2D's ``b2PolygonShape::Set`` recomputes the convex hull and may rotate the vertex order, and
    vertex order is load-bearing for ``draw_fillpoly``'s parity counter -- so these are baked from a
    live body, not transcribed from ``HULL_POLY1..4``. They are constants of the env (checked: every
    wheel has the same four, and the four hull fixtures never change), which is what makes the car a
    function of the four body poses.
    """
    wheels = [[(float(v[0]), float(v[1])) for v in w.fixtures[0].shape.vertices] for w in u.car.wheels]
    hull = [[(float(v[0]), float(v[1])) for v in f.shape.vertices] for f in u.car.hull.fixtures]
    return wheels, hull


def body_to_world(local, px, py, ang):
    """``f.body.transform * v`` -- Box2D's transform is about the body ORIGIN, not its centre of mass.

    ``local`` is ``(n,2)`` shared or ``(B,n,2)``; ``px``/``py``/``ang`` are ``(B,)``.
    """
    c, s = torch.cos(ang).view(-1, 1), torch.sin(ang).view(-1, 1)
    x, y = local[..., 0], local[..., 1]
    return torch.stack([px.view(-1, 1) + x * c - y * s, py.view(-1, 1) + x * s + y * c], -1)


def wheel_stripe(px, py, ang, phase, wheel_w, wheel_r):
    """``Car.draw``'s white stripe: a quad whose height is ``cos(phase)`` and ``cos(phase+1.2)``,
    with each replaced by its SIGN when the corresponding sine is positive, and the whole thing
    skipped when both sines are. Returns ``(verts, gate)``."""
    a1, a2 = phase, phase + 1.2
    s1, s2 = torch.sin(a1), torch.sin(a2)
    c1 = torch.where(s1 > 0, torch.sign(torch.cos(a1)), torch.cos(a1))
    c2 = torch.where(s2 > 0, torch.sign(torch.cos(a2)), torch.cos(a2))
    gate = ~((s1 > 0) & (s2 > 0))
    z = torch.zeros_like(c1)
    loc = torch.stack([
        torch.stack([z - wheel_w, wheel_r * c1], -1),
        torch.stack([z + wheel_w, wheel_r * c1], -1),
        torch.stack([z + wheel_w, wheel_r * c2], -1),
        torch.stack([z - wheel_w, wheel_r * c2], -1)], 1)          # (B,4,2)
    return body_to_world(loc, px, py, ang), gate


def indicator_polys(true_speed, omegas, steer, gyro, dev):
    """``_render_indicators``, in POST-flip coordinates. Returns [(verts, colour, gate)] in order."""
    s, h = W / 40.0, H / 40.0
    B = true_speed.shape[0]
    one = torch.ones(B, dtype=torch.bool, device=dev)

    def vert_ind(place, val):
        top = H - (h + h * val)
        bot = torch.full_like(val, H - h)
        return torch.stack([
            torch.stack([torch.full_like(val, place * s), top], -1),
            torch.stack([torch.full_like(val, (place + 1) * s), top], -1),
            torch.stack([torch.full_like(val, (place + 1) * s), bot], -1),
            torch.stack([torch.full_like(val, place * s), bot], -1)], 1)

    def horiz_ind(place, val):
        x0 = torch.full_like(val, place * s)
        x1 = (place + val) * s
        y0 = torch.full_like(val, H - 4 * h)
        y1 = torch.full_like(val, H - 2 * h)
        return torch.stack([torch.stack([x0, y0], -1), torch.stack([x1, y0], -1),
                            torch.stack([x1, y1], -1), torch.stack([x0, y1], -1)], 1)

    panel = torch.tensor([[W, H], [W, H - 5 * h], [0, H - 5 * h], [0, H]],
                         dtype=torch.float64, device=dev).view(1, 4, 2).expand(B, 4, 2)
    out = [(panel, (0, 0, 0), one)]
    out.append((vert_ind(5, 0.02 * true_speed), (255, 255, 255), true_speed.abs() > 1e-4))
    for k, place, col in ((0, 7, (0, 0, 255)), (1, 8, (0, 0, 255)),
                          (2, 9, (51, 0, 255)), (3, 10, (51, 0, 255))):
        out.append((vert_ind(place, 0.01 * omegas[:, k]), col, omegas[:, k].abs() > 1e-4))
    out.append((horiz_ind(20, -10.0 * steer), (0, 255, 0), steer.abs() > 1e-4))
    out.append((horiz_ind(30, -0.8 * gyro), (255, 0, 0), gyro.abs() > 1e-4))
    return out


# --------------------------------------------------------------------------- score text
def bake_text(lo=-100, hi=1000):
    """Every string ``"%04i" % reward`` can produce, as one padded grey atlas plus its widths.

    Kerning makes the rendered width depend on the digits (78..102 px measured), so the atlas is keyed
    by the whole STRING and the blit x follows from ``text_rect.center = (60, 750)``. The glyphs are
    grey (R==G==B, 118 levels) so one channel is stored.
    """
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.font.init()
    f = pygame.font.Font(pygame.font.get_default_font(), 42)
    strings = ["%04i" % v for v in range(lo, hi + 1)]
    surfs = [f.render(t, True, (255, 255, 255), (0, 0, 0)) for t in strings]
    ws = [s.get_size()[0] for s in surfs]
    hs = {s.get_size()[1] for s in surfs}
    assert len(hs) == 1, f"score text is not a fixed height: {hs}"
    th = hs.pop()
    atlas = np.zeros((len(strings), th, max(ws)), np.uint8)
    for i, s in enumerate(surfs):
        buf = pygame.Surface(s.get_size(), depth=32)
        buf.fill((0, 0, 0))
        buf.blit(s, (0, 0))
        a = np.transpose(np.array(pygame.surfarray.pixels3d(buf)), (1, 0, 2))
        atlas[i, :, :a.shape[1]] = a[..., 0]
    return atlas, np.array(ws, np.int64), th, lo


def blit_text(cv, reward, atlas, widths, th, lo):
    """The opaque blit of the score. ``reward`` is ``(B,)`` float64; the string index is
    ``int(reward)`` -- ``%i`` truncates a float toward zero -- clamped to the baked range."""
    B = reward.shape[0]
    dev = cv.buf.device
    idx = (torch.trunc(reward).to(torch.int64) - lo).clamp(0, atlas.shape[0] - 1)
    w = widths[idx]                                          # (B,)
    y0 = int(750 - th // 2)
    x0 = 60 - torch.div(w, 2, rounding_mode="floor")         # Rect.center: x = cx - w//2
    # one fixed window covers every string: the widest is 102 px and every blit is centred on x=60,
    # so the leftmost column any of them can touch is 60 - 102//2 and the window is 102 wide.
    win_w = int(widths.max())
    win_lo = 60 - win_w // 2
    j = torch.arange(win_w, device=dev).view(1, -1)
    src = win_lo + j - x0.view(-1, 1)                        # (B,win_w)
    ok = (src >= 0) & (src < w.view(-1, 1))
    g = atlas[idx.view(-1, 1, 1), torch.arange(th, device=dev).view(1, -1, 1),
              src.clamp(0, atlas.shape[2] - 1).view(-1, 1, win_w)]        # (B,th,win_w)
    dst = cv.buf[:, y0:y0 + th, win_lo:win_lo + win_w, :]
    dst.copy_(torch.where(ok.view(-1, 1, win_w, 1), g.unsqueeze(-1).expand(-1, -1, -1, 3), dst))
