"""Batched torch rasterizer for gymnasium classic control: state ``(B,k)`` -> frames ``(B,H,W,3)``.

Not an approximation of the pygame renderer -- a transcription of it. Everything SDL_gfx does to a
classic-control frame is integer arithmetic on truncated-to-int vertices, so the GPU can do exactly
the same arithmetic and "do they agree" has an exact answer (zero differing pixels) rather than an
anti-aliasing tolerance. Every rule was established by pixel-diffing pygame (``probe_*.py``) and
re-checked on CPU (``verify_ref.py``) before any of this was written.

Three things make it parallel:

  * ``filled_polygon`` is a scanline fill, so a frame costs ``O(H * edges)`` span arithmetic plus one
    compare per pixel -- cheaper than the per-edge signed-distance test an SDF rasterizer uses, and
    it is the exact rule rather than a supersampled approximation of it.
  * ``filled_circle`` is a per-row half-width table indexed by ``|dy|``. The table is midpoint-circle
    output, NOT ``floor(sqrt(r^2-dy^2))``, which disagrees for every radius >= 2.
  * ``aapolygon``'s Wu line looks sequential (a 32-bit error accumulator) but has a closed form,
    ``wraps = (k*E) >> 16`` and ``weight = ((k*E) & 0xFFFF) >> 8``, so all of an edge's pixels are
    emitted at once. Within one edge the major axis advances every step, so no two writes collide
    and the scatter needs no ordering.

Two details that are easy to get wrong and expensive to get wrong:

  * geometry is float64. The output depends on ``trunc()`` of the vertex coordinates, so a float32
    pipeline lands on the wrong integer whenever a vertex is within ~1e-5 of one, and the frame is
    off by a whole row.
  * no host syncs in the inner loop: out-of-bounds AA writes are routed to a trash slot instead of
    being filtered with boolean indexing, and every step-count bound is a static constant, because
    ``.item()`` on a batched renderer costs more than the rasterization.
"""
from __future__ import annotations

import math

import torch

M32 = (1 << 32) - 1
SOLID = 256
SHIFT = (16, 8, 0)          # plain pygame.Surface channel masks are (0xFF0000, 0xFF00, 0xFF)


# ------------------------------------------------------------------ primitives
def circle_hw(r: int):
    """Row half-widths of ``gfxdraw.filled_circle``, indexed by |dy|.

    ``filledCircleColor`` walks the midpoint circle but paints each span only the FIRST time its
    octant coordinate changes (the ``ocx``/``ocy`` guards). Dropping those guards -- taking the max
    over every iteration instead -- widens the last row from 0 to 1 and paints a 5-pixel spur across
    the top and bottom of every disc. So does ``floor(sqrt(r^2-dy^2))``, which disagrees with
    SDL_gfx for every radius >= 2.
    """
    hw = [-1] * (r + 1)

    def span(dy, half):
        if half > hw[dy]:
            hw[dy] = half

    cx, cy = 0, r
    ocx = ocy = -1
    df, d_e, d_se = 1 - r, 3, -2 * r + 5
    while True:
        if ocy != cy:
            span(cy, cx)
            ocy = cy
        if ocx != cx:
            if cx != cy:
                span(cx, cy)
            ocx = cx
        if df < 0:
            df, d_e, d_se = df + d_e, d_e + 2, d_se + 2
        else:
            df, d_e, d_se = df + d_se, d_e + 2, d_se + 4
            cy -= 1
        cx += 1
        if cx > cy:
            break
    return hw


def _round16(v):
    return (v >> 16) + ((v & 32768) >> 15)


def fill_polygon_mask(vx, vy, ys, xs, W):
    """``gfxdraw.filled_polygon`` for a batch of CONVEX polygons -> ``(B,H,W)`` bool.

    Per scanline the crossing of every non-horizontal edge is evaluated in 16.16 fixed point,
    ``X = (65536*(y-y1))//(y2-y1) * (x2-x1) + 65536*x1``, an edge counting when ``y1 <= y < y2``
    plus a special case that reopens the bottom edges on the last scanline. Convexity means exactly
    two crossings, so the span is ``min``/``max`` instead of a sort.
    """
    B, n = vx.shape
    ay, by = vy.roll(1, 1), vy
    ax, bx = vx.roll(1, 1), vx
    swap = ay > by
    ylo = torch.where(swap, by, ay).unsqueeze(-1)                 # (B,n,1)
    yhi = torch.where(swap, ay, by).unsqueeze(-1)
    xlo = torch.where(swap, bx, ax).unsqueeze(-1)
    xhi = torch.where(swap, ax, bx).unsqueeze(-1)
    live = (ay != by).unsqueeze(-1)
    maxy = vy.max(1, keepdim=True).values.unsqueeze(-1)           # (B,1,1)

    y = ys.view(1, 1, -1)
    act = live & (((y >= ylo) & (y < yhi)) | ((y == maxy) & (y > ylo) & (y <= yhi)))
    den = (yhi - ylo).clamp_min(1)
    X = torch.div(65536 * (y - ylo), den, rounding_mode="floor") * (xhi - xlo) + 65536 * xlo
    BIG = 1 << 40
    xmin = torch.where(act, X, BIG).amin(1)                       # (B,H)
    xmax = torch.where(act, X, -BIG).amax(1)
    ok = act.sum(1) >= 2
    xa = _round16(xmin + 1).clamp_min(0)
    xb = _round16(xmax - 1).clamp_max(W - 1)
    x = xs.view(1, 1, -1)
    return ok.unsqueeze(-1) & (x >= xa.unsqueeze(-1)) & (x <= xb.unsqueeze(-1))


def fill_circle_mask(cx, cy, r, ys, xs, hw_t):
    """``gfxdraw.filled_circle``. ``cx``/``cy`` are ``(B,)`` int64, ``r`` a python int."""
    dy = (ys.view(1, -1) - cy.view(-1, 1)).abs()                  # (B,H)
    h = torch.where(dy <= r, hw_t[dy.clamp(0, r)], torch.full_like(dy, -1))
    dx = (xs.view(1, 1, -1) - cx.view(-1, 1, 1)).abs()            # (B,1,W)
    return dx <= h.unsqueeze(-1)


def gfx_blend(src, dst, w, sh):
    """``_putPixelAlpha`` on SDL's masked channel word. ``src``/``dst`` (...,3) int64, ``w`` (...,1).

    The blend runs on the channel *in its packed bit position*, so the truncating ``>>8`` lands below
    the channel for green and blue, and the unsigned wrap turns it into a ceiling whenever the source
    is darker than the destination. Weight 255 maps to alpha ``(255*255)>>8 = 254`` and therefore
    still blends -- only ``SOLID`` writes the colour outright, which is why an ``aapolygon`` edge
    never quite paints its own colour.
    """
    a = (255 * w) >> 8
    S = src << sh
    D = dst << sh
    v = ((((S - D) & M32) * a) & M32) >> 8
    out = ((D + v) & (255 << sh)) >> sh
    return torch.where(w == SOLID, src.expand_as(out), out)


class Canvas:
    """A batch of RGB surfaces. One trailing trash pixel absorbs every out-of-bounds AA write, so
    clipping costs an index clamp instead of a boolean-mask host sync."""

    def __init__(self, B, H, W, device, sh, ys, xs, bg=(255, 255, 255)):
        self.B, self.H, self.W = B, H, W
        self.device = device
        self.flat = torch.empty((B * H * W + 1, 3), dtype=torch.uint8, device=device)
        self.buf = self.flat[:-1].view(B, H, W, 3)
        for c in range(3):                      # scalar fills: no host->device copy, so the whole
            self.buf[..., c].fill_(bg[c])       # render is CUDA-graph capturable
        self.ys, self.xs, self.sh = ys, xs, sh
        self.trash = B * H * W

    def paint(self, mask, color):
        for c in range(3):
            self.buf[..., c].masked_fill_(mask, color[c])

    def blend_at(self, b, y, x, color_t, w, valid):
        """Blend ``color_t`` (1,3) int64 at (b,y,x); ``valid`` False routes to the trash slot."""
        ok = valid & (y >= 0) & (y < self.H) & (x >= 0) & (x < self.W)
        flat = torch.where(ok, (b * self.H + y.clamp(0, self.H - 1)) * self.W
                           + x.clamp(0, self.W - 1), self.trash)
        dst = self.flat[flat].to(torch.int64)
        self.flat[flat] = gfx_blend(color_t, dst, w.view(-1, 1), self.sh).to(torch.uint8)

    def rgb(self):
        return torch.flip(self.buf, dims=(1,))


def _clip_encode(x, y, W, H):
    z = torch.zeros_like(x)
    c = torch.where(x < 0, z + 1, torch.where(x > W - 1, z + 2, z))
    return c | torch.where(y < 0, z + 8, torch.where(y > H - 1, z + 4, z))


def clip_line(x1, y1, x2, y2, W, H, iters=6):
    """SDL_gfx ``clipLine``: Cohen-Sutherland on Sint16 with a float32 slope, batched.

    ``aalineColorInt`` runs this BEFORE the Wu walk, so an edge that leaves the surface restarts its
    error accumulator from the clipped endpoint and lays down a completely different fringe than an
    unclipped walk would. Skipping it costs ~150 px a frame on CartPole past |theta| ~ 2.4 and on
    MountainCar near the goal, where the car hangs off the right edge. Note the swap: when the first
    point is already inside, the endpoints are exchanged and the exchange survives into the walk,
    where it decides which end gets the solid initial pixel. Cohen-Sutherland converges in at most
    four clips; the loop is run to a fixed bound so there is no host sync.
    """
    draw = torch.zeros_like(x1, dtype=torch.bool)
    done = torch.zeros_like(draw)
    for _ in range(iters):
        c1 = _clip_encode(x1, y1, W, H)
        c2 = _clip_encode(x2, y2, W, H)
        acc = ((c1 | c2) == 0) & ~done
        draw = draw | acc
        done = done | acc | (((c1 & c2) != 0) & ~done)
        act = ~done
        sw = act & (c1 == 0)
        x1, x2 = torch.where(sw, x2, x1), torch.where(sw, x1, x2)
        y1, y2 = torch.where(sw, y2, y1), torch.where(sw, y1, y2)
        c1 = torch.where(sw, c2, c1)

        dxne = x2 != x1
        den = (x2 - x1).to(torch.float32)
        m = torch.where(dxne, (y2 - y1).to(torch.float32) / torch.where(dxne, den, den + 1),
                        torch.ones_like(den))
        msafe = torch.where(m == 0, torch.ones_like(m), m)
        L = act & ((c1 & 1) != 0)
        Rt = act & ~L & ((c1 & 2) != 0)
        Bt = act & ~L & ~Rt & ((c1 & 4) != 0)
        Tp = act & ~L & ~Rt & ~Bt & ((c1 & 8) != 0)
        dyL = torch.trunc((0 - x1).to(torch.float32) * m).to(torch.int64)
        dyR = torch.trunc((W - 1 - x1).to(torch.float32) * m).to(torch.int64)
        dxB = torch.trunc((H - 1 - y1).to(torch.float32) / msafe).to(torch.int64)
        dxT = torch.trunc((0 - y1).to(torch.float32) / msafe).to(torch.int64)
        ny = torch.where(L, y1 + dyL, torch.where(Rt, y1 + dyR, torch.where(
            Bt, torch.full_like(y1, H - 1), torch.where(Tp, torch.zeros_like(y1), y1))))
        nx = torch.where(L, torch.zeros_like(x1), torch.where(
            Rt, torch.full_like(x1, W - 1), torch.where(
                Bt & dxne, x1 + dxB, torch.where(Tp & dxne, x1 + dxT, x1))))
        x1, y1 = nx, ny
    return draw, x1, y1, x2, y2


def _aaline(cv: Canvas, x1, y1, x2, y2, color_t, kmax, bidx):
    """``_aalineColor(..., draw_endpoint=0)``: initial solid pixel, then the Wu pairs in closed form."""
    dev = x1.device
    B = x1.shape[0]
    draw, x1, y1, x2, y2 = clip_line(x1, y1, x2, y2, cv.W, cv.H)
    cv.blend_at(bidx, y1, x1, color_t,
                torch.full((B,), SOLID, dtype=torch.int64, device=dev), draw)

    flip = y1 > y2
    xx0 = torch.where(flip, x2, x1)
    yy0 = torch.where(flip, y2, y1)
    dxs = torch.where(flip, x1, x2) - xx0
    dy = torch.where(flip, y1, y2) - yy0
    xdir = torch.where(dxs >= 0, 1, -1)
    dx = dxs.abs()

    vert = dx == 0
    horiz = (dy == 0) & ~vert
    k = torch.arange(kmax + 1, device=dev).view(1, -1)
    bb = bidx.view(-1, 1).expand(B, kmax + 1).reshape(-1)
    solid = torch.full((B * (kmax + 1),), SOLID, dtype=torch.int64, device=dev)
    # vertical edge -> solid vline; horizontal edge -> solid hline (SDL_gfx short-circuits, no AA)
    dw = draw.view(-1, 1)
    cv.blend_at(bb, (yy0.view(-1, 1) + k).reshape(-1),
                xx0.view(-1, 1).expand(B, kmax + 1).reshape(-1), color_t, solid,
                (dw & vert.view(-1, 1) & (dy > 0).view(-1, 1) & (k <= dy.view(-1, 1))).reshape(-1))
    lo = torch.minimum(xx0, xx0 + xdir * dx)
    cv.blend_at(bb, yy0.view(-1, 1).expand(B, kmax + 1).reshape(-1),
                (lo.view(-1, 1) + k).reshape(-1), color_t, solid,
                (dw & horiz.view(-1, 1) & (k <= dx.view(-1, 1))).reshape(-1))

    gen = ~vert & ~horiz
    ymaj = dy > dx
    major = torch.where(ymaj, dy, dx)
    minor = torch.where(ymaj, dx, dy)
    E = torch.div(minor << 16, major.clamp_min(1), rounding_mode="floor")
    kk = k[:, 1:]                                                        # k = 1 .. kmax
    acc = kk * E.view(-1, 1)
    wraps = acc >> 16
    wgt = (acc & 0xFFFF) >> 8
    live = (dw & gen.view(-1, 1) & (kk < major.view(-1, 1))).reshape(-1)
    ym = ymaj.view(-1, 1)
    base_x = xx0.view(-1, 1)
    base_y = yy0.view(-1, 1)
    d = xdir.view(-1, 1)
    xa = torch.where(ym, base_x + d * wraps, base_x + d * kk)
    ya = torch.where(ym, base_y + kk, base_y + wraps)
    xb = torch.where(ym, base_x + d * (wraps + 1), xa)
    yb = torch.where(ym, ya, ya + 1)
    bk = bidx.view(-1, 1).expand_as(xa).reshape(-1)
    cv.blend_at(bk, ya.reshape(-1), xa.reshape(-1), color_t, (255 - wgt).reshape(-1), live)
    cv.blend_at(bk, yb.reshape(-1), xb.reshape(-1), color_t, wgt.reshape(-1), live)


def draw_polygon(cv: Canvas, verts, color, ct, bidx, kmax, aa=True):
    """The gymnasium idiom: ``aapolygon`` then ``filled_polygon``. ``verts`` is (B,n,2) float64."""
    vi = torch.trunc(verts).to(torch.int64)
    vx, vy = vi[..., 0], vi[..., 1]
    if aa:
        n = vx.shape[1]
        for e in range(n):
            _aaline(cv, vx[:, e], vy[:, e], vx[:, (e + 1) % n], vy[:, (e + 1) % n], ct, kmax, bidx)
    cv.paint(fill_polygon_mask(vx, vy, cv.ys, cv.xs, cv.W), color)


def draw_circle(cv: Canvas, cx, cy, r, color, ct, bidx, hw_t, stencil=None, aa=True):
    """``aacircle`` (fringe stencil baked per radius/colour) then ``filled_circle``.

    ``stencil`` is a list of per-pass (dy, dx, w) tensors: ``aaellipseColor`` writes a handful of
    pixels twice, and the two writes have to land in order, so they go in separate passes.
    """
    if aa and stencil is not None:
        B = cx.shape[0]
        for st in stencil:
            S = st.shape[0]
            if not S:
                continue
            bb = bidx.view(-1, 1).expand(B, S).reshape(-1)
            cv.blend_at(bb, (cy.view(-1, 1) + st[:, 0].view(1, -1)).reshape(-1),
                        (cx.view(-1, 1) + st[:, 1].view(1, -1)).reshape(-1), ct,
                        st[:, 2].view(1, -1).expand(B, S).reshape(-1),
                        torch.ones(B * S, dtype=torch.bool, device=cv.device))
    cv.paint(fill_circle_mask(cx, cy, r, cv.ys, cv.xs, hw_t), color)


# ------------------------------------------------------------------ envs
def _rot(v, a):
    """``pygame.math.Vector2.rotate_rad``: [[cos,-sin],[sin,cos]] in a y-down raster."""
    c, s = torch.cos(a), torch.sin(a)
    return torch.stack([v[..., 0] * c - v[..., 1] * s, v[..., 0] * s + v[..., 1] * c], -1)


class _Base:
    H = W = 0
    STATE_DIM = 0

    def __init__(self, device="cuda", aa=True):
        self.device = torch.device(device)
        self.aa = aa
        self._hw = {}
        self._st = {}
        self._k = {}
        # every constant the render needs, materialised once. A host->device copy inside render()
        # is not just slow, it makes the whole thing un-capturable as a CUDA graph.
        self.sh = torch.tensor(SHIFT, dtype=torch.int64, device=self.device)
        self.ys = torch.arange(self.H, device=self.device)
        self.xs = torch.arange(self.W, device=self.device)
        self._bidx = {}

    def _canvas(self, B):
        return Canvas(B, self.H, self.W, self.device, self.sh, self.ys, self.xs)

    def _bi(self, B):
        if B not in self._bidx:
            self._bidx[B] = torch.arange(B, device=self.device)
        return self._bidx[B]

    def _f64(self, vals):
        key = ("f64", tuple(map(tuple, vals)))
        if key not in self._k:
            self._k[key] = torch.tensor(vals, dtype=torch.float64, device=self.device)
        return self._k[key]

    def _col(self, c):
        key = ("col", c)
        if key not in self._k:
            self._k[key] = torch.tensor(c, dtype=torch.int64, device=self.device).view(1, 3)
        return self._k[key]

    def _hw_t(self, r):
        if r not in self._hw:
            self._hw[r] = torch.tensor(circle_hw(r), dtype=torch.int64, device=self.device)
        return self._hw[r]

    def _stencil(self, r, color):
        """``aacircle``'s fringe, recovered once per (radius, colour) by inverting the blend.

        SDL_gfx's ``aaellipseColor`` is not transcribed: its write set is a constant of the radius
        (verified -- the solid pixels are identical with and without it), so it is cheaper and safer
        to read the stencil off pygame at construction than to re-derive an ellipse walker.
        """
        key = (r, color)
        if key not in self._st:
            import ref_gfx
            st = ref_gfx.bake_aacircle(r, color)
            self._st[key] = [torch.tensor([[d[0], d[1], d[2]] for d in st if d[3] == p],
                                          dtype=torch.int64, device=self.device).view(-1, 3)
                             for p in (0, 1)]
        return self._st[key]

    def _prep(self, state, dim):
        st = torch.as_tensor(state, dtype=torch.float64, device=self.device)
        if st.ndim == 1:
            st = st.view(-1, 1) if dim == 1 else st.view(1, -1)
        return st


class CartPoleGPU(_Base):
    """state ``(B,2)`` = (x, theta) -> ``(B,400,600,3)`` uint8."""
    H, W, STATE_DIM = 400, 600, 2
    SCALE = 600 / 4.8
    POLELEN, CARTY, AXLEOFF = 125.0, 100.0, 7.5
    CART_C, POLE_C, AXLE_C = (0, 0, 0), (202, 152, 101), (129, 132, 203)

    @torch.no_grad()
    def render(self, state):
        st = self._prep(state, 2)
        B = st.shape[0]
        x, th = st[:, 0], st[:, 1]
        cv = self._canvas(B)
        bi = self._bi(B)
        cartx = x * self.SCALE + self.W / 2.0
        carty = torch.full_like(cartx, self.CARTY)
        ctr = torch.stack([cartx, carty], -1).view(B, 1, 2)

        loc = self._f64([(-25.0, -15.0), (-25.0, 15.0), (25.0, 15.0), (25.0, -15.0)])
        draw_polygon(cv, loc.view(1, 4, 2) + ctr, self.CART_C, self._col(self.CART_C), bi,
                     52, self.aa)

        ploc = self._f64([(-5.0, -5.0), (-5.0, self.POLELEN - 5.0),
                          (5.0, self.POLELEN - 5.0), (5.0, -5.0)]).view(1, 4, 2).expand(B, 4, 2)
        pole = _rot(ploc, (-th).view(B, 1)) + torch.stack(
            [cartx, carty + self.AXLEOFF], -1).view(B, 1, 2)
        draw_polygon(cv, pole, self.POLE_C, self._col(self.POLE_C), bi, 132, self.aa)

        cxi = torch.trunc(cartx).to(torch.int64)
        cyi = torch.full_like(cxi, int(self.CARTY + self.AXLEOFF))
        draw_circle(cv, cxi, cyi, 5, self.AXLE_C, self._col(self.AXLE_C), bi, self._hw_t(5),
                    self._stencil(5, self.AXLE_C), self.aa)
        cv.buf[:, int(self.CARTY), :, :] = 0                          # gfxdraw.hline, drawn last
        return cv.rgb()


class PendulumGPU(_Base):
    """state ``(B,1)`` = (theta,) -> ``(B,500,500,3)`` uint8. Torque sprite is a separate layer."""
    H = W = 500
    STATE_DIM = 1
    SCALE = 500 / 4.4
    ROD_C, AXLE_C = (204, 77, 77), (0, 0, 0)

    @torch.no_grad()
    def render(self, state):
        st = self._prep(state, 1)
        B = st.shape[0]
        th = st[:, 0]
        sc, off = self.SCALE, 250
        cv = self._canvas(B)
        bi = self._bi(B)
        rl, rw = 1.0 * sc, 0.2 * sc
        loc = self._f64([(0.0, -rw / 2), (0.0, rw / 2), (rl, rw / 2),
                         (rl, -rw / 2)]).view(1, 4, 2).expand(B, 4, 2)
        ang = (th + math.pi / 2).view(B, 1)
        draw_polygon(cv, _rot(loc, ang) + off, self.ROD_C, self._col(self.ROD_C), bi, 120, self.aa)

        r = int(rw / 2)
        ctr = torch.full((B,), off, dtype=torch.int64, device=self.device)
        draw_circle(cv, ctr, ctr, r, self.ROD_C, self._col(self.ROD_C), bi, self._hw_t(r),
                    self._stencil(r, self.ROD_C), self.aa)

        end = _rot(self._f64([(rl, 0.0)]).view(1, 1, 2).expand(B, 1, 2), ang)[:, 0, :] + off
        draw_circle(cv, torch.trunc(end[:, 0]).to(torch.int64),
                    torch.trunc(end[:, 1]).to(torch.int64), r, self.ROD_C, self._col(self.ROD_C),
                    bi, self._hw_t(r), self._stencil(r, self.ROD_C), self.aa)

        ra = int(0.05 * sc)
        draw_circle(cv, ctr, ctr, ra, self.AXLE_C, self._col(self.AXLE_C), bi, self._hw_t(ra),
                    self._stencil(ra, self.AXLE_C), self.aa)
        return cv.rgb()


class AcrobotGPU(_Base):
    """state ``(B,2)`` = (theta1, theta2) -> ``(B,500,500,3)`` uint8."""
    H = W = 500
    STATE_DIM = 2
    SCALE = 500 / 4.4
    LINK_C, JOINT_C = (0, 204, 204), (204, 204, 0)

    @torch.no_grad()
    def render(self, state):
        st = self._prep(state, 2)
        B = st.shape[0]
        t1, t2 = st[:, 0], st[:, 1]
        sc, off = self.SCALE, 250.0
        cv = self._canvas(B)
        bi = self._bi(B)
        cv.buf[:, int(1 * sc + off), :, :] = 0                        # ground, pygame.draw.line
        # the source builds xys = [[0,0], p1, p2][:, ::-1], so link 1 hangs off (p1_y, p1_x)
        jx = [torch.zeros_like(t1), torch.sin(t1) * sc]
        jy = [torch.zeros_like(t1), -torch.cos(t1) * sc]
        r = int(0.1 * sc)
        loc = self._f64([(0.0, -0.1 * sc), (0.0, 0.1 * sc), (sc, 0.1 * sc),
                         (sc, -0.1 * sc)]).view(1, 4, 2).expand(B, 4, 2)
        for i, th in enumerate([t1 - math.pi / 2, t1 + t2 - math.pi / 2]):
            ctr = torch.stack([jx[i] + off, jy[i] + off], -1).view(B, 1, 2)
            draw_polygon(cv, _rot(loc, th.view(B, 1)) + ctr, self.LINK_C, self._col(self.LINK_C),
                         bi, 120, self.aa)
            draw_circle(cv, torch.trunc(jx[i] + off).to(torch.int64),
                        torch.trunc(jy[i] + off).to(torch.int64), r, self.JOINT_C,
                        self._col(self.JOINT_C), bi, self._hw_t(r),
                        self._stencil(r, self.JOINT_C), self.aa)
        return cv.rgb()


class MountainCarGPU(_Base):
    """state ``(B,1)`` = (position,) -> ``(B,400,600,3)`` uint8, on a baked static terrain layer."""
    H, W, STATE_DIM = 400, 600, 1
    MINP, SCALE = -1.2, 600 / 1.8
    CAR_C, WHEEL_C, FLAG_C = (0, 0, 0), (128, 128, 128), (204, 204, 0)

    def __init__(self, device="cuda", aa=True, base=None):
        super().__init__(device, aa)
        self.base = torch.as_tensor(base if base is not None else bake_mountaincar_base(),
                                    dtype=torch.uint8, device=self.device)
        self.flagx = int((0.5 - self.MINP) * self.SCALE)
        self.flagy1 = int((math.sin(3 * 0.5) * 0.45 + 0.55) * self.SCALE)
        self.flagy2 = self.flagy1 + 50

    @torch.no_grad()
    def render(self, state):
        st = self._prep(state, 1)
        B = st.shape[0]
        pos = st[:, 0]
        cv = self._canvas(B)
        bi = self._bi(B)
        cv.buf[...] = self.base
        ang = torch.cos(3 * pos).view(B, 1)
        ctr = torch.stack([(pos - self.MINP) * self.SCALE,
                           10.0 + (torch.sin(3 * pos) * 0.45 + 0.55) * self.SCALE], -1).view(B, 1, 2)
        loc = self._f64([(-20.0, 0.0), (-20.0, 20.0), (20.0, 20.0),
                         (20.0, 0.0)]).view(1, 4, 2).expand(B, 4, 2)
        draw_polygon(cv, _rot(loc, ang) + ctr, self.CAR_C, self._col(self.CAR_C), bi, 46, self.aa)
        wl = self._f64([(10.0, 0.0), (-10.0, 0.0)]).view(1, 2, 2).expand(B, 2, 2)
        wr = _rot(wl, ang) + ctr
        for j in range(2):
            draw_circle(cv, torch.trunc(wr[:, j, 0]).to(torch.int64),
                        torch.trunc(wr[:, j, 1]).to(torch.int64), 8, self.WHEEL_C,
                        self._col(self.WHEEL_C), bi, self._hw_t(8),
                        self._stencil(8, self.WHEEL_C), self.aa)
        cv.buf[:, self.flagy1:self.flagy2 + 1, self.flagx, :] = 0
        tri = self._f64([(float(self.flagx), float(self.flagy2)),
                         (float(self.flagx), float(self.flagy2 - 10)),
                         (float(self.flagx + 25), float(self.flagy2 - 5))]
                        ).view(1, 3, 2).expand(B, 3, 2)
        draw_polygon(cv, tri, self.FLAG_C, self._col(self.FLAG_C), bi, 30, self.aa)
        return cv.rgb()


def bake_mountaincar_base():
    """The static terrain: 99 anti-aliased segments that never depend on the state.

    ``pygame.draw.aalines`` is a different (sub-pixel, float-endpoint) AA path from gfxdraw's, and it
    draws a constant of the environment, so it is rasterised once at construction -- exact by
    construction and free per frame.
    """
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import numpy as np
    import pygame
    pygame.init()
    surf = pygame.Surface((600, 400))
    surf.fill((255, 255, 255))
    xs = np.linspace(-1.2, 0.6, 100)
    ys = np.sin(3 * xs) * 0.45 + 0.55
    pygame.draw.aalines(surf, points=list(zip((xs + 1.2) * (600 / 1.8), ys * (600 / 1.8))),
                        closed=False, color=(0, 0, 0))
    return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2)).copy()
