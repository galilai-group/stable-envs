from __future__ import annotations

import numpy as np
import torch

from ..env import Env
from ..registry import register
from ..render import gfxdraw as G
from ..spaces import Dict, Discrete, RGBBox

SCALE = 30.0
VIEWPORT_W, VIEWPORT_H = 600, 400
#: ``reset``: 11 terrain chunks over a world 20.0 wide, so the x of every terrain vertex is a constant
CHUNKS = 11
WORLD_W, WORLD_H = VIEWPORT_W / SCALE, VIEWPORT_H / SCALE
CHUNK_X = tuple(WORLD_W / (CHUNKS - 1) * i for i in range(CHUNKS))
HELIPAD_X = (CHUNK_X[CHUNKS // 2 - 1], CHUNK_X[CHUNKS // 2 + 1])
HELIPAD_Y = WORLD_H / 4

# Pixel-space fixture vertices follow Box2D winding, which controls antialias edge order.
LANDER_VERTS = ((17.0, -10.0), (17.0, 0.0), (14.0, 17.0), (-14.0, 17.0), (-17.0, 0.0), (-17.0, -10.0))
LEG_VERTS = ((-2.0, -8.0), (2.0, -8.0), (2.0, 8.0), (-2.0, 8.0))

#: ``pygame.draw.circle(centre, 2.0)`` as ``(dy, dx)`` offsets from the TRUNCATED centre. Read off
#: pygame; the shape is a constant because every particle has the same radius. It is not symmetric.
PARTICLE_STENCIL = (
    (-2, -1),
    (-2, 0),
    (-1, -2),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -2),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
)
#: ``render()``'s own decrement, applied to every particle before anything is drawn
TTL_DECAY = 0.15
# Six slots suffice when rendering at least once per step: the seventh lifetime decay removes a particle.
# State extraction rejects longer lists rather than truncating them.
N_PARTICLES = 6

LANDER_COLOR = (128, 102, 230)
TERRAIN_COLOR = (0, 0, 0)
FLAG_COLOR = (204, 204, 0)
POLE_COLOR = (255, 255, 255)
BACKGROUND = (255, 255, 255)

#: state layout
N_TERRAIN = CHUNKS
BODY_OFF = N_TERRAIN  # lander, leg0, leg1: (x, y, angle) each, in world units
PART_OFF = BODY_OFF + 9  # N_PARTICLES x (x, y, ttl)
STATE_DIM = PART_OFF + 3 * N_PARTICLES


def _check_lunar_lander_works() -> None:
    """Check reset and rendering at import so unusable optional dependencies prevent registration."""
    import gymnasium

    try:
        env = gymnasium.make("LunarLander-v3", render_mode="rgb_array")
        env.reset(seed=0)
        env.unwrapped.render()
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            f"LunarLander-v3 cannot be reset and rendered: {type(e).__name__}: {e}. This env needs "
            f"BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the "
            f"reference renderer parity is measured against); `pip install swig` then "
            f'`pip install "gymnasium[box2d]"`.'
        ) from e


_check_lunar_lander_works()


class _AARecorder(G.Canvas):
    """Record ordered Wu fringe writes for static flag geometry."""

    def __init__(self, gfx: G.Gfx):
        super().__init__(1, gfx.H, gfx.W, gfx.device, gfx.sh, gfx.ys, gfx.xs)
        self.writes: list[tuple] = []

    def blend_at(self, b, y, x, color_t, w, valid) -> None:
        ok = valid & (y >= 0) & (y < self.H) & (x >= 0) & (x < self.W)
        if bool(ok.any()):
            # Expand scalar weights before applying the per-write boolean mask.
            self.writes.append((y[ok].clone(), x[ok].clone(), w.expand(ok.shape)[ok].clone()))


def _record_aa(gfx: G.Gfx, vi: torch.Tensor) -> list[tuple]:
    """Return the ordered fringe writes for a constant (1, V, 2) polygon."""
    rec = _AARecorder(gfx)
    n = vi.shape[1]
    bidx = torch.zeros(1, dtype=torch.long, device=gfx.device)
    for e in range(n):
        G.aaline(rec, vi[:, e, 0], vi[:, e, 1], vi[:, (e + 1) % n, 0], vi[:, (e + 1) % n, 1], None, 64, bidx)
    return rec.writes


@register
class LunarLander(Env):
    """Render LunarLander-v3 terrain, three body poses, and six exhaust particles.

    The 38-value render state stores particle lifetimes after the source renderer
    would decrement them. It is not a complete Box2D simulator snapshot.
    """

    family = "lunarlander"
    domain = "lunarlander"

    name = "lunar-lander"
    reference_id = "LunarLander-v3"
    default_size = VIEWPORT_H
    native = (VIEWPORT_H, VIEWPORT_W)
    state_dim = STATE_DIM
    state_names = (
        tuple(f"terrain[{i}]" for i in range(N_TERRAIN))
        + tuple(f"{b}_{f}" for b in ("lander", "leg0", "leg1") for f in ("x", "y", "angle"))
        + tuple(f"p{k}_{f}" for k in range(N_PARTICLES) for f in ("x", "y", "ttl"))
    )
    # Render geometry only; velocities and Box2D simulation state are not represented.
    state_layout = (
        (("terrain",), N_TERRAIN, 0),
        (("lander", "pose"), 3, BODY_OFF),
        (("leg0", "pose"), 3, BODY_OFF + 3),
        (("leg1", "pose"), 3, BODY_OFF + 6),
        (("particles",), 3 * N_PARTICLES, PART_OFF),
    )

    # The declared interior threshold excludes the unimplemented pygame outline edge behavior.
    parity_interior_min = 1.0

    @property
    def frame_shape(self) -> tuple[int, int]:
        """Return the fixed 400 × 600 frame shape."""
        return self.native

    def _close(self) -> None:
        """Release the cached sample pool."""
        self._pool = None

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        H, W = self.native
        self.gfx = G.Gfx(H, W, self.device)
        d = dict(device=self.device, dtype=torch.float64)
        self._chunk_px = torch.tensor([x * SCALE for x in CHUNK_X], **d)
        self._lander_v = torch.tensor(LANDER_VERTS, **d)
        self._leg_v = torch.tensor(LEG_VERTS, **d)
        self._stencil = torch.tensor(PARTICLE_STENCIL, dtype=torch.long, device=self.device)
        self._pool: np.ndarray | None = None
        # Flag coordinates are static, so cache their pixel locations and fringe weights.
        y1, y2 = int(HELIPAD_Y * SCALE), int(HELIPAD_Y * SCALE) + 50
        xa, xb = int(HELIPAD_X[0] * SCALE), int(HELIPAD_X[1] * SCALE)
        dx = xb - xa
        rows = torch.arange(y1, y2 + 1, device=self.device)  # pygame.draw.line lights both ends
        self._pole_px = torch.cat([rows * W + xa, rows * W + xb])
        tri = torch.tensor([[xa, y2], [xa, y2 - 10], [xa + 25, y2 - 5]], **d).view(1, 3, 2)
        vi = torch.trunc(tri).to(torch.int64)
        m = G.fill_polygon_mask(vi[..., 0], vi[..., 1], self.gfx.ys, self.gfx.xs, W)[0]
        px = m.reshape(-1).nonzero().reshape(-1)
        self._tri_px = torch.cat([px, px + dx])
        # Cache ordered fringe blends; flags are drawn repeatedly within a frame.
        self._tri_aa = [
            (torch.cat([y, y]), torch.cat([x, x + dx]), torch.cat([w, w])) for y, x, w in _record_aa(self.gfx, vi)
        ]

    def make_variation_space(self) -> Dict:
        """Return painted colors and the antialiasing switch.

        Exhaust color depends on particle lifetime. The unimplemented source outline
        rasterizer has no exposed color control.
        """
        return Dict(
            {
                "background": Dict({"color": RGBBox(init_value=BACKGROUND)}),
                "terrain": Dict({"color": RGBBox(init_value=TERRAIN_COLOR)}),
                "lander": Dict({"color": RGBBox(init_value=LANDER_COLOR)}),
                "flag": Dict({"color": RGBBox(init_value=FLAG_COLOR), "pole_color": RGBBox(init_value=POLE_COLOR)}),
                "rendering": Dict({"antialias": Discrete(2, init_value=1)}),
            }
        )

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
        """Render (..., 38) rows to (..., 400, 600, 3) uint8 on the selected device."""
        H, W = self.native
        want = int(self.size if size is None else size)
        if want != H:
            raise ValueError(
                f"LunarLander renders at its reference's native {H}x{W} and nothing else (got "
                f"size={want}). Every primitive is hardcoded in pixels -- a 34 px lander, a 2 px "
                f"exhaust particle, a 50 px flagpole -- so a resized frame has no reference to be "
                f"exact against; resample the output if you want one."
            )
        if supersample not in (None, 1):
            raise ValueError(
                "LunarLander: supersample is not available. Its geometry is truncated to integers "
                "before rasterisation and its anti-aliasing is Wu weights rather than coverage, so "
                "sampling twice would average a picture the reference never draws."
            )
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        # float64 from here: every coordinate is truncated to an integer before it is rasterised, so
        # float32 lands on the wrong integer whenever a vertex falls within ~1e-5 of one
        s = s.reshape(-1, self.state_dim).to(torch.float64)
        n = s.shape[0]
        sty = self.resolve_style(style, variation, batch=n)
        cv = G.StyleCanvas(self.gfx, n, sty["background.color"])
        cv.aa = sty["rendering.antialias"] > 0.5
        self._draw(cv, s, sty)
        img = cv.rgb(None if out is None else out.reshape(-1, H, W, 3))
        return img.view(*lead, H, W, 3) if lead else img

    def _draw(self, cv, s, sty) -> None:
        """Draw terrain, particles, bodies, and repeated flags in source order before flipping."""
        B = s.shape[0]
        bidx = self.gfx.batch_index(B)
        terrain = sty["terrain.color"]
        lander_c = sty["lander.color"]

        # Only the dead prefix is removed. A dead slot following a live particle is still drawn.
        ttl = s[:, PART_OFF + 2 :: 3]  # (B, N)
        alive = torch.cummax((ttl >= 0).to(torch.uint8), dim=1).values.bool()
        red = torch.clamp(torch.trunc(torch.clamp(0.15 + ttl, min=0.2) * 255), 0, 255)
        gb = torch.clamp(torch.trunc(torch.clamp(0.5 * ttl, min=0.2) * 255), 0, 255)
        pcol = torch.stack([red, gb, gb], -1)  # (B, N, 3)

        # 2. the terrain, as CHUNKS-1 quads from the ground line down to the bottom of the frame
        sy = s[:, :N_TERRAIN] * SCALE
        cx = self._chunk_px
        bottom = torch.full_like(sy[:, :1], float(VIEWPORT_H))
        cv.gated(None)
        cv.use(terrain)
        quads = torch.stack(
            [
                torch.stack(
                    [
                        torch.stack([cx[i].expand(B), sy[:, i]], -1),
                        torch.stack([cx[i + 1].expand(B), sy[:, i + 1]], -1),
                        torch.stack([cx[i + 1].expand(B), bottom[:, 0]], -1),
                        torch.stack([cx[i].expand(B), bottom[:, 0]], -1),
                    ],
                    1,
                )
                for i in range(CHUNKS - 1)
            ],
            1,
        )  # (B, 10, 4, 2)
        qi = torch.trunc(quads).to(torch.int64)
        for i in range(CHUNKS - 1):
            # Vertical terrain boundaries confine each fill to its own screen band.
            x0 = int(CHUNK_X[i] * SCALE)
            x1 = min(int(CHUNK_X[i + 1] * SCALE) + 1, VIEWPORT_W)
            cv.paint(
                G.fill_polygon_mask(qi[:, i, :, 0], qi[:, i, :, 1], self.gfx.ys, self.gfx.xs[x0:x1], VIEWPORT_W),
                window=(0, VIEWPORT_H, x0, x1),
            )
        self._aa_poly(
            cv, qi.reshape(B * (CHUNKS - 1), 4, 2), bidx.view(-1, 1).expand(B, CHUNKS - 1).reshape(-1), VIEWPORT_H
        )

        # 3. ``for obj in self.particles + self.drawlist`` -- and the flags INSIDE the fixture loop
        for k in range(N_PARTICLES):
            cv.gated(alive[:, k])
            self._particle(cv, s[:, PART_OFF + 3 * k], s[:, PART_OFF + 3 * k + 1], pcol[:, k])
            self._flags(cv, sty, bidx)
        cv.gated(None)
        for j, verts in enumerate((self._lander_v, self._leg_v, self._leg_v)):
            o = BODY_OFF + 3 * j
            self._body(cv, verts, s[:, o], s[:, o + 1], s[:, o + 2], lander_c, bidx)
            self._flags(cv, sty, bidx)

    def _particle(self, cv, x, y, color) -> None:
        """Paint the radius-two pygame circle stencil at a truncated pixel center."""
        cx = torch.trunc(x * SCALE).to(torch.long)
        cy = torch.trunc(y * SCALE).to(torch.long)
        H, W = self.native
        ys = cy.view(-1, 1) + self._stencil[:, 0].view(1, -1)
        xs = cx.view(-1, 1) + self._stencil[:, 1].view(1, -1)
        ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        if cv.gate is not None:
            ok = ok & cv.gate.view(-1, 1)
        B = x.shape[0]
        b = torch.arange(B, device=x.device).view(-1, 1).expand_as(ys)
        flat = torch.where(ok, (b * H + ys.clamp(0, H - 1)) * W + xs.clamp(0, W - 1), torch.full_like(ys, cv.trash))
        n = ys.shape[1]
        # The planar canvas keeps all color channels in one scatter.
        cv.flat[:, flat.reshape(-1)] = G.to_u8(color).t().reshape(3, B, 1).expand(3, B, n).reshape(3, -1)

    def _aa_poly(self, cv, vi, bidx, kmax) -> None:
        """Blend all edges of (N, V, 2) integer polygons in one Wu pass.

        Shared vertices lie over the same fill color, so concurrent same-color edge
        writes do not change the intended blend there.
        """
        N, V, _ = vi.shape
        a = vi.reshape(-1, 2)
        b = torch.roll(vi, -1, dims=1).reshape(-1, 2)
        G.aaline(cv, a[:, 0], a[:, 1], b[:, 0], b[:, 1], None, kmax, bidx.view(-1, 1).expand(N, V).reshape(-1))

    def _body(self, cv, verts, x, y, angle, color, bidx) -> None:
        """Fill and antialias one transformed rigid-body polygon.

        The additional pygame aalines outline is not reproduced; boundary pixels can differ.
        """
        c, sn = torch.cos(angle), torch.sin(angle)
        vx, vy = verts[:, 0].view(1, -1), verts[:, 1].view(1, -1)
        px = c.view(-1, 1) * vx - sn.view(-1, 1) * vy + (x * SCALE).view(-1, 1)
        py = sn.view(-1, 1) * vx + c.view(-1, 1) * vy + (y * SCALE).view(-1, 1)
        vi = torch.trunc(torch.stack([px, py], -1)).to(torch.int64)
        cv.use(color).paint(G.fill_polygon_mask(vi[..., 0], vi[..., 1], self.gfx.ys, self.gfx.xs, VIEWPORT_W))
        self._aa_poly(cv, vi, bidx, 64)

    def _flags(self, cv, sty, bidx) -> None:
        """Draw both static flags once per source fixture, preserving repeated fringe blends."""
        cv.use(sty["flag.pole_color"]).paint_at(self._pole_px)
        cv.use(sty["flag.color"]).paint_at(self._tri_px)
        B = cv.B
        b = bidx.view(-1, 1)
        one = torch.ones(1, dtype=torch.bool, device=cv.device)
        for y, x, w in self._tri_aa:
            n = y.numel()
            cv.blend_at(
                b.expand(B, n).reshape(-1),
                y.view(1, -1).expand(B, n).reshape(-1),
                x.view(1, -1).expand(B, n).reshape(-1),
                None,
                w.view(1, -1).expand(B, n).reshape(-1),
                one.expand(B * n),
            )

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample cached source rollouts captured after step and before render.

        The sampling policy frequently fires the main engine to exercise exhaust geometry.
        """
        rng = rng or self.rng
        if self._pool is None:
            self._build_pool(max(256, batch))
        idx = rng.choice(len(self._pool), batch, replace=batch > len(self._pool))
        return self._pool[idx].astype(np.float32)

    def _build_pool(self, n: int) -> None:
        """Collect source rollout frames with one render per step to bound the particle list."""
        env = self.reference_env()
        u = env.unwrapped
        rng = np.random.default_rng(0)
        out, ep = [], 0
        while len(out) < n:
            env.reset(seed=1000 + ep)
            for _ in range(400):
                a = 2 if rng.random() < 0.7 else int(rng.integers(0, 4))
                _, _, te, tr, _ = env.step(a)
                # Capture post-decay particle geometry before the source renderer mutates the list.
                out.append(self.reference_get_state(u))
                u.render()  # what prunes the list; without it nothing does
                if te or tr or len(out) >= n:
                    break
            ep += 1
        self._pool = np.stack(out).astype(np.float32)

    @classmethod
    def reference_env(cls, size: int | None = None):
        import gymnasium

        if size is not None and int(size) != VIEWPORT_H:
            raise ValueError(f"the official renderer draws {VIEWPORT_H}x{VIEWPORT_W} and nothing else")
        env = gymnasium.make("LunarLander-v3", render_mode="rgb_array")
        env.reset(seed=0)
        return env

    @staticmethod
    def reference_get_state(u) -> np.ndarray:
        """Read terrain, body poses, and particles as the next source frame will draw them.

        Use smoothed sky polygons and apply lifetime decay and dead-prefix pruning.
        Raise if more than six particles remain; this bound requires rendering at
        least as often as stepping. Restoration adds the decay back before drawing.
        """
        sy = [float(u.sky_polys[i][0][1]) for i in range(CHUNKS - 1)]
        sy.append(float(u.sky_polys[CHUNKS - 2][1][1]))
        out = list(sy)
        for b in (u.lander, u.legs[0], u.legs[1]):
            out += [float(b.position[0]), float(b.position[1]), float(b.angle)]
        drawn = [(float(p.position[0]), float(p.position[1]), float(p.ttl) - TTL_DECAY) for p in u.particles]
        while drawn and drawn[0][2] < 0:  # _clean_particles(False), same rule
            drawn.pop(0)
        if len(drawn) > N_PARTICLES:
            raise AssertionError(
                f"{len(drawn)} particles would be drawn but the state carries {N_PARTICLES}. The "
                f"bound is a property of rendering at least as often as stepping; if that changed, "
                f"N_PARTICLES has to change with it rather than silently truncating the list."
            )
        # dead slots go at the FRONT, where _clean_particles pops them, so a short list is expressible
        out += [0.0, 0.0, -1.0] * (N_PARTICLES - len(drawn))
        for p in drawn:
            out += list(p)
        return np.asarray(out, np.float64)

    @classmethod
    def reference_set_state(cls, u, st) -> None:
        """Restore drawn terrain, three body poses, and particle lifetimes in the reference."""
        from Box2D.b2 import circleShape, fixtureDef

        v = np.asarray(st, np.float64)
        sy = v[:N_TERRAIN]
        u.sky_polys = [
            [
                (CHUNK_X[i], float(sy[i])),
                (CHUNK_X[i + 1], float(sy[i + 1])),
                (CHUNK_X[i + 1], WORLD_H),
                (CHUNK_X[i], WORLD_H),
            ]
            for i in range(CHUNKS - 1)
        ]
        for j, b in enumerate((u.lander, u.legs[0], u.legs[1])):
            o = BODY_OFF + 3 * j
            b.position = (float(v[o]), float(v[o + 1]))
            b.angle = float(v[o + 2])
        while u.particles:
            u.world.DestroyBody(u.particles.pop(0))
        for k in range(N_PARTICLES):
            o = PART_OFF + 3 * k
            p = u.world.CreateDynamicBody(
                position=(float(v[o]), float(v[o + 1])),
                angle=0.0,
                fixtures=fixtureDef(shape=circleShape(radius=2 / SCALE, pos=(0, 0)), density=1.0),
            )
            # the decrement added BACK, so that render()'s own decrement lands on the stored value
            p.ttl = float(v[o + 2]) + TTL_DECAY
            u.particles.append(p)

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Restore each render state, check its round trip, and draw the official frame.

        The source renderer mutates particles, so restoration is required before every frame.
        """
        u = env.unwrapped
        out = []
        for st in np.asarray(states):
            cls.reference_set_state(u, st)
            got = cls.reference_get_state(u)
            err = np.abs(got - np.asarray(st, np.float64)).max()
            if err > 1e-6:
                raise AssertionError(
                    f"LunarLander: the reference did not reach the requested state (max err "
                    f"{err:.3e}); parity measured against this frame would compare two scenes"
                )
            out.append(np.asarray(u.render()).copy())
        return np.stack(out)
