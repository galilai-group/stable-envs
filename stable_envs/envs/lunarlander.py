"""LunarLander-v3: the env that was excluded because ``render()`` mutates what it draws.

``docs/research/classic_control.md`` left this one out with a specific reason -- ``render()`` runs
``for obj in self.particles: obj.ttl -= 0.15`` and then ``self._clean_particles(False)``, so rendering
twice gives two different images and the frame is not a function of the observation. That is an
argument for WIDENING the state, not for excluding the env, and the three questions that decide which
have now been asked:

  **A. Is the mutation deterministic?** Yes. Restoring the env to a recorded state and rendering it
  twice gives **max absolute pixel difference 0**; rendering twice WITHOUT restoring gives 63/255,
  which is the mutation and is exactly why the particles have to be in the state.

  **B. Is the particle list bounded?** Yes, in the only regime that produces frames. A discrete
  LunarLander creates at most one particle per ``step``, and only when ``render_mode`` is set; the
  discrete env fixes ``m_power = s_power = 1.0``, so every particle starts at ``ttl = 1.0`` and loses
  0.15 per render, and the seventh decrement takes it below zero. Measured over full episodes under
  three policies (main engine every step, main and side alternating, uniform random): **peak 6 drawn**
  -- and 118, 126 and 77 alive when ``render()`` is never called, because nothing else prunes the
  list. The bound is a property of "render as often as you step", which is the only way frames exist
  at all, and ``reference_get_state`` raises rather than truncating if it is ever exceeded.

  **C. Does a fixed-width state determine the frame?** Yes, and this is the measurement that decides
  the verdict. A DIFFERENT env instance, with a different terrain and twenty steps of a different
  history behind it, driven to the same numbers, renders **max absolute pixel difference 0** against
  the original. The frame is a pure function of (terrain, lander, legs, particles) and of nothing else.

The state stores each particle's ``ttl`` **as it is drawn**, not as the simulator holds it, and that is
what makes it minimal rather than merely sufficient. ``render`` decrements every ``ttl`` and drops the
dead prefix BEFORE it draws anything, so the env's list is one particle longer than the frame's -- and
always by the same slot, the oldest. A state that copied the list verbatim would carry an x and a y
that provably cannot move a pixel, which the conformance suite reports as a dead dimension and did.

So the verdict is GO, on a wider contract than "render the observation": this is "replay the
simulator's render state", the same contract MiniGrid's top-down classes and PushT already use. The
observation is a lossy function of it -- it carries neither the terrain nor the leg joint angles nor
the exhaust.

Three things about this renderer that are not visible in the source
--------------------------------------------------------------------

**The landing flags are drawn once per FIXTURE, so the number of live particles changes their colour.**
The ``for x in [helipad_x1, helipad_x2]`` loop is nested INSIDE ``for f in obj.fixtures``, so the flags
are painted ``len(particles) + 3`` times per frame. The opaque parts of a flag are idempotent; its
``gfxdraw.aapolygon`` fringe is not, because a Wu blend at alpha ``(255*w)>>8`` never reaches the
source colour and converges toward it with repetition. Measured on the flag alone: 2 repetitions move
40 pixels by up to 62/255, 3 by 98/255, 7 by 152/255. The draw order is reproduced exactly for that
reason -- particle, flag, flag, particle, flag, flag, ..., lander, flag, flag, leg, flag, flag.

**Three different rasterizers appear in one frame**, and only one of them is the integer SDL_gfx path
``render/gfxdraw.py`` reproduces exactly:

    pygame.draw.polygon  +  gfxdraw.aapolygon  +  pygame.draw.aalines     (lander, legs)
    pygame.draw.polygon  +  gfxdraw.aapolygon                             (terrain, flags)
    pygame.draw.circle   x2                                               (particles)

``pygame.draw.polygon`` is NOT ``gfxdraw.filled_polygon``: measured over 300 random convex polygons
they differ on **3.63% of lit pixels**, all of them at the ends of scanlines, and the difference is not
the vertex handling (both truncate -- a quad shifted by 0.9 px moves nothing and by 1.0 px moves one
column). ``pygame.draw.aalines`` is a sub-pixel anti-aliasing path unrelated to gfxdraw's Wu line and
is not reimplemented. What that costs is measured and reported below rather than hidden: it is the
whole of the all-pixel residual, and none of it is an interior pixel, because every one of those
disagreements is within a pixel or two of a shape boundary.

**``pygame.draw.circle`` is a 12-pixel stencil here, not a circle.** Every particle has
``radius = 2/SCALE``, and ``radius * SCALE`` is 2.0000001 which truncates to 2; the centre truncates
too (60.0, 60.4, 60.5 and 60.9 all give the same columns). So the shape is a constant, read off pygame
once, and it is not symmetric: it spans dy and dx from -2 to +1.

Physics is not ported. Box2D's contact solver is chaotic in exactly the way
``env.py`` describes, the legs are two revolute joints with motors and limits, and a
reimplementation would diverge; the renderer stands on its own, and states come from the official env.
"""

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

#: Fixture vertices, in PIXELS, read off the live bodies. ``render`` draws ``trans * v * SCALE`` where
#: ``v`` is the shape's vertex in world units, so a vertex in pixels is ``R(angle) @ v_px + pos*SCALE``.
#: Box2D reorders ``LANDER_POLY`` into its own counter-clockwise winding, which is the order the edges
#: are anti-aliased in, so it is taken from the object rather than from the source constant.
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
#: The most particles that can be DRAWN in one frame, measured over full episodes under three
#: policies; ``reference_get_state`` raises rather than truncating if a state ever needs more, so the
#: bound is checked on every state read rather than assumed. Six, and not by luck: the discrete env
#: sets ``m_power = s_power = 1.0``, so every particle starts at ``ttl = 1.0`` and loses 0.15 a frame,
#: and the seventh decrement takes it below zero before it is ever drawn again.
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
    """Probe the whole runtime AT IMPORT, so a machine without it loses this env cleanly.

    ``import X`` succeeding is not the same as X working, and this module imports nothing at the top
    that could fail: ``render`` is pure torch. What cannot be done without Box2D and pygame is produce
    a STATE -- ``sample_states`` rolls the official env out, because the terrain, the leg joint angles
    and the exhaust have no closed form -- and compare against the reference. An env that renders but
    cannot say what to render is not usable, so the dependency is a hard one and is declared as one:
    ``ImportError`` here means ``stable_envs/envs/__init__.py`` records it in ``import_errors()`` and
    the env is cleanly ABSENT, rather than present and failing inside every fixture.

    The probe is the failing path itself -- reset and render one frame, which is milliseconds -- and
    not merely ``import Box2D``: gymnasium raises ``DependencyNotInstalled`` from inside
    ``envs/box2d/lunar_lander.py``, and pygame is a separate requirement that only ``render`` reaches.
    """
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
    """A canvas that records what ``aaline`` WOULD write instead of writing it.

    Used once, at construction, on the flag triangle -- whose vertices are compile-time constants -- so
    that its Wu fringe becomes a list of ``(row, column, weight)`` in the reference's own order and the
    per-frame cost drops from a clip plus a Wu walk to a scatter. Empty writes are dropped, which is
    most of them: ``aaline`` issues five scatters covering the vertical, horizontal and general cases
    and any one edge uses two or three.
    """

    def __init__(self, gfx: G.Gfx):
        super().__init__(1, gfx.H, gfx.W, gfx.device, gfx.sh, gfx.ys, gfx.xs)
        self.writes: list[tuple] = []

    def blend_at(self, b, y, x, color_t, w, valid) -> None:
        ok = valid & (y >= 0) & (y < self.H) & (x >= 0) & (x < self.W)
        if bool(ok.any()):
            # ``expand`` before indexing: the weight a primitive hands over may be one element that
            # every write shares (``Canvas.solid``), and a boolean index wants it at full length.
            # Belt and braces -- ``aaline_emit`` expands it too -- because this override is what a
            # weight-shape change breaks first, and it breaks inside ``LunarLander.__init__``.
            self.writes.append((y[ok].clone(), x[ok].clone(), w.expand(ok.shape)[ok].clone()))


def _record_aa(gfx: G.Gfx, vi: torch.Tensor) -> list[tuple]:
    """``gfxdraw.aapolygon`` on a CONSTANT polygon ``(1, V, 2)`` -> its ordered write list."""
    rec = _AARecorder(gfx)
    n = vi.shape[1]
    bidx = torch.zeros(1, dtype=torch.long, device=gfx.device)
    for e in range(n):
        G.aaline(rec, vi[:, e, 0], vi[:, e, 1], vi[:, (e + 1) % n, 0], vi[:, (e + 1) % n, 1], None, 64, bidx)
    return rec.writes


@register
class LunarLander(Env):
    """LunarLander-v3, from a 38-number render state: terrain, lander, legs and the exhaust.

    11 smoothed terrain heights, 3 poses of 3 numbers, and 6 particle slots of ``(x, y, ttl)``. A
    400x600x3 frame is 720,000 bytes against 38 float32, a **4,737x** shrink.
    """

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
    #: Public state. The row is already four runs and the layout names them: the
    #: smoothed terrain profile, the three rigid bodies as ``(x, y, angle)`` each, and the exhaust
    #: particle slots as ``(x, y, ttl)`` each. Column order is unchanged, every column is covered,
    #: nothing is hidden. The lander's VELOCITY is deliberately absent: Box2D integrates it, this
    #: package renders from a pose, and a column the frame never reads would be a dead dimension --
    #: which is also why ``step``/``transition`` refuse rather than pretending to simulate.
    state_layout = (
        (("terrain",), N_TERRAIN, 0),
        (("lander", "pose"), 3, BODY_OFF),
        (("leg0", "pose"), 3, BODY_OFF + 3),
        (("leg1", "pose"), 3, BODY_OFF + 6),
        (("particles",), 3 * N_PARTICLES, PART_OFF),
    )

    #: Interior pixels are exact; the residual is the two ``pygame.draw`` rasterizers this module does
    #: not reimplement, and it lives entirely on shape boundaries. Kept as a declared floor of 1.0
    #: because that is what is measured -- see ``docs/research/lunarlander.md`` for the all-pixel
    #: number and its decomposition.
    parity_interior_min = 1.0

    @property
    def frame_shape(self) -> tuple[int, int]:
        """``(H, W)`` of a frame: always ``native``, i.e. 400x600.

        ``Env.observation_spec`` assumes a SQUARE frame unless the env says otherwise, and this one is
        not: ``size`` is the reference viewport's HEIGHT, so an unqualified spec described a 400x400
        buffer and the frame is 400x600. There is no size at which it is square -- ``render`` raises
        for any ``size != 400`` because every primitive is hardcoded in pixels -- so this is a
        constant, not an aspect ratio. Same declaration, for the same reason, as
        ``classic_control.ClassicControlEnv.frame_shape`` (three of those five are also 400x600) and
        ``highway``/``minihack``.
        """
        return self.native

    def _close(self) -> None:
        """Drop the sampled state pool. ``Env.close`` calls this once and is idempotent.

        This env owns no external handle: the Box2D world it samples from is built per call and
        discarded, and everything else it holds is a constant device tensor baked from the module's
        own geometry. The pool is the one growable thing, and a closed env has no use for it.
        """
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
        # The flags are at CONSTANT coordinates -- the helipad sits at fixed chunk boundaries and
        # helipad_y is VIEWPORT_H/SCALE/4 -- so their whole rasterisation is a compile-time constant
        # and each draw is a scatter of ~180 pixels rather than a pass over the frame. They are drawn
        # once per fixture, i.e. up to ten times a frame, which is what makes that worth doing.
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
        # The anti-aliased fringe is a constant too, so it is recorded ONCE here rather than
        # re-derived per frame. That is the difference between a usable renderer and an unusable one:
        # a Wu edge costs a six-iteration Cohen-Sutherland clip, and the flags are drawn up to ten
        # times a frame, which measured at 513 ms of the 979 ms a batch of 256 took.
        self._tri_aa = [
            (torch.cat([y, y]), torch.cat([x, x + dx]), torch.cat([w, w])) for y, x, w in _record_aa(self.gfx, vi)
        ]

    # --- variation --------------------------------------------------------------------------------

    def make_variation_space(self) -> Dict:
        """Colours that are literally painted, and the anti-aliasing switch. Nothing else.

        No knob for the exhaust: a particle's colour is
        ``int(max(0.2, 0.15 + ttl) * 255)`` on red and ``int(max(0.2, 0.5 * ttl) * 255)`` on green and
        blue, i.e. a function of a STATE dimension rather than an appearance constant. A knob that
        multiplied it would not put its own colour in the frame, and the conformance suite checks
        exactly that for an RGB leaf.

        No knob for the lander's outline colour either, for a different reason: the outline is drawn by
        ``pygame.draw.aalines``, which this module does not reimplement, so declaring the knob would
        advertise something that cannot move a pixel.
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

    # --- rendering --------------------------------------------------------------------------------

    @torch.no_grad()
    def render(
        self,
        states=None,
        size: int | None = None,
        supersample: int | None = None,
        variation: dict | None = None,
        style=None,
        out=None,
    ) -> torch.Tensor:
        """``(...,38)`` -> ``(...,400,600,3)`` uint8 on device, for arbitrary leading dims."""
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
        s = self._state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
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
        """``render()``'s own sequence, in its own order, on a pre-flip (y down) canvas."""
        B = s.shape[0]
        bidx = self.gfx.batch_index(B)
        terrain = sty["terrain.color"]
        lander_c = sty["lander.color"]

        # 1. the exhaust. ``ttl`` is stored AS DRAWN (see the class docstring), so nothing is
        #    decremented here; what does have to be reproduced is ``_clean_particles(False)``, which
        #    pops from the FRONT while the front particle is dead. A dead slot in the MIDDLE of the
        #    list is not removed and would be drawn, so the survivors are the slots at or after the
        #    first live one -- a running maximum rather than a filter.
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
            # each quad's left and right edges are VERTICAL and at known columns, so its fill is
            # confined to a 61 px band and pays for that band rather than for the whole frame
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
        """``pygame.draw.circle`` twice in the same colour: one 12-pixel stencil at a truncated centre."""
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
        # ``Canvas`` is planar ``(3, B*H*W+1)``, so the channel axis is the one NOT being indexed and
        # the scatter is still one operation. See ``render/gfxdraw.Canvas`` for why it is planar.
        cv.flat[:, flat.reshape(-1)] = G.to_u8(color).t().reshape(3, B, 1).expand(3, B, n).reshape(3, -1)

    def _aa_poly(self, cv, vi, bidx, kmax) -> None:
        """``gfxdraw.aapolygon`` for ``(N, V, 2)`` integer polygons, as ONE Wu pass over N*V edges.

        Every edge of every polygon is issued in a single call rather than one call per edge, and that
        is a throughput decision with a stated cost. A Wu edge carries a six-iteration Cohen-Sutherland
        clip -- about 150 tiny kernel launches whatever its length -- so the price of this pass is the
        NUMBER of calls, not the number of pixels: 8.6 ms a call, measured, at any batch size. Issuing
        the terrain's 40 edges together instead of four at a time takes the render from 262 to 3,300
        frames/s at batch 256.

        What it gives up is the ordering of writes where two edges land on the same pixel, i.e. at the
        shared vertices, because a batched scatter reads every destination before any of them is
        written. Measured against the official renderer over 32 states: **identical**, 2,376 differing
        pixels either way. That is not luck -- ``aapolygon`` draws every edge in ONE colour, and the
        fill underneath it is that same colour, so the two candidate values at a shared vertex are a
        blend of a colour over itself.
        """
        N, V, _ = vi.shape
        a = vi.reshape(-1, 2)
        b = torch.roll(vi, -1, dims=1).reshape(-1, 2)
        G.aaline(cv, a[:, 0], a[:, 1], b[:, 0], b[:, 1], None, kmax, bidx.view(-1, 1).expand(N, V).reshape(-1))

    def _body(self, cv, verts, x, y, angle, color, bidx) -> None:
        """One rigid body: ``pygame.draw.polygon`` then ``gfxdraw.aapolygon``, on the SAME vertices.

        ``pygame.draw.aalines`` paints the outline on top of both and is not reimplemented; see the
        module docstring for what that costs and why it is all boundary.
        """
        c, sn = torch.cos(angle), torch.sin(angle)
        vx, vy = verts[:, 0].view(1, -1), verts[:, 1].view(1, -1)
        px = c.view(-1, 1) * vx - sn.view(-1, 1) * vy + (x * SCALE).view(-1, 1)
        py = sn.view(-1, 1) * vx + c.view(-1, 1) * vy + (y * SCALE).view(-1, 1)
        vi = torch.trunc(torch.stack([px, py], -1)).to(torch.int64)
        cv.use(color).paint(G.fill_polygon_mask(vi[..., 0], vi[..., 1], self.gfx.ys, self.gfx.xs, VIEWPORT_W))
        self._aa_poly(cv, vi, bidx, 64)

    def _flags(self, cv, sty, bidx) -> None:
        """The two landing flags, drawn once per FIXTURE by the reference and so once per call here.

        Every coordinate is a constant: the helipad is at fixed chunk boundaries and ``helipad_y`` is
        ``VIEWPORT_H / SCALE / 4``, so the poles are columns 240 and 360, rows 100 to 150 inclusive
        (``pygame.draw.line`` at width 1 truncates its endpoints and lights both of them), and each
        triangle is 25 px wide.
        """
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

    # --- states -----------------------------------------------------------------------------------

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """States from real rollouts of the official env, captured after ``step`` and before ``render``.

        The policy fires the main engine most of the time, and that is not a detail: an episode that
        never fires an engine never creates a particle, so all 21 exhaust dimensions would be dead and
        the flags -- whose anti-aliased fringe is blended once per live particle -- would never be
        drawn more than three times. A distribution that never exercises the thing the env is being
        widened FOR proves nothing about it.
        """
        rng = rng or self.rng
        if self._pool is None:
            self._build_pool(max(256, batch))
        idx = rng.choice(len(self._pool), batch, replace=batch > len(self._pool))
        return self._pool[idx].astype(np.float32)

    def _build_pool(self, n: int) -> None:
        """Rollouts of the official env, with ``render`` called every step -- which is what bounds the
        particle list, and the only regime in which frames exist at all."""
        env = self.reference_env()
        u = env.unwrapped
        rng = np.random.default_rng(0)
        out, ep = [], 0
        while len(out) < n:
            env.reset(seed=1000 + ep)
            for _ in range(400):
                a = 2 if rng.random() < 0.7 else int(rng.integers(0, 4))
                _, _, te, tr, _ = env.step(a)
                # captured BEFORE the render: reference_get_state applies render()'s own decrement and
                # prune, so this is the frame that is about to be drawn. It also raises if the list is
                # longer than N_PARTICLES, which is where that bound is checked rather than assumed.
                out.append(self.reference_get_state(u))
                u.render()  # what prunes the list; without it nothing does
                if te or tr or len(out) >= n:
                    break
            ep += 1
        self._pool = np.stack(out).astype(np.float32)

    # --- reference bridge ---------------------------------------------------------------------------

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
        """The render state of a live env: terrain, lander, legs, and the exhaust AS IT WILL BE DRAWN.

        The terrain is read from ``sky_polys`` rather than from the ``height`` array ``reset`` sampled,
        because ``sky_polys`` is what is DRAWN -- it holds the smoothed heights, and the smoothing
        reads ``height[i-1]`` at ``i = 0``, which numpy resolves as ``height[11]``. Reading the drawn
        quantity means that quirk cannot be transcribed wrongly.

        The particles get the same treatment, and it is what makes the state MINIMAL. ``render``
        decrements every ``ttl`` and then drops the dead prefix before it draws anything, so the list
        the env is holding is one particle longer than the list the next frame shows -- and the extra
        one is always the same slot, the oldest. Storing the env's list verbatim therefore leaves the
        oldest slot's x and y unable to change any pixel, which is a dead state dimension and was
        caught as one. So the decrement and the prune are applied HERE, and the state describes the
        frame rather than the simulator: ``reference_set_state`` adds the decrement back so that the
        reference's own ``render`` lands on exactly these values.
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
        """Drive the reference to ``st``: rebuild the terrain, the two rigid bodies and the exhaust."""
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
        """Official frames, with the round-trip asserted before each one.

        ``render()`` MUTATES the particles it draws, so the state is re-established for every frame
        rather than once for the batch -- which is the whole reason the particles are in the state and
        is also why the assertion below is on the state as it is BEFORE the render.
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
