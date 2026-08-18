"""``swm/SimplePointMaze-v0``: a 5x5 point maze, and the first env here whose reference is matplotlib.

Every other env in this repo is drawn by pygame, by a tile atlas or by MuJoCo. This one is a
matplotlib figure: ``plt.subplots(figsize=(5,5))`` at the default 100 dpi, ``tight_layout(pad=0)``,
``Rectangle`` patches for the walls, a ``Circle`` for the agent, and the frame read back through
``canvas.tostring_argb()``. That makes the interesting question not "can we draw rectangles" but
"is the data-to-pixel map recoverable exactly, and is anything on the canvas outside our vocabulary".
Both were answered by measurement before any of this was written
(``scratch/simple_point_maze_gates.py`` and ``scratch/simple_point_maze_calib.py``):

  * **The axes fill the figure exactly, at 100 px per data unit.** ``tight_layout(pad=0)`` with no
    ticks and no labels leaves no margin at all. Measured by painting a wall of known extent and
    reading off which pixels it covers: data ``x in [1,2] x y in [1,2]`` lands on **rows 300..399,
    columns 100..199**, and three more probes agree, giving ``col = 100*x``, ``row = 500 - 100*y``
    to the pixel. So the scene is simply ``world = (0, 0, 5, 5)`` with ``flip_y``, and nothing here
    converts to pixels by hand.
  * **There is an axes spine, and it is asymmetric.** matplotlib draws the default 0.8 pt black
    frame, which at 100 dpi is 1.11 px and lands differently on the two sides of the canvas.
    Measured on an otherwise empty frame: **row 0 and column 0 are pure black across their whole
    length**, rows/columns **1 and 499 are (241,241,241)**, and the four places where two grey lines
    cross are **(227,227,227)**. Row/column 498 is plain background. A port that draws a symmetric
    1-px border, or none, is wrong on ~2,000 pixels — 0.8% of the frame, all of it at the edge where
    nobody looks.
  * **The spine paints over the walls.** matplotlib's spines have zorder 2.5 against a patch's 1, so
    a wall touching x=0 is cut by the black line rather than covering it. Measured: a wall placed at
    data x in [0, 0.5] starts at column **1**, not 0.
  * **The goal marker is composited at alpha, and this class does not draw it** — see below.

State and appearance
--------------------

The state is the agent's position and nothing else: 2 floats against a 500x500x3 frame, a ratio of
**1/93,750**, the largest in this repo. Everything else on the canvas — the walls' count, sizes,
positions and colour, the agent's colour and radius, the background — is variation, and all of it is
per-environment, so a batch of 4096 mazes with 4096 different layouts is still one launch.

The default wall layout is a constant of the family rather than a free parameter: the official env
draws it once from ``np.random.default_rng(234232)``, shapes first and then positions, and bakes the
result in as the leaf's ``init_value``. The same two draws are reproduced here so that
``default_style()`` is the official appearance, and the parity script asserts the two agree rather
than trusting the reproduction.

Why the goal disc is not drawn
------------------------------

``SimplePointMazeEnv`` takes ``show_goal`` and defaults it to True, drawing the goal as a
``Circle(..., alpha=0.5)``. Two measured facts decide what to do with it:

  1. **The compositing rule is closed-form and was solved.** Over 12 (background, colour) pairs =
     36 channels, ``out = floor(bg + (c - bg) * (128 - eps)/255)`` reproduces **36/36**; plain
     ``floor(bg + (c-bg)*128/255)`` gets 34/36 and ``floor((c+bg)/2)`` gets 27/36. The two channels
     that separate them are the ones where the product is an exact integer — Agg's multiply-shift
     lands one below — which is precisely the kind of constant this repo measures rather than reads.
     So an exact port of the goal disc is available, and the env's own constraint that the goal may
     never overlap a wall (``constrain_fn`` on ``goal.position`` rejects any placement that
     collides) means one pre-blend against the background is always sufficient.
  2. **A half-transparent primitive cannot satisfy ``check_colors_apply_exactly``.** That check sets
     each declared RGB knob to a literal colour and requires the frame to contain that colour
     somewhere. A disc composited at 128/255 never puts its own colour on the canvas — the official
     env does not either: ``goal.color = (17,33,200)`` over the default white background produces
     ``(135,143,227)``. The check is right about opaque primitives and has no way to express this
     one, and it lives in a file this env does not own.

So the configuration registered here is ``show_goal=False``, exactly as ``TwoRoom`` and
``PiecewiseZones`` register one configuration of their official env each. The measured blend rule is
recorded above so that adding the goal disc is a short job once the shared check can express a
translucent knob; the note in ``docs/research/swm_coverage.md`` says what that would take.

Output size
-----------

``render(size=...)`` accepts 500 only. The world is in data units and would scale, but the spine is
defined in OUTPUT PIXELS — a 0.8 pt line at 100 dpi — and there is no size-independent description
of it. Inventing one would produce a picture the reference never draws.

Measured
--------

H200, 48 states from ``sample_states(rng=default_rng(3))``, default appearance
(``scratch/spm_parity.py``):

    interior pixels exact   100.000%          <- the bar; no tolerance relaxed
    all pixels exact         99.9690%         (3,719 px of 12.0 M, max 241/255)
    of those, on the border     233           the four (227,227,227) spine crossings, 4 per frame
    the rest                  3,486           the agent disc's anti-aliased rim, ~73 px per frame
    GPU frames/s             26,301  at batch 256, into a caller-owned buffer
    official frames/s            46.2
    speedup                     569x

    step vs the official simulator: exact_frac 1.000, max error 0.0 over 8 episodes x 60 steps
    fused kernel vs the torch reference: 100% of pixels, max abs diff 0

The border residue is the one place a flat-primitive renderer cannot follow matplotlib: where two
anti-aliased grey spine lines cross, Agg composites them twice and gets (227,227,227) where a single
flat over-paint gets 241. Four pixels per frame, all four at the extreme corners, and all four
outside the interior mask because their 3x3 neighbourhood is not flat in the reference either.
"""
from __future__ import annotations

import numpy as np
import torch

from ..core.env import StableEnv, register
from ..render.scene import Scene2D, SceneBuilder
from ..spaces import Box, Dict, Discrete, RGBBox

#: figsize 5x5 inches at matplotlib's default 100 dpi
SIZE = 500
#: the data extent of the axes; ``ax.set_xlim(0, width)`` / ``set_ylim(0, height)``
WORLD = 5.0
PX_PER_UNIT = SIZE / WORLD                       # 100.0, measured
MAX_WALLS = 6
MIN_WALLS = 4

AGENT_C = (255, 0, 0)
WALL_C = (0, 0, 0)
BG_C = (255, 255, 255)
AGENT_R = 0.1
N_WALLS = 5
#: The reference's own action bounds: ``Box(-0.2, 0.2)``.
ACTION_LIMIT = 0.2
AGENT_SPEED = 1.0

#: One output pixel, in data units. The spine is defined in pixels, which is why this env renders at
#: one size — see the module docstring.
PX = WORLD / SIZE
#: How much of the background the anti-aliased spine line removes, as a fraction. Measured on a white
#: canvas: 255 -> 241, i.e. 14/255. Applied multiplicatively so the line still reads as a darkened
#: background when the background is not white.
SPINE_AA = 14.0 / 255.0


def _default_walls() -> tuple[np.ndarray, np.ndarray]:
    """The official default wall layout, reproduced from the draw that produced it.

    ``SimplePointMazeEnv.__init__`` makes one ``default_rng(234232)`` and takes the shapes first and
    then the positions; the ORDER is load-bearing, because both draws come from the same stream and
    swapping them gives a completely different maze that still looks plausible. The parity script
    asserts these against the live reference rather than trusting the reproduction.
    """
    rng = np.random.default_rng(234232)
    shape = rng.uniform(low=0.5, high=1.5, size=(MAX_WALLS, 2)).astype(np.float32)
    pos = rng.uniform(low=1.0, high=3.5, size=(MAX_WALLS, 2)).astype(np.float32)
    return shape, pos


DEFAULT_WALL_SHAPE, DEFAULT_WALL_POS = _default_walls()


@register
class SimplePointMaze(StableEnv):
    """``swm/SimplePointMaze-v0`` with ``show_goal=False``. State: the agent's ``(x, y)`` in a 5x5 box."""

    name = "swm-simple-point-maze"
    state_dim = 2
    state_names = ("agent_x", "agent_y")
    default_size = SIZE
    reference_id = "swm/SimplePointMaze-v0"
    action_dim = 2
    action_low = -ACTION_LIMIT
    action_high = ACTION_LIMIT
    step_reference_checkable = True
    step_reference_exact = True
    step_reference_groups = (("agent", (0, 1), "euclidean"),)
    step_reference_control = "agent"
    step_reference_gate = "agent"
    step_reference_hold = 8

    #: DYNAMICS only: ``agent.speed`` scales the move and changes no pixel, so it is an attribute
    #: rather than a variation leaf. Declaring it would advertise a knob the renderer cannot honour.
    speed = AGENT_SPEED

    def make_variation_space(self) -> Dict:
        return Dict({
            "agent": Dict({
                "color": RGBBox(init_value=AGENT_C),
                "radius": Box(low=0.05, high=0.5, shape=(1,), init_value=[AGENT_R]),
            }),
            "walls": Dict({
                "number": Discrete(MAX_WALLS - MIN_WALLS + 1, start=MIN_WALLS, init_value=N_WALLS),
                "color": RGBBox(init_value=WALL_C),
                "shape": Box(low=0.5, high=1.5, shape=(MAX_WALLS, 2),
                             init_value=DEFAULT_WALL_SHAPE),
                "positions": Box(low=0.0, high=WORLD, shape=(MAX_WALLS, 2),
                                 init_value=DEFAULT_WALL_POS),
            }),
            "background": Dict({"color": RGBBox(init_value=BG_C)}),
        })

    # ---------------------------------------------------------------- scene

    def scene(self, states, variation: dict | None = None, style=None) -> Scene2D:
        s = torch.as_tensor(states, device=self.device, dtype=torch.float32).reshape(-1, 2)
        n = int(s.shape[0])
        sty = self.resolve_style(style, variation, batch=n)
        sb = SceneBuilder(n, device=self.device, vertices=4)

        bg = sty["background.color"]
        shape = sty["walls.shape"].reshape(n, MAX_WALLS, 2)
        pos = sty["walls.positions"].reshape(n, MAX_WALLS, 2)
        num = sty["walls.number"].trunc()
        wcol = sty["walls.color"]
        for i in range(MAX_WALLS):
            # matplotlib's Rectangle((x, y), w, h) covers [x, x+w] x [y, y+h], so the centre is the
            # corner plus half the size. Slots beyond walls.number are DISABLED rather than dropped,
            # so the primitive count never depends on the state.
            sb.add_rect(pos[:, i] + 0.5 * shape[:, i], shape[:, i], wcol, enabled=num > i)
        # walls are variation-only; baking them is refused automatically if a per-environment colour
        # or layout makes the prefix differ across the batch (SceneBuilder.mark_static)
        sb.mark_static()

        sb.add_disc(s, sty["agent.radius"], sty["agent.color"])

        # The spine, last, because matplotlib gives it zorder 2.5 against a patch's 1 — measured: a
        # wall placed at x in [0, 0.5] starts at column 1, so the black line is on top of it.
        # The rows and columns each band lands on, with ``flip_y`` and 100 px per unit: row r samples
        # ``y = 5 - (r + 0.5)/100``, so row 0 is ``y in [4.99, 5]`` and row 1 is ``y in [4.98,
        # 4.99]``. Writing the grey top band as ``[4.99, 5]`` puts it under the black one and leaves
        # row 1 as background — measured as 24,089 disagreeing border pixels before this comment
        # existed, which is what the pixel diff is for.
        grey = bg * (1.0 - SPINE_AA)
        edges = ((0.0, WORLD - 2 * PX, WORLD, WORLD - PX, grey),   # row 1
                 (0.0, 0.0, WORLD, PX, grey),                      # row 499
                 (PX, 0.0, 2 * PX, WORLD, grey),                   # col 1
                 (WORLD - PX, 0.0, WORLD, WORLD, grey),            # col 499
                 (0.0, WORLD - PX, WORLD, WORLD, None),            # row 0   (black)
                 (0.0, 0.0, PX, WORLD, None))                      # col 0   (black)
        black = torch.zeros(n, 3, device=self.device, dtype=torch.float32)
        for x0, y0, x1, y1, col in edges:
            sb.add_rect(((x0 + x1) * 0.5, (y0 + y1) * 0.5), (x1 - x0, y1 - y0),
                        black if col is None else col)
        return sb.build(background=bg, world=(0.0, 0.0, WORLD, WORLD), flip_y=True)

    def render(self, states=None, size: int | None = None, supersample: int | None = None,
               variation: dict | None = None, style=None, out=None) -> torch.Tensor:
        """As the base class, but at 500px only.

        The world is in data units and would scale happily; the SPINE is defined in output pixels (a
        0.8 pt line at 100 dpi) and has no size-independent description. Rendering at 1000 would
        produce a 1-pixel frame around a picture whose reference draws a 2-pixel one, i.e. a
        different image rather than the same one sampled more finely.
        """
        t = int(size or self.size)
        if t != SIZE:
            raise ValueError(
                f"SimplePointMaze renders at {SIZE}px only: the reference is a 5x5 inch matplotlib "
                f"figure at 100 dpi and its axes spine is a 0.8 pt line, i.e. a constant in OUTPUT "
                f"PIXELS with no world-unit description. Asked for {t}.")
        return super().render(states, size=t, supersample=supersample, variation=variation,
                              style=style, out=out)

    # ---------------------------------------------------------------- states and dynamics

    def _wall_boxes(self, sty, n):
        shape = sty["walls.shape"].reshape(n, MAX_WALLS, 2)
        pos = sty["walls.positions"].reshape(n, MAX_WALLS, 2)
        num = sty["walls.number"].trunc().view(n, 1)
        lo = torch.minimum(pos, pos + shape)
        hi = torch.maximum(pos, pos + shape)
        live = torch.arange(MAX_WALLS, device=self.device).view(1, -1) < num
        return lo, hi, live

    def _collides(self, p, sty, n) -> torch.Tensor:
        """``(n,)`` bool: is a disc of ``agent.radius`` at ``p`` touching any live wall?

        The reference's test verbatim — the closest point of each axis-aligned box to the centre,
        then a squared-distance comparison against the radius — because this decides the DYNAMICS,
        and a collision rule that is merely similar produces trajectories that diverge on the first
        contact rather than pixels that differ by one.
        """
        lo, hi, live = self._wall_boxes(sty, n)
        r = sty["agent.radius"].reshape(n, 1)
        c = torch.clamp(p.unsqueeze(1), lo, hi)
        d2 = ((c - p.unsqueeze(1)) ** 2).sum(-1)
        return (live & (d2 <= r * r)).any(1)

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """The official reset distribution for ``agent.position``: ``Box(0,5)^2``, rejecting walls.

        ``DEFAULT_VARIATIONS`` is ``('agent.position',)``, so this is the ONLY thing the official
        reset resamples, and its ``constrain_fn`` rejects any position whose disc touches a wall.
        Sampling the raw box instead would spend a fifth of every parity measurement on the inside of
        a wall, which the env never resets into.
        """
        rng = rng or self.rng
        n = int(batch)
        sty = self.resolve_style(batch=1)
        out = np.empty((n, 2), np.float32)
        filled = 0
        for _ in range(1000):
            if filled >= n:
                break
            cand = rng.uniform(0.0, WORLD, (n, 2)).astype(np.float32)
            t = torch.as_tensor(cand, device=self.device)
            ok = ~self._collides(t, self._style_for(sty, n), n)
            keep = cand[ok.cpu().numpy()]
            take = min(n - filled, len(keep))
            out[filled:filled + take] = keep[:take]
            filled += take
        if filled < n:
            raise RuntimeError("agent.position rejection sampling did not converge; the wall layout "
                               "leaves no free space")
        return out

    def _style_for(self, sty, n):
        """Broadcast a batch-1 style to ``n`` rows without rebuilding it from the space."""
        from ..core.style import StyleTensor
        if sty.batch == n:
            return sty
        return StyleTensor(sty.values[:1].expand(n, -1).contiguous(),
                           [(p, o, w) for p, (o, w) in sty.layout.items()])

    @torch.no_grad()
    def step(self, action) -> torch.Tensor:
        """One control step. Exact against the reference: a clip, a collision test and a clip.

        The order is the reference's and it is not the obvious one — the collision test runs on the
        UNCLAMPED candidate and reverts the whole move, and only then is the result clipped to the
        arena. Testing after clipping would let an agent slide along a wall that the reference stops
        dead.
        """
        a = torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(self.B, 2)
        a = a.clamp(-ACTION_LIMIT, ACTION_LIMIT)
        p = self._state
        nxt = p + float(self.speed) * a
        blocked = self._collides(nxt, self._style, self.B)
        nxt = torch.where(blocked.unsqueeze(1), p, nxt)
        self._state.copy_(nxt.clamp(0.0, WORLD))
        return self._state

    # ---------------------------------------------------------------- reference bridge

    @staticmethod
    def reference_env(size: int = SIZE):
        import gymnasium
        import stable_worldmodel  # noqa: F401  (registers the swm/* ids)
        return gymnasium.make("swm/SimplePointMaze-v0", show_goal=False)

    @staticmethod
    def reference_sim_state(env) -> np.ndarray:
        u = getattr(env, "unwrapped", env)
        return np.asarray(u.state, np.float64).reshape(2)

    @staticmethod
    def reference_step(ref, a):
        return ref.step(np.asarray(a, np.float32))

    @staticmethod
    def step_reference_reset(ref, seed):
        ref.reset(seed=int(seed))
        return np.asarray(getattr(ref, "unwrapped", ref).state, np.float64).reshape(2)

    @staticmethod
    def reference_render(env, states) -> np.ndarray:
        """Official frames for the given agent positions.

        ``options={'variation': []}`` matters as much here as it does for the other swm ports: the
        default is ``('agent.position',)``, so a plain reset resamples the leaf whose value our
        ``default_style`` is defined to equal, and every number measured afterwards would describe a
        different appearance.
        """
        u = env.unwrapped
        env.reset(seed=0, options={"variation": []})
        out = []
        for st in np.asarray(states):
            u.state = np.asarray(st, np.float32).reshape(2)
            got = np.asarray(u.state, np.float64)
            err = float(np.abs(got - np.asarray(st, np.float64)).max())
            if err > 1e-6:
                raise AssertionError(
                    f"the reference env did not reach the requested state (max err {err:.3e})")
            out.append(np.ascontiguousarray(u.render("rgb_array")))
        return np.stack(out)
