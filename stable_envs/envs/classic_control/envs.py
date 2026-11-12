from __future__ import annotations

import math

import numpy as np
import torch
from tensordict import TensorDict

from ...env import Env, _split_appearance
from ...registry import register
from ...render import _resources as _RES
from ...render import gfxdraw as G
from ...spaces import Dict, Discrete, RGBBox
from .semantics import ClassicControlSemantics

WHITE = (255, 255, 255)


_StyleCanvas = G.StyleCanvas
_u8 = G.to_u8


def _poly(cv, verts, color, bidx, kmax, win=None) -> None:
    G.polygon(cv.use(color), verts, None, None, bidx, kmax, True, window=win)


def _circ(cv, cx, cy, r, color, bidx, gfx) -> None:
    """Draw an antialiased circle fringe followed by its filled disc."""
    G.circle(cv.use(color), cx, cy, r, None, None, bidx, gfx.hw(r), gfx.stencil(r), True, disc=gfx.disc(r))


def _row(cv, y: int, color) -> None:
    """Paint a full-width row with per-environment colors."""
    cv.buf[:, :, y, :] = _u8(color).t().unsqueeze(-1)


def _box(H: int, W: int, y0, y1, x0, x1, m: int = 2):
    """Clamp a primitive footprint to the canvas, expanding it by m pixels.

    Bounds must hold for every input pose. The margin covers coordinate truncation
    at integer boundaries.
    """
    return (max(0, int(y0) - m), min(H, int(y1) + 1 + m), max(0, int(x0) - m), min(W, int(x1) + 1 + m))


class _Classic(Env):
    """Share native canvas sizing, float64 geometry, and batched output allocation.

    Subclasses draw primitives before the final row flip or describe them for
    the fused kernel. Compact render rows omit simulation-only fields.
    """

    family = "classic_control"

    # Native (H, W); source primitive dimensions are fixed in pixels.
    native: tuple[int, int] = (0, 0)

    # Numerical variables map explicitly to render columns; torque may be drawn while velocity is omitted.
    sim_state_names: tuple[str, ...] = ()
    #: index of each render-state dim inside ``sim_state_names``
    render_from_sim: tuple[int, ...] = ()
    # Half-open discrete bounds make floored uniform samples uniform over the action set.
    action_dim: int = 1
    action_low: float = 0.0
    action_high: float = 1.0
    #: use the fused per-pixel kernel where it is available. False forces the per-primitive path,
    #: which a CPU render always takes.
    use_fused_render: bool = True

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        H, W = self.native
        self.gfx = G.Gfx(H, W, self.device)
        f = dict(device=self.device, dtype=torch.float64)
        self._sim = torch.zeros(self.B, max(1, len(self.sim_state_names)), **f)
        self._cols = torch.tensor(self.render_from_sim, dtype=torch.long, device=self.device)
        self._disc_tables = None
        self._reference = self._reference_style = self._has_pygame = None

    def _set_render_state(self, states, variation=None, style=None) -> _Classic:
        """Install compact render rows and zero simulation variables absent from those rows."""
        super()._set_render_state(states, variation=variation, style=style)
        self._sim.zero_()
        self._sim[:, self._cols] = self._render_state.to(torch.float64)
        return self

    def _restore_state(self, state: TensorDict) -> None:
        self.set_sim_state(self._sim_from_state(state))

    def sim_state(self) -> torch.Tensor:
        """Return the live (B, sim_dim) float64 numerical state, including velocities."""
        return self._sim

    def set_sim_state(self, v) -> _Classic:
        self._sim.copy_(
            torch.as_tensor(v, device=self.device, dtype=torch.float64).reshape(self.B, self._sim.shape[1])
        )
        self._render_state.copy_(self._sim.index_select(1, self._cols).to(torch.float32))
        return self

    # Derive packed offsets from render_from_sim; simulation-only variables have offset None.

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls.sim_state_names:
            col = {i: j for j, i in enumerate(cls.render_from_sim)}
            # Keep numerical fields in float64; the packed renderer row remains float32.
            cls.state_layout = tuple(((nm,), 1, col.get(i), torch.float64) for i, nm in enumerate(cls.sim_state_names))
            cls._check_state_layout_is_complete()

    @classmethod
    def _check_state_layout_is_complete(cls) -> None:
        """Reject duplicate, missing, or invalid numerical-state mappings at class creation."""
        drawn = tuple(cls.render_from_sim)
        if len(set(drawn)) != len(drawn) or any(i >= len(cls.sim_state_names) for i in drawn):
            raise TypeError(
                f"{cls.__name__}.render_from_sim={drawn} must index distinct entries of "
                f"sim_state_names={cls.sim_state_names}"
            )
        if len(drawn) != int(cls.state_dim):
            raise TypeError(
                f"{cls.__name__}: render_from_sim names {len(drawn)} of the "
                f"{cls.state_dim} packed column(s); every column of the frozen row "
                f"belongs to exactly one dynamical variable"
            )
        if len(set(cls.sim_state_names)) != len(cls.sim_state_names):
            raise TypeError(
                f"{cls.__name__}.sim_state_names={cls.sim_state_names} has a duplicate; "
                f"they become the public state's keys and must be distinct"
            )

    def _sim_columns(self, drawn: bool) -> dict:
        """Select live float64 simulation columns by their presence in render rows."""
        want = set(self.render_from_sim)
        n = self._live_rows
        return {(nm,): self._sim[:n, i : i + 1] for i, nm in enumerate(self.sim_state_names) if (i in want) == drawn}

    def _hidden_state(self) -> dict:
        """Numerical variables omitted from the packed render row."""
        return self._sim_columns(drawn=False)

    def _live_state(self) -> dict:
        """Drawn variables at simulation precision, avoiding a float32 round trip."""
        return self._sim_columns(drawn=True)

    def _state_td(self, sim: torch.Tensor) -> TensorDict:
        """Copy (n, sim_dim) float64 into named state fields without reducing precision."""
        n = int(sim.shape[0])
        return TensorDict(
            {nm: sim[:, i : i + 1].clone() for i, nm in enumerate(self.sim_state_names)},
            batch_size=[n],
            device=self.device,
        )

    def _sim_from_state(self, state) -> torch.Tensor:
        """Validate state and concatenate named fields into an (n, sim_dim) float64 tensor."""
        names = self.sim_state_names
        self._validate_state(state)
        n = int(state.batch_size[0]) if state.batch_size else 1
        try:
            cols = [torch.as_tensor(state[nm], device=self.device, dtype=torch.float64).reshape(n, 1) for nm in names]
        except KeyError as e:
            raise KeyError(
                f"{type(self).__name__}.transition: the state is missing {e.args[0]!r}. "
                f"This env's state keys are {list(names)}."
            ) from e
        return torch.cat(cols, dim=1)

    @torch.no_grad()
    def _render(
        self,
        states=None,
        size: int | None = None,
        supersample: int | None = None,
        variation: dict | None = None,
        style=None,
        out=None,
        backend: str = "auto",
    ) -> torch.Tensor:
        """Render (..., state_dim) rows to native (..., H, W, 3) uint8 frames.

        ``backend`` is ``"auto"``, ``"source"`` (the Gymnasium renderer; CPU and the default appearance only)
        or ``"port"`` (gfxdraw in Numba, Triton or Torch). On CPU, ``"auto"`` and ``"port"`` use Numba when
        numba is installed, except that ``"auto"`` draws batches of fewer than 8 rows of the default appearance with
        the source. Without numba, ``"port"`` uses Torch and ``"auto"`` uses the source for the default appearance
        when pygame is installed and Torch otherwise. On CUDA, ``"auto"`` and ``"port"`` use Triton when triton is
        installed and Torch otherwise. All give the same frames.
        """
        if backend not in ("auto", "source", "port"):
            raise ValueError(f"backend must be 'auto', 'source' or 'port', got {backend!r}")
        H, W = self.native
        want = int(self.size if size is None else size)
        if want != W:
            raise ValueError(
                f"{type(self).__name__} renders only at its reference's native {H}x{W} (got size={want}); "
                f"the reference draws every primitive in fixed pixels. Resample the output instead."
            )
        if supersample not in (None, 1):
            raise ValueError(
                f"{type(self).__name__}: supersample={supersample} is not available; gfxdraw geometry is "
                f"integer and its anti-aliasing uses Wu weights, not coverage."
            )
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        # float64 from here on: the frame depends on trunc() of the vertex coordinates, so float32
        # arithmetic lands on the wrong integer whenever a vertex falls within ~1e-5 of one.
        rows = s.reshape(-1, self.state_dim)
        s = rows.to(torch.float64)
        n = s.shape[0]
        variation, style = _split_appearance(variation, style)
        sty = self.resolve_style(style, variation, batch=n)
        o = None if out is None else out.reshape(-1, H, W, 3)
        if backend == "source" and self.device.type != "cpu":
            raise ValueError(f"{type(self).__name__}: backend='source' renders on CPU only")
        small = backend == "auto" and n < _NUMBA_MIN_ROWS and self._reference_draws(sty)
        if not small and self._use_numba(backend):
            from ..._core.stepping import _threads
            from ...render.cpu_raster import classic as N

            geo = self._varied_geometry(sty, own=style is None and not variation)
            extra = () if geo is None else (geo,)
            # Numba draws the frames in its own threads; idle Torch threads that spin beside them slow both.
            with _threads(1):
                ops = self._ops(s, sty, *extra).finish()
                img = N.render_classic_cpu(ops, out=o)
            return img.view(*lead, H, W, 3) if lead else img
        if self.device.type == "cpu" and not self._use_fused():
            from ..._execution import render_pool

            host = rows.cpu().numpy()
            # The source draws only the default appearance; it is the fastest exact CPU path for it.
            reference = backend != "port" and self._reference_draws(sty)
            if backend == "source" and not reference:
                raise ValueError(f"{type(self).__name__}: backend='source' needs pygame and the default appearance")
            img = render_pool.render_rows(
                self,
                (type(self).__name__, "reference" if reference else "classic"),
                host,
                sty.values.numpy(),
                (H, W, 3),
                "classic-render",
                o,
                factory=render_pool.ReferenceRows(type(self).__name__) if reference else None,
            )
            if img is None and reference:
                if self._reference is None:
                    self._reference = type(self).reference_env()
                frames = torch.from_numpy(type(self).reference_render(self._reference, host))
                img = frames if o is None else o.copy_(frames)
            if img is not None:
                return img.view(*lead, H, W, 3) if lead else img
        geo = self._varied_geometry(sty, own=style is None and not variation)
        extra = () if geo is None else (geo,)
        if self._use_fused():
            from ... import tuning
            from ...render.gfxdraw import classic as F

            tuning.on_render("gfxdraw", W, n, self)
            img = F.render_classic(self._ops(s, sty, *extra).finish(), out=o)
        else:
            cv = _StyleCanvas(self.gfx, n, sty["background.color"])
            cv.aa = sty["rendering.antialias"] > 0.5
            self._draw(cv, s, sty, *extra)
            img = cv.rgb(o)
        return img.view(*lead, H, W, 3) if lead else img

    def _render_frame_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """Draw (B, F, state_dim) rows of the live lanes to (B, F, H, W, 3) frames in one render call."""
        from ..._core.stepping import _padded_buffer

        b, f = rows.shape[:2]
        H, W = self.native
        # Host frames reuse the memory of a previous step's frames once no tensor uses it.
        spares = self.__dict__.setdefault("_frame_buffers", {})
        out = _padded_buffer(spares, "frames", rows.new_empty(0, dtype=torch.uint8), (b, f, H, W, 3))
        style = self._style
        if f == 1 or style.shared:
            return self._render(rows, out=out)
        return self._render(rows, style=style.values.repeat_interleave(f, 0), out=out)

    @property
    def frame_shape(self) -> tuple[int, int]:
        """Return native (H, W); size denotes width and the canvas need not be square."""
        return self.native

    #: physics leaves the frame's geometry reads, each with the value the class-constant geometry encodes.
    geometry_leaves: dict[str, float] = {}
    _geometry_of = None
    _geometry_host = None

    def _varied_geometry(self, sty, own: bool) -> np.ndarray | None:
        """Return host (n, k) float64 geometry-leaf values, or None when every lane draws the constant geometry.

        The answer is cached per installed variation, so rendering the env's own appearance reads
        nothing back from the device after the first frame.
        """
        if not self.geometry_leaves:
            return None
        src = self._style if own else sty
        if self._geometry_of is not src:
            host = torch.stack([src[p].reshape(-1) for p in self.geometry_leaves], 1).cpu().numpy()
            const = np.asarray(list(self.geometry_leaves.values()), np.float32)
            self._geometry_host = None if (host == const).all() else host.astype(np.float64)
            self._geometry_of = src
        return self._geometry_host

    def _reference_draws(self, sty) -> bool:
        """Return whether the Gymnasium reference can draw ``sty``: the default appearance, with pygame installed."""
        if self._has_pygame is None:
            import importlib.util

            self._has_pygame = importlib.util.find_spec("pygame") is not None
        if not self._has_pygame:
            return False
        if self._reference_style is None:
            # The appearance the source draws: the defaults of a new variation space, not the env's current rows.
            fresh = torch.as_tensor(self.make_variation_space().pack(), dtype=torch.float32, device=self.device)
            self._reference_style = fresh.reshape(1, -1)
        return bool((sty.values == self._reference_style).all())

    def _use_numba(self, backend: str) -> bool:
        """Select the Numba twin of the fused kernel on CPU for ``"auto"`` and ``"port"`` when numba imports."""
        if backend == "source" or self.device.type != "cpu" or not (self.fused_capable and self.use_fused_render):
            return False
        from ...render import cpu_raster

        return cpu_raster.available()

    def _use_fused(self) -> bool:
        """Select the fused renderer only when enabled and CUDA/Triton are available."""
        if not (self.fused_capable and self.use_fused_render and self.device.type == "cuda"):
            return False
        from ...render.gfxdraw import classic as F

        return F.HAVE_TRITON

    def _draw(self, cv, s, sty) -> None:
        raise NotImplementedError

    # The per-primitive and fused descriptions are independent implementations checked for pixel equality.

    #: radii this environment's discs use. Static geometry, so the disc tables may be cached on it.
    disc_radii: tuple[int, ...] = ()
    #: False when the fused kernel does not describe this class's frame.
    fused_capable: bool = False

    def _ops(self, s, sty):
        """Describe the ordered primitives for the fused renderer."""
        raise NotImplementedError

    def _ops_begin(self, s, sty):
        from ...render.gfxdraw import classic as F

        H, W = self.native
        ops = F.Ops(s.shape[0], H, W, self.device)
        if self._disc_tables is None and self.disc_radii:
            self._disc_tables = F.DiscTables(self.disc_radii, self.device)
        ops.discs = self._disc_tables
        ops.aa = (sty["rendering.antialias"] > 0.5).to(torch.int32)
        ops.bg = _u8(sty["background.color"]).to(torch.int32)
        return ops

    @staticmethod
    def _colors(**kw) -> dict:
        return {k: RGBBox(init_value=v) for k, v in kw.items()}

    def _aa_space(self) -> Dict:
        """Return the per-environment antialiasing switch, enabled by default.

        Disabling it changes boundary pixels. Mixed batches still evaluate the Wu
        pass and discard blends for rows that disable it.
        """
        return Dict({"antialias": Discrete(2, init_value=1)})

    @classmethod
    def reference_env(cls, size: int | None = None):
        """Construct the official Gymnasium renderer at its fixed native resolution."""
        import gymnasium

        return gymnasium.make(cls.reference_id, render_mode="rgb_array").unwrapped

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        out = []
        for st in np.asarray(states):
            cls.reference_set_state(env, st)
            out.append(env.render().copy())
        return np.stack(out)

    @classmethod
    def reference_set_state(cls, env, st) -> None:
        raise NotImplementedError


@register
class CartPole(ClassicControlSemantics, _Classic):
    """CartPole-v1 geometry from [x, theta].

    Draw the cart, pole, axle, then the track over them. Axis-aligned cart edges
    use solid SDL line rules; the rotated pole uses Wu antialiasing.
    """

    name = "cartpole"
    state_dim = 2
    state_names = ("x", "theta")
    native = (400, 600)
    default_size = 600
    reference_id = "CartPole-v1"
    CLASSIC_DOMAIN = "cartpole"

    SCALE = 600 / 4.8  # screen_width / (2 * x_threshold)
    POLELEN = 125.0  # scale * 2 * length
    CARTY = 100.0
    AXLEOFF = 30.0 / 4.0
    AXLE_R = 5

    sim_state_names = ("x", "x_dot", "theta", "theta_dot")
    render_from_sim = (0, 2)
    action_low, action_high = 0.0, 2.0
    LENGTH = 0.5

    def make_variation_space(self) -> Dict:
        return self._with_physics_variation(
            Dict(
                {
                    "cart": Dict(self._colors(color=(0, 0, 0))),
                    "pole": Dict(self._colors(color=(202, 152, 101))),
                    "axle": Dict(self._colors(color=(129, 132, 203))),
                    "track": Dict(self._colors(color=(0, 0, 0))),
                    "background": Dict(self._colors(color=WHITE)),
                    "rendering": self._aa_space(),
                }
            )
        )

    # Only row extent is bounded for every pose; horizontal cart motion can span the full canvas.
    WIN_CART = (85, 115)
    WIN_POLE = (
        int(CARTY + AXLEOFF - math.hypot(5.0, POLELEN - 5.0)),
        int(CARTY + AXLEOFF + math.hypot(5.0, POLELEN - 5.0)),
    )
    #: Per-edge step bounds: each quad has two long and two short sides, and the Wu pass costs its
    #: bound in steps regardless of the actual edge length.
    KMAX_CART = (32, 52, 32, 52)
    KMAX_POLE = (132, 16, 132, 16)

    geometry_leaves = {"physics.length": LENGTH}

    def _pole(self, B, sty, geo):
        """Pole quad before rotation, its per-edge step bounds and its row window."""
        g = self.gfx
        if geo is None:
            ploc = (
                g.verts([(-5.0, -5.0), (-5.0, self.POLELEN - 5.0), (5.0, self.POLELEN - 5.0), (5.0, -5.0)])
                .view(1, 4, 2)
                .expand(B, 4, 2)
            )
            return ploc, self.KMAX_POLE, self.WIN_POLE
        ploc = g.verts([(-5.0, -5.0), (-5.0, 0.0), (5.0, 0.0), (5.0, -5.0)]).view(1, 4, 2).repeat(B, 1, 1)
        ploc[:, 1:3, 1] = (self.SCALE * (2 * sty["physics.length"].to(torch.float64)) - 5.0).view(B, 1)
        polelen = self.SCALE * (2 * float(geo.max()))
        reach = math.hypot(5.0, polelen - 5.0)
        k = int(polelen) + 8
        return ploc, (k, 16, k, 16), (int(self.CARTY + self.AXLEOFF - reach), int(self.CARTY + self.AXLEOFF + reach))

    def _draw(self, cv, s, sty, geo=None) -> None:
        g, B = self.gfx, s.shape[0]
        H, W = self.native
        bi = g.batch_index(B)
        cartx = s[:, 0] * self.SCALE + 300.0
        carty = torch.full_like(cartx, self.CARTY)

        ctr = torch.stack([cartx, carty], -1).view(B, 1, 2)
        cart = g.verts([(-25.0, -15.0), (-25.0, 15.0), (25.0, 15.0), (25.0, -15.0)]).view(1, 4, 2) + ctr
        _poly(cv, cart, sty["cart.color"], bi, self.KMAX_CART, _box(H, W, *self.WIN_CART, 0, W - 1))

        ploc, kmax, win = self._pole(B, sty, geo)
        pole = G.rotate(ploc, (-s[:, 1]).view(B, 1)) + torch.stack([cartx, carty + self.AXLEOFF], -1).view(B, 1, 2)
        _poly(cv, pole, sty["pole.color"], bi, kmax, _box(H, W, *win, 0, W - 1))

        cxi = torch.trunc(cartx).to(torch.int64)
        cyi = torch.full_like(cxi, int(self.CARTY + self.AXLEOFF))
        _circ(cv, cxi, cyi, self.AXLE_R, sty["axle.color"], bi, g)

        _row(cv, int(self.CARTY), sty["track.color"])

    disc_radii = (AXLE_R,)
    fused_capable = True

    def _ops(self, s, sty, geo=None):
        """Describe the ordered primitives for the fused renderer."""
        g, B = self.gfx, s.shape[0]
        ops = self._ops_begin(s, sty)
        cartx = s[:, 0] * self.SCALE + 300.0
        carty = torch.full_like(cartx, self.CARTY)
        ctr = torch.stack([cartx, carty], -1).view(B, 1, 2)
        ops.polygon(
            g.verts([(-25.0, -15.0), (-25.0, 15.0), (25.0, 15.0), (25.0, -15.0)]).view(1, 4, 2) + ctr,
            sty["cart.color"],
        )
        ploc = self._pole(B, sty, geo)[0]
        ops.polygon(
            G.rotate(ploc, (-s[:, 1]).view(B, 1)) + torch.stack([cartx, carty + self.AXLEOFF], -1).view(B, 1, 2),
            sty["pole.color"],
        )
        cxi = torch.trunc(cartx).to(torch.int64)
        ops.disc(cxi, torch.full_like(cxi, int(self.CARTY + self.AXLEOFF)), self.AXLE_R, sty["axle.color"])
        ops.hline(int(self.CARTY), sty["track.color"])
        return ops

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample the episode position/angle range, including clipped edge geometry."""
        rng = rng or self.rng
        return np.stack([rng.uniform(-2.4, 2.4, batch), rng.uniform(-0.2095, 0.2095, batch)], 1).astype(np.float32)

    @classmethod
    def reference_set_state(cls, env, st) -> None:
        env.state = np.array([st[0], 0.0, st[1], 0.0])


@register
class Pendulum(ClassicControlSemantics, _Classic):
    """Pendulum-v1 geometry from [theta, torque].

    The torque arrow uses last_u, which is absent from the observation. Zero torque
    and an unset last_u both draw no arrow.
    """

    name = "pendulum"
    state_dim = 2
    state_names = ("theta", "torque")
    native = (500, 500)
    default_size = 500
    reference_id = "Pendulum-v1"
    CLASSIC_DOMAIN = "pendulum"

    SCALE = 500 / 4.4
    OFF = 250
    ROD_R = 11  # int(0.2 * scale / 2)
    AXLE_R = 5  # int(0.05 * scale)
    MAX_SPRITE = 113  # int(scale * max_torque / 2), max_torque = 2
    WIN = 113  # sprite window: every size fits rows/cols 194..306

    # The render row draws last_u (torque), not velocity.
    sim_state_names = ("theta", "theta_dot", "torque")
    render_from_sim = (0, 2)
    action_low, action_high = -2.0, 2.0

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._sprite = _torque_sprites(self.device)

    def make_variation_space(self) -> Dict:
        return self._with_physics_variation(
            Dict(
                {
                    "rod": Dict(self._colors(color=(204, 77, 77))),
                    "axle": Dict(self._colors(color=(0, 0, 0))),
                    "background": Dict(self._colors(color=WHITE)),
                    "rendering": self._aa_space(),
                }
            )
        )

    # Rotation bounds cover the rod and cap for every angle.
    _REACH = math.hypot(SCALE, 0.1 * SCALE)
    WIN_ROD = (int(OFF - _REACH), int(OFF + _REACH))
    KMAX_ROD = (28, 120, 28, 120)  # the quad starts on its short side

    def _draw(self, cv, s, sty) -> None:
        g, B = self.gfx, s.shape[0]
        H, W = self.native
        bi = g.batch_index(B)
        th, off = s[:, 0], self.OFF
        rl, rw = 1.0 * self.SCALE, 0.2 * self.SCALE
        ang = (th + math.pi / 2).view(B, 1)
        rod = sty["rod.color"]

        loc = g.verts([(0.0, -rw / 2), (0.0, rw / 2), (rl, rw / 2), (rl, -rw / 2)]).view(1, 4, 2).expand(B, 4, 2)
        _poly(cv, G.rotate(loc, ang) + off, rod, bi, self.KMAX_ROD, _box(H, W, *self.WIN_ROD, *self.WIN_ROD))

        ctr = torch.full((B,), off, dtype=torch.int64, device=self.device)
        _circ(cv, ctr, ctr, self.ROD_R, rod, bi, g)
        end = G.rotate(g.verts([(rl, 0.0)]).view(1, 1, 2).expand(B, 1, 2), ang)[:, 0, :] + off
        _circ(
            cv, torch.trunc(end[:, 0]).to(torch.int64), torch.trunc(end[:, 1]).to(torch.int64), self.ROD_R, rod, bi, g
        )

        self._blit_torque(cv, s[:, 1])
        _circ(cv, ctr, ctr, self.AXLE_R, sty["axle.color"], bi, g)

    disc_radii = (ROD_R, AXLE_R)
    fused_capable = True

    def _ops(self, s, sty):
        g, B = self.gfx, s.shape[0]
        ops = self._ops_begin(s, sty)
        ops.sprite = self._sprite
        th, off = s[:, 0], self.OFF
        rl, rw = 1.0 * self.SCALE, 0.2 * self.SCALE
        ang = (th + math.pi / 2).view(B, 1)
        rod = sty["rod.color"]
        loc = g.verts([(0.0, -rw / 2), (0.0, rw / 2), (rl, rw / 2), (rl, -rw / 2)]).view(1, 4, 2).expand(B, 4, 2)
        ops.polygon(G.rotate(loc, ang) + off, rod)
        ctr = torch.full((B,), off, dtype=torch.int64, device=self.device)
        ops.disc(ctr, ctr, self.ROD_R, rod)
        end = G.rotate(g.verts([(rl, 0.0)]).view(1, 1, 2).expand(B, 1, 2), ang)[:, 0, :] + off
        ops.disc(torch.trunc(end[:, 0]).to(torch.int64), torch.trunc(end[:, 1]).to(torch.int64), self.ROD_R, rod)
        u = s[:, 1]
        n = torch.trunc(self.SCALE * u.abs() / 2).to(torch.int64).clamp(0, self.MAX_SPRITE)
        ops.blit(250 - self.WIN // 2, self.WIN, (u > 0).to(torch.int64), n, sty["background.color"])
        ops.disc(ctr, ctr, self.AXLE_R, sty["axle.color"])
        return ops

    def _blit_torque(self, cv, u) -> None:
        """Gather a cached torque sprite and alpha-blend it below the axle.

        All reachable sprite sizes fit one fixed window around the canvas center.
        """
        B = u.shape[0]
        n = torch.trunc(self.SCALE * u.abs() / 2).to(torch.int64).clamp(0, self.MAX_SPRITE)
        flip = (u > 0).to(torch.int64)
        w = self.WIN
        i = torch.arange(w, device=self.device)
        off = (w // 2) - torch.div(n, 2, rounding_mode="floor")
        ly = i.view(1, -1) - off.view(-1, 1)  # (B,w)
        lx = ly
        valid = ((ly >= 0) & (ly < n.view(-1, 1))).view(B, w, 1) & ((lx >= 0) & (lx < n.view(-1, 1))).view(B, 1, w)
        rgba = self._sprite[
            flip.view(B, 1, 1),
            n.view(B, 1, 1),
            ly.clamp(0, self.MAX_SPRITE - 1).view(B, w, 1),
            lx.clamp(0, self.MAX_SPRITE - 1).view(B, 1, w),
        ].to(torch.int32)
        y0 = 250 - w // 2
        dst = cv.buf[:, :, y0 : y0 + w, y0 : y0 + w].to(torch.int32)  # (3,B,w,w)
        a = torch.where(valid, rgba[..., 3], torch.zeros_like(rgba[..., 3])).unsqueeze(0)
        src = rgba[..., :3].permute(3, 0, 1, 2)
        cv.buf[:, :, y0 : y0 + w, y0 : y0 + w] = G.blit_over(dst, src, a).to(torch.uint8)

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample a full turn and the supported torque range."""
        rng = rng or self.rng
        return np.stack([rng.uniform(-np.pi, np.pi, batch), rng.uniform(-2.0, 2.0, batch)], 1).astype(np.float32)

    @classmethod
    def reference_set_state(cls, env, st) -> None:
        env.state = np.array([st[0], 0.0])
        # Keep float32 last_u: its dtype affects truncated torque-sprite dimensions near integer boundaries.
        env.last_u = np.float32(st[1])


@register
class Acrobot(ClassicControlSemantics, _Classic):
    """Acrobot-v1 geometry from [theta1, theta2].

    Match the source coordinate swap for the second link origin and truncate the
    ground-line coordinate rather than rounding it.
    """

    name = "acrobot"
    state_dim = 2
    state_names = ("theta1", "theta2")
    native = (500, 500)
    default_size = 500
    reference_id = "Acrobot-v1"
    CLASSIC_DOMAIN = "acrobot"

    SCALE = 500 / 4.4
    OFF = 250.0
    JOINT_R = 11  # int(0.1 * scale)

    sim_state_names = ("theta1", "theta2", "dtheta1", "dtheta2")
    render_from_sim = (0, 1)
    action_low, action_high = 0.0, 3.0
    L1 = 1.0

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._varied_discs = {}
        self._varied_host = None

    def make_variation_space(self) -> Dict:
        return self._with_physics_variation(
            Dict(
                {
                    "link": Dict(self._colors(color=(0, 204, 204))),
                    "joint": Dict(self._colors(color=(204, 204, 0))),
                    "ground": Dict(self._colors(color=(0, 0, 0))),
                    "background": Dict(self._colors(color=WHITE)),
                    "rendering": self._aa_space(),
                }
            )
        )

    # Joint displacement and link radius bound each link window for every pose.
    _REACH = math.hypot(SCALE, 0.1 * SCALE)
    WIN_LINK = ((int(OFF - _REACH), int(OFF + _REACH)), (int(OFF - SCALE - _REACH), int(OFF + SCALE + _REACH)))
    KMAX_LINK = (28, 120, 28, 120)  # the quad starts on its short side

    geometry_leaves = {"physics.link_length_1": L1, "physics.link_length_2": L1}

    def _varied_bounds(self, geo):
        """Joint radii present, and each link's per-edge step bounds and window, over every lane."""
        if self._varied_host is not None and self._varied_host[0] is geo:
            return self._varied_host[1]
        H, W = self.native
        off = self.OFF
        sc = 500 / ((geo[:, 0] + geo[:, 1] + 0.2) * 2)
        radii = tuple(int(v) for v in np.unique(np.trunc(0.1 * sc)))
        bounds = []
        for i in range(2):
            span = geo[:, i] * sc
            reach = float((geo[:, 0] * sc * i + np.hypot(span, 0.1 * sc)).max())
            kl, ks = int(span.max()) + 8, int(0.2 * sc.max()) + 8
            bounds.append(((ks, kl, ks, kl), _box(H, W, off - reach, off + reach, off - reach, off + reach)))
        self._varied_host = (geo, (radii, bounds))
        return radii, bounds

    def _varied(self, s, sty):
        """Per-lane ground row and joint radius, then both links' (B, 2, ...) quads and joint centers.

        The whole picture scales with 500 / (2 * (l1 + l2 + 0.2)), as AcrobotEnv.render draws it.
        """
        B, W = s.shape[0], self.native[1]
        ln = torch.stack([sty["physics.link_length_1"], sty["physics.link_length_2"]], 1).to(torch.float64)
        sc, off = 500 / ((ln[:, 0] + ln[:, 1] + 0.2) * 2), self.OFF
        t1, t2 = s[:, 0], s[:, 1]
        ground = (
            torch.trunc(1 * sc + off).to(torch.int64),
            torch.trunc(-2.2 * sc + off).to(torch.int64).clamp_min(0),
            torch.trunc(2.2 * sc + off).to(torch.int64).clamp_max(W - 1),
        )
        jx = torch.stack([torch.zeros_like(t1), ln[:, 0] * torch.sin(t1) * sc], 1) + off
        jy = torch.stack([torch.zeros_like(t1), -ln[:, 0] * torch.cos(t1) * sc], 1) + off
        size = torch.stack([ln * sc.view(B, 1), (0.1 * sc).view(B, 1).expand(B, 2)], -1).view(B, 2, 1, 2)
        quad = self.gfx.verts([(0.0, -1.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0)]).view(1, 1, 4, 2) * size
        th = torch.stack([t1 - math.pi / 2, t1 + t2 - math.pi / 2], 1)
        quads = G.rotate(quad, th.view(B, 2, 1)) + torch.stack([jx, jy], -1).view(B, 2, 1, 2)
        i64 = torch.int64
        return ground, torch.trunc(0.1 * sc).to(i64), quads, torch.trunc(jx).to(i64), torch.trunc(jy).to(i64)

    def _draw(self, cv, s, sty, geo=None) -> None:
        if geo is not None:
            return self._draw_varied(cv, s, sty, geo)
        g, B = self.gfx, s.shape[0]
        H, W = self.native
        bi = g.batch_index(B)
        sc, off = self.SCALE, self.OFF
        t1, t2 = s[:, 0], s[:, 1]

        _row(cv, int(1 * sc + off), sty["ground.color"])

        jx = [torch.zeros_like(t1), torch.sin(t1) * sc]
        jy = [torch.zeros_like(t1), -torch.cos(t1) * sc]
        loc = (
            g.verts([(0.0, -0.1 * sc), (0.0, 0.1 * sc), (sc, 0.1 * sc), (sc, -0.1 * sc)]).view(1, 4, 2).expand(B, 4, 2)
        )
        lc, jc = sty["link.color"], sty["joint.color"]
        for i, th in enumerate([t1 - math.pi / 2, t1 + t2 - math.pi / 2]):
            ctr = torch.stack([jx[i] + off, jy[i] + off], -1).view(B, 1, 2)
            _poly(
                cv,
                G.rotate(loc, th.view(B, 1)) + ctr,
                lc,
                bi,
                self.KMAX_LINK,
                _box(H, W, *self.WIN_LINK[i], *self.WIN_LINK[i]),
            )
            _circ(
                cv,
                torch.trunc(jx[i] + off).to(torch.int64),
                torch.trunc(jy[i] + off).to(torch.int64),
                self.JOINT_R,
                jc,
                bi,
                g,
            )

    def _draw_varied(self, cv, s, sty, geo) -> None:
        g, B = self.gfx, s.shape[0]
        bi = g.batch_index(B)
        radii, bounds = self._varied_bounds(geo)
        (gy, gx0, gx1), r, quads, jx, jy = self._varied(s, sty)
        hit = (g.xs.view(1, -1) >= gx0.view(B, 1)) & (g.xs.view(1, -1) <= gx1.view(B, 1))
        row = cv.buf[:, bi, gy]
        cv.buf[:, bi, gy] = torch.where(hit.unsqueeze(0), _u8(sty["ground.color"]).t().unsqueeze(-1), row)
        lc, jc = sty["link.color"], sty["joint.color"]
        # Each radius draws only its own lanes, so a batch spanning many radii costs about one disc pass.
        lanes = [(rad, at) for rad in radii if len(at := (r == rad).nonzero().view(-1))]
        for i, (kmax, win) in enumerate(bounds):
            _poly(cv, quads[:, i], lc, bi, kmax, win)
            cv.use(jc)
            for rad, at in lanes:
                G.circle(
                    cv, jx[at, i], jy[at, i], rad, None, None, at, None, g.stencil(rad), disc=g.disc(rad), lanes=at
                )

    disc_radii = (JOINT_R,)
    fused_capable = True

    def _ops(self, s, sty, geo=None):
        if geo is not None:
            return self._ops_varied(s, sty, geo)
        g, B = self.gfx, s.shape[0]
        ops = self._ops_begin(s, sty)
        sc, off = self.SCALE, self.OFF
        t1, t2 = s[:, 0], s[:, 1]
        ops.hline(int(1 * sc + off), sty["ground.color"])
        jx = [torch.zeros_like(t1), torch.sin(t1) * sc]
        jy = [torch.zeros_like(t1), -torch.cos(t1) * sc]
        loc = (
            g.verts([(0.0, -0.1 * sc), (0.0, 0.1 * sc), (sc, 0.1 * sc), (sc, -0.1 * sc)]).view(1, 4, 2).expand(B, 4, 2)
        )
        lc, jc = sty["link.color"], sty["joint.color"]
        for i, th in enumerate([t1 - math.pi / 2, t1 + t2 - math.pi / 2]):
            ctr = torch.stack([jx[i] + off, jy[i] + off], -1).view(B, 1, 2)
            ops.polygon(G.rotate(loc, th.view(B, 1)) + ctr, lc)
            ops.disc(
                torch.trunc(jx[i] + off).to(torch.int64), torch.trunc(jy[i] + off).to(torch.int64), self.JOINT_R, jc
            )
        return ops

    def _ops_varied(self, s, sty, geo):
        from ...render.gfxdraw import classic as F

        ops = self._ops_begin(s, sty)
        radii, _ = self._varied_bounds(geo)
        (gy, gx0, gx1), r, quads, jx, jy = self._varied(s, sty)
        if radii not in self._varied_discs:
            self._varied_discs[radii] = F.DiscTables(radii, self.device)
        ops.discs = self._varied_discs[radii]
        ops.hline(gy, sty["ground.color"], gx0, gx1)
        for i in range(2):
            ops.polygon(quads[:, i], sty["link.color"])
            ops.disc(jx[:, i], jy[:, i], r, sty["joint.color"])
        return ops

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample both link angles over a full turn."""
        rng = rng or self.rng
        return rng.uniform(-np.pi, np.pi, (batch, 2)).astype(np.float32)

    @classmethod
    def reference_set_state(cls, env, st) -> None:
        env.state = np.array([st[0], st[1], 0.0, 0.0])


@register
class MountainCar(ClassicControlSemantics, _Classic):
    """MountainCar-v0 geometry from [position].

    Bake the static pygame terrain per appearance. Draw the car and then the flag;
    the flag cannot be part of the background because it overlaps the car.
    """

    name = "mountain-car"
    state_dim = 1
    state_names = ("position",)
    native = (400, 600)
    default_size = 600
    reference_id = "MountainCar-v0"
    CLASSIC_DOMAIN = "mountain_car"

    MINP, MAXP = -1.2, 0.6
    # Compute world width as MAXP - MINP; literal 1.8 rounds differently and moves the flag by a pixel.
    SCALE = 600 / (0.6 - (-1.2))
    WHEEL_R = 8  # int(carheight / 2.5)
    # The discrete and continuous variants place their flags at different goal positions.
    GOALP = 0.5

    # Velocity affects dynamics but is absent from the render row.
    sim_state_names = ("position", "velocity")
    render_from_sim = (0,)
    action_low, action_high = 0.0, 3.0

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.flagx = int((self.GOALP - self.MINP) * self.SCALE)
        self.flagy1 = int(_mc_height(self.GOALP) * self.SCALE)
        self.flagy2 = self.flagy1 + 50

    def make_variation_space(self) -> Dict:
        return self._with_physics_variation(
            Dict(
                {
                    "car": Dict(self._colors(color=(0, 0, 0))),
                    "wheel": Dict(self._colors(color=(128, 128, 128))),
                    "terrain": Dict(self._colors(color=(0, 0, 0))),
                    "flag": Dict(self._colors(color=(204, 204, 0), pole_color=(0, 0, 0))),
                    "background": Dict(self._colors(color=WHITE)),
                    "rendering": self._aa_space(),
                }
            )
        )

    def _terrain_images(self, bg, terrain):
        """Return the distinct baked (U, 3, H, W) terrain images and the (B,) image index of each lane.

        Pygame quantizes each color blend, so coverage derived from one baked color
        pair cannot reproduce arbitrary other pairs exactly.
        """
        key = torch.cat([bg, terrain], 1).to(torch.uint8).cpu().numpy()
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        imgs = torch.stack(
            [
                torch.as_tensor(
                    _bake_terrain(tuple(int(c) for c in u[:3]), tuple(int(c) for c in u[3:])),
                    dtype=torch.uint8,
                    device=self.device,
                ).permute(2, 0, 1)
                for u in uniq
            ]
        )
        return imgs, torch.as_tensor(inv.reshape(-1), device=self.device)

    def _terrain_base(self, bg, terrain) -> torch.Tensor:
        """Return a planar static background, baked per distinct background/terrain color pair."""
        imgs, inv = self._terrain_images(bg, terrain)
        # Keep the background planar so a shared appearance broadcasts without transposing each frame.
        if len(imgs) == 1:
            return imgs[0].unsqueeze(1)  # (3,1,H,W)
        return imgs.index_select(0, inv).transpose(0, 1)  # (3,B,H,W)

    # The sine bounds car rows for the entire track; flag coordinates are fixed and disc stencils bound wheels.
    _CARY = (10.0 + 0.1 * SCALE, 10.0 + 1.0 * SCALE)
    WIN_CAR = (int(_CARY[0] - math.hypot(20.0, 20.0)), int(_CARY[1] + math.hypot(20.0, 20.0)))
    KMAX_CAR = (26, 46, 26, 46)
    KMAX_FLAG = (14, 30, 30)

    def _draw(self, cv, s, sty) -> None:
        g, B = self.gfx, s.shape[0]
        H, W = self.native
        bi = g.batch_index(B)
        cv.fill_from(self._terrain_base(sty["background.color"], sty["terrain.color"]))

        pos = s[:, 0]
        ang = torch.cos(3 * pos).view(B, 1)  # the source rotates by cos(3*pos), in radians
        ctr = torch.stack(
            [(pos - self.MINP) * self.SCALE, 10.0 + (torch.sin(3 * pos) * 0.45 + 0.55) * self.SCALE], -1
        ).view(B, 1, 2)
        loc = g.verts([(-20.0, 0.0), (-20.0, 20.0), (20.0, 20.0), (20.0, 0.0)]).view(1, 4, 2).expand(B, 4, 2)
        _poly(cv, G.rotate(loc, ang) + ctr, sty["car.color"], bi, self.KMAX_CAR, _box(H, W, *self.WIN_CAR, 0, W - 1))

        wl = g.verts([(10.0, 0.0), (-10.0, 0.0)]).view(1, 2, 2).expand(B, 2, 2)
        wr = G.rotate(wl, ang) + ctr
        wc = sty["wheel.color"]
        for j in range(2):
            _circ(
                cv,
                torch.trunc(wr[:, j, 0]).to(torch.int64),
                torch.trunc(wr[:, j, 1]).to(torch.int64),
                self.WHEEL_R,
                wc,
                bi,
                g,
            )

        cv.buf[:, :, self.flagy1 : self.flagy2 + 1, self.flagx] = _u8(sty["flag.pole_color"]).t().unsqueeze(-1)
        tri = (
            g.verts(
                [
                    (float(self.flagx), float(self.flagy2)),
                    (float(self.flagx), float(self.flagy2 - 10)),
                    (float(self.flagx + 25), float(self.flagy2 - 5)),
                ]
            )
            .view(1, 3, 2)
            .expand(B, 3, 2)
        )
        _poly(
            cv,
            tri,
            sty["flag.color"],
            bi,
            self.KMAX_FLAG,
            _box(H, W, self.flagy2 - 10, self.flagy2, self.flagx, self.flagx + 25),
        )

    disc_radii = (WHEEL_R,)
    fused_capable = True

    def _ops(self, s, sty):
        g, B = self.gfx, s.shape[0]
        ops = self._ops_begin(s, sty)
        if self.device.type == "cpu":
            imgs, ops.plate_idx = self._terrain_images(sty["background.color"], sty["terrain.color"])
            ops.plate = imgs.transpose(0, 1)
        else:
            ops.plate = self._terrain_base(sty["background.color"], sty["terrain.color"])
        pos = s[:, 0]
        ang = torch.cos(3 * pos).view(B, 1)
        ctr = torch.stack(
            [(pos - self.MINP) * self.SCALE, 10.0 + (torch.sin(3 * pos) * 0.45 + 0.55) * self.SCALE], -1
        ).view(B, 1, 2)
        loc = g.verts([(-20.0, 0.0), (-20.0, 20.0), (20.0, 20.0), (20.0, 0.0)]).view(1, 4, 2).expand(B, 4, 2)
        ops.polygon(G.rotate(loc, ang) + ctr, sty["car.color"])
        wl = g.verts([(10.0, 0.0), (-10.0, 0.0)]).view(1, 2, 2).expand(B, 2, 2)
        wr = G.rotate(wl, ang) + ctr
        wc = sty["wheel.color"]
        for j in range(2):
            ops.disc(
                torch.trunc(wr[:, j, 0]).to(torch.int64), torch.trunc(wr[:, j, 1]).to(torch.int64), self.WHEEL_R, wc
            )
        ops.vline(self.flagx, self.flagy1, self.flagy2, sty["flag.pole_color"])
        ops.polygon(
            g.verts(
                [
                    (float(self.flagx), float(self.flagy2)),
                    (float(self.flagx), float(self.flagy2 - 10)),
                    (float(self.flagx + 25), float(self.flagy2 - 5)),
                ]
            )
            .view(1, 3, 2)
            .expand(B, 3, 2),
            sty["flag.color"],
        )
        return ops

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample the full track, including the right-edge clipping region."""
        rng = rng or self.rng
        return rng.uniform(self.MINP, self.MAXP, (batch, 1)).astype(np.float32)

    @classmethod
    def reference_set_state(cls, env, st) -> None:
        env.state = np.array([st[0], 0.0])


@register
class MountainCarContinuous(MountainCar):
    """Continuous MountainCar geometry, with the flag at 0.45 instead of 0.5."""

    name = "mountain-car-continuous"
    reference_id = "MountainCarContinuous-v0"
    CLASSIC_DOMAIN = "mountain_car_continuous"
    GOALP = 0.45
    action_low, action_high = -1.0, 1.0


def _mc_height(x):
    """Evaluate source terrain height with NumPy sine before integer pixel conversion."""
    return float(np.sin(3 * x) * 0.45 + 0.55)


# Below this many rows the in-process source draws faster than building the Numba records.
_NUMBA_MIN_ROWS = 8

# Bound the color-keyed cache because randomized appearance can create a new background every call.
_TERRAIN_CACHE_MAX = 512
_TERRAIN_CACHE = _RES.BoundedCache(_TERRAIN_CACHE_MAX, name="classic.terrain")
_SPRITE_CACHE: dict = {}


def _bake_terrain(bg, terrain) -> np.ndarray:
    """Bake pygame terrain to pre-flip (400, 600, 3) uint8."""
    key = (tuple(bg), tuple(terrain))
    hit = _TERRAIN_CACHE.get(key)
    if hit is not None:
        return hit
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.init()
    surf = pygame.Surface((600, 400))
    surf.fill(tuple(bg))
    sc = MountainCar.SCALE
    xs = np.linspace(MountainCar.MINP, MountainCar.MAXP, 100)
    ys = np.sin(3 * xs) * 0.45 + 0.55
    pygame.draw.aalines(
        surf, points=list(zip((xs - MountainCar.MINP) * sc, ys * sc)), closed=False, color=tuple(terrain)
    )
    return _TERRAIN_CACHE.put(key, np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2)).copy())


def _torque_sprites(device) -> torch.Tensor:
    """Build zero-padded torque sprites indexed by [flip, size] on the target device.

    Smoothscale truncates the requested extent: both flip variants have 114
    reachable sizes. Return (2, 114, 113, 113, 4) uint8.
    """
    key = str(device)
    if key not in _SPRITE_CACHE:
        import os

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        from os import path

        import gymnasium.envs.classic_control as cc
        import pygame

        pygame.init()
        img = pygame.image.load(path.join(path.dirname(cc.__file__), "assets/clockwise.png"))
        m = Pendulum.MAX_SPRITE
        buf = np.zeros((2, m + 1, m, m, 4), np.uint8)
        for n in range(1, m + 1):
            for fl in (0, 1):
                spr = pygame.transform.flip(pygame.transform.smoothscale(img, (n, n)), bool(fl), True)
                buf[fl, n, :n, :n, :3] = np.transpose(np.array(pygame.surfarray.pixels3d(spr)), (1, 0, 2))
                buf[fl, n, :n, :n, 3] = np.transpose(np.array(pygame.surfarray.pixels_alpha(spr)), (1, 0))
        _SPRITE_CACHE[key] = torch.as_tensor(buf, device=device)
    return _SPRITE_CACHE[key]
