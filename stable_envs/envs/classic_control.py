"""Gymnasium classic control — the family where the frame is bit-identical, not merely close.

Reference: ``gymnasium.envs.classic_control`` (CartPole-v1, Pendulum-v1, Acrobot-v1, MountainCar-v0,
MountainCarContinuous-v0). Measured against the official renderer on 256 states each, half from real
rollouts and half spread across the full render range: **0 differing pixels out of 250.9 M, 1024/1024
frames bit-identical**, at the reference's own resolution. Not "interior pixels exact" — every pixel,
including the anti-aliased rims.

That is achievable here and not on PushT for a structural reason, not because anyone tried harder.
These renderers draw through ``pygame.gfxdraw``, whose primitives take **Sint16** vertices: every float
coordinate is truncated to an integer before SDL_gfx sees it, so the whole reference is integer
arithmetic on integer geometry and there is no sub-pixel coverage to approximate. The primitive layer
that redoes that arithmetic is ``stable_envs.render.gfxdraw``, and its docstring carries the six rules
this depends on. Consequently these envs do NOT build a ``Scene2D``: the SDF rasterizer approximates
coverage by supersampling, which is the right tool for an anti-aliased reference and the wrong one for
an integer-exact one. They override ``render`` instead.

STATE IS THE COMPACT RENDER ROW, NOT THE GYM OBSERVATION. Velocities never reach the screen, so they
are not in that row: CartPole renders from ``(x, theta)`` although its observation is 4-D, Acrobot from
``(theta1, theta2)`` although its observation is 6-D, and MountainCar from ``position`` alone — one
number for a 400x600 frame. Public State still carries the full dynamical vector (velocities included)
so ``transition`` can be pure. Compact render state is what ``pack_render_state`` produces and is
not a lossless dump of that State. This is the compact-state claim at its sharpest, and it is checked rather
than asserted: the conformance suite fails an env whose render row has a dimension the renderer ignores.

Pendulum is the interesting case. Its second dimension is ``torque``, which is **not in the
observation** — ``render()`` blits a torque-arrow PNG scaled by ``|last_u|``, a value ``step()`` sets
and ``reset()`` clears. Two measurements make it a clean state dimension anyway:

  * ``pygame.transform.smoothscale`` truncates its target size, so the sprite is one of exactly
    **114 sizes x 2 flips = 228 bitmaps (3.9 MB)**, baked once and indexed by the state;
  * ``u = 0`` renders **identically** to ``last_u = None`` (verified over 9 angles), because the
    zero-size sprite blits nothing — so no separate "is there a torque" flag is needed and the state
    stays two plain floats.

The blit is SDL's own composite, ``d + ((s-d)*a) >> 8``, a different rule from SDL_gfx's pixel writer;
recovered by diffing over four backgrounds (100% of pixels, against 96.2% for the plausible ``//255``
forms) and verified by replaying whole frames.

Physics
-------

All five also ship ``step`` — a batched GPU simulator, so ``env.step`` / ``env.unroll`` run
entirely on device. **Renderer parity and simulator parity are different claims and this file must not
blur them**, but here they are unusually close, and for a reason worth stating: these dynamics are a
published closed form (Euler for CartPole and the two mountain cars, one RK4 step for Acrobot), so a
port can be transcribed rather than re-derived, and there is no contact and no chaos excuse. The
reference is float64 and so is this, term for term, in the same association order.

Measured against the official CPU env on identical action sequences from identical states, 16 episodes
x 100 steps (actions held 4 steps), comparing the FULL dynamical state including the velocities. Every
number below is reproduced by the classic-control step regression:

    env                      max deviation over 1,600 steps   after ONE step     bit-identical steps
    CartPole                 4.4e-14  (theta_dot)             1.1e-16            50.5%
    Pendulum                 2.8e-13  (theta_dot)             2.2e-16            71.9%
    Acrobot                  4.4e-13  (dtheta2)               1.1e-16            15.4%
    MountainCar              3.3e-16  (position)              5.4e-20            74.9%
    MountainCarContinuous    1.2e-07  (position)              6.0e-08            74.7%

The one-step column is the honest control, and it is the analogue of PushT's kinematic agent: it is an
integrator check with no room for divergence to hide in. The growth from there is not error in any
useful sense, it is the divergence of two trajectories that started one ulp apart — Acrobot is the
chaotic one and duly has the widest spread and the fewest bit-identical steps.

**Where the ulp comes from is measured, not assumed.** ``+ - * /`` are IEEE operations and are identical
on both machines, so the only place a float64 port can lose exactness is the transcendentals. Over 1e6
random inputs, ``torch.sin``/``torch.cos`` on the GPU differ from ``np.sin``/``np.cos`` on **16.0% and
16.6%** of inputs, by **at most 1 ulp** (1.1e-16). That is the entire explanation for the first column,
and it is a property of two correctly-rounded-to-within-1-ulp libraries, not of this port.

MountainCarContinuous is the exception and its number is a dtype story, not a physics one. Its ``step``
stores ``np.array([position, velocity], dtype=np.float32)`` while the discrete MountainCar keeps
float64, so under NEP 50 the whole update runs in float32; this implements that. But ``reset`` leaves a
float64 state behind, and the reference's FIRST step after a reset still accumulates in float64. Seeded
from a state the reference has already stored — which is every step of a rollout — the agreement is
**exact: exact_frac 1.0, max deviation 0.0 over 16 x 100 steps**. The 1.2e-07 above is that first step
and its consequences, and it is one float32 ulp of position.

Throughput on one H200, batched GPU step against one official CPU env (env-steps/s):

    env                      B=256      B=4,096      B=16,384    official CPU     step ms 256->16k
    CartPole                 4.34 M      65.7 M      248.7 M       181,000      0.059 -> 0.066
    Pendulum                 9.00 M     128.8 M      476.6 M        97,700      0.028 -> 0.034
    Acrobot                  0.61 M       9.43 M      36.1 M        31,100      0.419 -> 0.454
    MountainCar              6.78 M     100.2 M      376.5 M       123,000      0.038 -> 0.044
    MountainCarContinuous    6.34 M      89.8 M      331.9 M       488,000      0.040 -> 0.049

The step is batch-INDEPENDENT to within 12% across a 64x range, which is the signature of a launch-bound
kernel sequence and is exactly why it is captured into a CUDA graph (4.2-5.5x, measured; see
``_stepgraph``). The consequence is that the speedup over the CPU env is not a number but a function of
how many environments were asked for: 24x at B=256 and 1,374x at B=16,384 for CartPole.

NOT REGISTERED: **LunarLander-v3**, deliberately. Its ``render()`` decrements ``obj.ttl`` on the engine
particles it is drawing and then culls them, so rendering twice gives two different images; the
particles are Box2D bodies created at randomised positions by ``step``; and the terrain is regenerated
per ``reset`` from 12 random chunk heights that appear nowhere in the 8-D observation, which also omits
the leg joint angles it draws. A workable render state exists — roughly lander pose (3) + two legs (6)
+ 12 terrain heights + 3 per live particle — but that is a "replay the simulator's render state"
contract, not this one, and pretending otherwise would make the state-to-frame claim false for it.
See ``docs/research/classic_control.md`` §5.

Also out of scope, with the reason recorded rather than left implicit: the DM Control 2-D domains.
Planar physics does not imply a planar image — every one of them loads a gradient skybox, a
checkerboard floor with reflectance, a headlight plus shadow mapping, and a perspective (fovy 45)
camera at an oblique position. There is no world-to-pixel map a polygon rasterizer can use. See
``docs/research/classic_control.md`` §6.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from tensordict import TensorDict

from ..env import Env
from ..registry import register
from ..render import _resources as _RES
from ..render import gfxdraw as G
from ..spaces import Dict, Discrete, RGBBox
from . import _stepgraph

WHITE = (255, 255, 255)


#: The per-environment-colour canvas lives in ``render/gfxdraw`` now, because it is not a classic-
#: control idea: LunarLander needs the same thing, plus the per-environment ON/OFF gate that a batch
#: holding different numbers of exhaust particles requires. Its docstring carries the reasoning.
_StyleCanvas = G.StyleCanvas
_u8 = G.to_u8


def _poly(cv, verts, color, bidx, kmax, win=None) -> None:
    G.polygon(cv.use(color), verts, None, None, bidx, kmax, True, window=win)


def _circ(cv, cx, cy, r, color, bidx, gfx) -> None:
    """``aacircle`` + ``filled_circle``. The fill is the disc's own offset stencil, so it costs its
    own pixels and needs no window: every disc in this family is under 45 px across."""
    G.circle(cv.use(color), cx, cy, r, None, None, bidx, gfx.hw(r), gfx.stencil(r), True, disc=gfx.disc(r))


def _row(cv, y: int, color) -> None:
    """A full-width solid row (``gfxdraw.hline`` across the surface), one colour per environment."""
    cv.buf[:, :, y, :] = _u8(color).t().unsqueeze(-1)


def _box(H: int, W: int, y0, y1, x0, x1, m: int = 2):
    """A drawing window, clamped to the surface and grown by ``m`` pixels.

    Every window in this file is a bound on a primitive's footprint that holds for EVERY state, not
    for the states a rollout happens to visit: the geometry moves under a sine, a cosine or a
    rotation, so its extent along at least one axis is a constant of the environment even though its
    position is not. That is what makes it safe to evaluate the span rule and write the colour over
    the box alone -- a state-dependent box would need a host sync to slice with (``render/gfxdraw``
    rule 6), and a box that were merely usually right would silently truncate a primitive.

    The margin is slack for the truncation that turns a float coordinate into a Sint16 one, where a
    vertex sitting on an integer can land either side of it. It costs rows, never pixels.
    """
    return (max(0, int(y0) - m), min(H, int(y1) + 1 + m), max(0, int(x0) - m), min(W, int(x1) + 1 + m))


class _Classic(Env):
    """Shared scaffolding: fixed native canvas, float64 geometry, arbitrary leading dims.

    Subclasses implement ``_draw(cv, s, sty)`` on a pre-flip canvas, where ``sty`` is the resolved
    per-environment style; this class owns the batching, the ``out=`` contract, and the resolution
    policy.
    """

    family = "classic_control"

    #: (H, W) of the reference surface. Fixed, because the official renderers hardcode their primitive
    #: sizes in PIXELS (a 50x30 cart, a 10 px pole, an r=5 axle) rather than in world units, so there is
    #: no resized frame to be exact against — a different size would be a different picture that no
    #: reference produces.
    native: tuple[int, int] = (0, 0)

    # --- physics ----------------------------------------------------------------------------
    #
    # The compact render row is NOT the dynamical state, and this family is where that is most visible: the
    # frame needs (x, theta) and the integrator needs (x, x_dot, theta, theta_dot). The compact row
    # stays exactly what it is — the renderer and every stored corpus depend on it — and the velocities
    # live on public State (and in a float64 buffer beside the packed row). ``unpack_render_state``
    # therefore zeroes omitted fields, which is not a compromise but the reference's own condition after
    # ``reset``: ``reference_set_state`` has always written ``[x, 0.0, theta, 0.0]``.
    #
    # float64 throughout, because the claim here is near-bit-exactness against a float64 reference and
    # float32 would forfeit it in the first step. The compact state is still float32 — that is the
    # contract, and it is what the renderer truncates to integer pixels anyway.

    #: The FULL dynamical state. A superset of ``state_names`` for the three envs whose velocities are
    #: not drawn; Pendulum goes the other way and carries a drawn quantity (torque) that is not a
    #: velocity, which is why the two vectors are related by an explicit index map rather than by
    #: "the render dims come first".
    sim_state_names: tuple[str, ...] = ()
    #: index of each render-state dim inside ``sim_state_names``
    render_from_sim: tuple[int, ...] = ()
    #: size of the discrete action set. 0 means the action is continuous.
    n_actions: int = 0
    #: ``rollout.RandomPolicy`` reads these three. For a discrete env the bounds are HALF-OPEN —
    #: ``[0, n_actions)`` — so that a uniform float floored to an integer is a uniform action; ``step``
    #: floors and clamps, so an integer-valued action tensor works unchanged.
    action_dim: int = 1
    action_low: float = 0.0
    action_high: float = 1.0
    #: capture the step into a CUDA graph on first use; see ``_stepgraph``. Set False before the first
    #: ``step`` to keep the eager path. The two are the same code and are asserted to be bit-identical.
    use_step_graph: bool = True
    #: use the fused per-pixel kernel where it is available. Set False to force the per-primitive
    #: path, which is what the identity tests compare against and what a CPU render always takes.
    use_fused_render: bool = True
    step_reference_checkable = True
    #: 4 rather than 1 because a force resampled every 20 ms leaves CartPole jittering near its start
    #: and MountainCar rocking in place; held actions drive both to the ends of their state space,
    #: which is where an integration difference would show up.
    step_reference_hold = 4

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        H, W = self.native
        self.gfx = G.Gfx(H, W, self.device)
        f = dict(device=self.device, dtype=torch.float64)
        self._sim = torch.zeros(self.B, max(1, len(self.sim_state_names)), **f)
        self._cols = torch.tensor(self.render_from_sim, dtype=torch.long, device=self.device)
        self._act = torch.zeros(self.B, self.action_dim, device=self.device, dtype=torch.float32)
        self._step_graph = None
        self._step_graph_error = ""
        self._disc_tables = None

    def _set_render_state(self, states, variation=None, style=None) -> _Classic:
        """As the base class, and put the velocities the compact state does not carry back at rest.

        The alternative — keeping whatever the env was doing before — would make ``set_state(s)`` depend
        on history, so two rollouts from the same stored state would differ. At rest is also exactly
        what ``reference_set_state`` writes into the official env, so the two paths start from the same
        condition rather than from two defensible ones.
        """
        super()._set_render_state(states, variation=variation, style=style)
        self._sim.zero_()
        self._sim[:, self._cols] = self._render_state.to(torch.float64)
        return self

    def _restore_state(self, state: TensorDict) -> None:
        self.set_sim_state(self._sim_from_state(state))
        self._live_rows = self.B

    def sim_state(self) -> torch.Tensor:
        """``(B, len(sim_state_names))`` float64 — the full dynamical state, velocities included."""
        return self._sim

    def set_sim_state(self, v) -> _Classic:
        self._sim.copy_(
            torch.as_tensor(v, device=self.device, dtype=torch.float64).reshape(self.B, self._sim.shape[1])
        )
        self._render_state.copy_(self._sim.index_select(1, self._cols).to(torch.float32))
        return self

    def _discrete(self, a: torch.Tensor) -> torch.Tensor:
        """``(B,1)`` float -> ``(B,)`` long in ``[0, n_actions)``. Floor, not round: with the half-open
        bounds above, flooring a uniform float gives a uniform action, and rounding would give the two
        end actions half the weight of the others."""
        return a[:, 0].floor().clamp(0, self.n_actions - 1).long()

    def _dynamics(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """``(B, sim_dim)`` float64 + ``(B, action_dim)`` float32 -> the next dynamical state."""
        raise NotImplementedError

    def _advance(self) -> None:
        """In place, through fixed addresses, so the whole thing is CUDA-graph capturable."""
        self._sim.copy_(self._dynamics(self._sim, self._act))
        self._render_state.copy_(self._sim.index_select(1, self._cols).to(torch.float32))

    # ------------------------------------------------------------------ state / dynamics
    #
    # ``state_layout`` is DERIVED, in ``__init_subclass__`` below, from ``sim_state_names`` and
    # ``render_from_sim`` rather than written out five times. Those two already say everything the
    # layout needs — which dynamical variables exist and which of them the packed row carries — so a
    # hand-written layout would be the same fact in a second place, free to disagree with the first.
    # The velocities the packed row does not carry get offset ``None``, which is exactly PushT's
    # agent velocity: real dynamical state, part of the public State so ``transition`` can be pure,
    # dropped by ``pack`` so the frozen corpus layout is untouched.

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls.sim_state_names:
            col = {i: j for j, i in enumerate(cls.render_from_sim)}
            # The fourth element is the PUBLIC DTYPE (foundation commit b1585fc): float64, because
            # this family's claim is near-bit-exactness against a float64 reference and a float32
            # State would make ``transition`` a lower-precision simulator. ``pack`` still writes the
            # unchanged float32 row, so no digest and no frozen corpus moves.
            cls.state_layout = tuple(((nm,), 1, col.get(i), torch.float64) for i, nm in enumerate(cls.sim_state_names))
            cls._check_state_layout_is_complete()

    @classmethod
    def _check_state_layout_is_complete(cls) -> None:
        """At CLASS CREATION: the derived layout covers the packed row and names every variable once.

        ``state_layout``, ``_hidden_state`` and ``_live_state`` are three views of one generated
                fact, and this is where the generation is checked — once, for every class that will ever
                exist, rather than on every ``step``. A cell that declared four ``sim_state_names`` and a
                ``render_from_sim`` pointing at a two-column packed row would fail to IMPORT rather than
                report zeros at run time.
        """
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
        """The live float64 columns of ``_sim``, split by whether the packed row carries them.

        One function for both halves of the answer the base class asks for, because they ARE one
        fact — the simulator's state — cut by ``render_from_sim``. Writing them separately is how the
        two would come to disagree.
        """
        want = set(self.render_from_sim)
        n = self._live_rows
        return {
            (nm,): self._sim[:n, i : i + 1].clone()
            for i, nm in enumerate(self.sim_state_names)
            if (i in want) == drawn
        }

    def _hidden_state(self) -> dict:
        """The velocities: dynamical variables the packed row does not carry (offset ``None``)."""
        return self._sim_columns(drawn=False)

    def _live_state(self) -> dict:
        """The DRAWN variables, at the precision they are simulated in rather than the row's.

        The packed row is ``_sim[:, render_from_sim].to(float32)``, so widening it back gives
        ``float64(float32(x))``, which is not ``x``: measured on CartPole after five steps the drawn
        ``x`` moves by 9.6e-08, and that is enough for ``transition(env.state, a)`` to stop equalling
        ``step(a)`` — the functional path quietly becoming a different simulator. Answering here is
        what the foundation's ``_live_state`` hook exists for (findings §10, R7); before it existed
        this family had to override the ``state`` property, which was the one thing the
        review's T1 defect asks cells not to do.
        """
        return self._sim_columns(drawn=True)

    def _state_td(self, sim: torch.Tensor) -> TensorDict:
        """``(n, sim_dim)`` float64 -> the public :data:`State`, one key per dynamical variable.

        FLOAT64, and that is the whole reason this family carries its own State builder. The claim
        this file makes is near-bit-exactness against a float64 reference (3.3e-16 over 100 steps for
        MountainCar), and the packed row is float32 because that is the renderer's contract — it
        truncates to integer pixels anyway. A State that rounded the dynamical variables to float32
        would make ``transition`` a DIFFERENT, lower-precision simulator from ``step``: identical for
        one step taken from a corpus state (where the velocities are zero and the drawn columns are
        float32 values), and drifting from the second step on. Keeping the public State at the
        dtype the dynamics are computed in is what makes ``transition`` equal ``step`` to the bit and
        ``unroll`` equal repeated ``transition`` for any number of steps.
        """
        n = int(sim.shape[0])
        return TensorDict(
            {nm: sim[:, i : i + 1].clone() for i, nm in enumerate(self.sim_state_names)},
            batch_size=[n],
            device=self.device,
        )

    def _sim_from_state(self, state) -> torch.Tensor:
        """A :data:`State` (or packed rows) -> the ``(n, sim_dim)`` float64 vector ``_dynamics`` takes.

        Packed rows are accepted and the velocities come back at rest, which is not a guess: it is
        exactly what ``set_state`` writes and exactly what ``reference_set_state`` puts into the
        official env, so ``transition(rows, a)`` and ``step(a)`` after ``set_state(rows)`` start from
        the same condition.
        """
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
    def transition(self, state, action) -> TensorDict:
        """``(state, action) -> next_state``. FUNCTIONAL: the env's own buffers are not touched.

        The same ``_dynamics`` over the same float64 inputs that ``step`` uses, so the two agree to
        the bit — eager, because a captured CUDA graph reads ``_sim``/``_render_state`` at fixed addresses
        and this call may not write them.
        """
        sim = self._sim_from_state(state)
        act = torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(-1, self.action_dim)
        return self._state_td(self._dynamics(sim, act))

    @torch.no_grad()
    def step(self, action) -> TensorDict:
        """Advance every environment one control step. ``action`` is ``(B, action_dim)``.

        STATEFUL: it writes ``env.state``. Returns a :data:`~stable_envs.env.StepResult` carrying
        ``"state"`` and nothing else. There is no reward and no termination here and that is
        deliberate: the reference's ``step`` integrates whatever state it is in and reports
        ``terminated`` as a separate flag, so a batched simulator that reset or froze an environment
        on its own would be simulating something else, and a fabricated zero reward would read as a
        measured one (contract §5). Apply the published predicate to the returned states.
        """
        self._check_open("step")
        self._act.copy_(
            torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(self.B, self.action_dim)
        )
        replayed = False
        if self.use_step_graph and self.device.type == "cuda":
            if self._step_graph is None and not self._step_graph_error:
                self._step_graph, self._step_graph_error = _stepgraph.capture(self._advance, (self._sim, self._render_state))
                if self._step_graph is None:
                    self.use_step_graph = False
            if self._step_graph is not None:
                self._step_graph.replay()
                replayed = True
        if not replayed:
            self._advance()
        # Built straight from the simulator buffer rather than read back through ``self.state``.
        # The two are the same object by construction — ``_live_state``/``_hidden_state`` are these
        # very columns, and ``test_step_returns_exactly_env_state`` asserts it for every cell — but
        # the property round-trips the packed row through ``unpack`` (a copy, a TensorDict, then two
        # splices) to arrive back where ``_sim`` already is. On the cheapest step in the repository
        # that doubled the cost of ``step``: measured 0.063 ms against 0.037 ms at 64 environments.
        return self._step_result(self._state_td(self._sim[: self._live_rows]))

    def _close(self) -> None:
        """Drop the captured step graph. ``Env.close`` calls this once and is idempotent."""
        self._step_graph = None

    # --- contract ---------------------------------------------------------------------------

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
        """``(...,state_dim)`` -> ``(...,H,W,3)`` uint8 on device, for arbitrary leading dims."""
        H, W = self.native
        want = int(self.size if size is None else size)
        if want != W:
            raise ValueError(
                f"{type(self).__name__} renders at its reference's native {H}x{W} and nothing else "
                f"(got size={want}). The official renderer hardcodes every primitive in pixels, so a "
                f"resized frame has no reference to be exact against; resample the output if you want "
                f"one."
            )
        if supersample not in (None, 1):
            raise ValueError(
                f"{type(self).__name__}: supersample={supersample} is not available. gfxdraw's "
                f"geometry is integer and its anti-aliasing is Wu weights, not coverage — "
                f"supersampling would average a picture the reference never draws."
            )
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        # float64 from here on: the frame depends on trunc() of the vertex coordinates, so float32
        # arithmetic lands on the wrong integer whenever a vertex falls within ~1e-5 of one.
        s = s.reshape(-1, self.state_dim).to(torch.float64)
        n = s.shape[0]
        sty = self.resolve_style(style, variation, batch=n)
        o = None if out is None else out.reshape(-1, H, W, 3)
        if self._use_fused():
            from ..render import gfxdraw_classic as F

            img = F.render_classic(self._ops(s, sty).finish(), out=o)
        else:
            cv = _StyleCanvas(self.gfx, n, sty["background.color"])
            cv.aa = sty["rendering.antialias"] > 0.5
            self._draw(cv, s, sty)
            img = cv.rgb(o)
        return img.view(*lead, H, W, 3) if lead else img

    @property
    def frame_shape(self) -> tuple[int, int]:
        """``(H, W)`` of the frame, which in this family is NOT ``(size, size)``.

        ``Env.frame_spec`` assumes a square frame unless the env says otherwise, and three of
        these five are 400x600: the reference hardcodes its canvas, ``size`` is its WIDTH, and a spec
        that reported ``(600, 600, 3)`` would be wrong about the buffer every consumer allocates.
        """
        return self.native

    def _use_fused(self) -> bool:
        """The fused kernel is CUDA-only and opt-outable, and it never silently replaces the CPU
        path: Triton has no CPU backend here, so a ``device="cpu"`` render is the per-primitive one
        by construction rather than by policy."""
        if not (self.fused_capable and self.use_fused_render and self.device.type == "cuda"):
            return False
        from ..render import gfxdraw_classic as F

        return F.HAVE_TRITON

    def _draw(self, cv, s, sty) -> None:
        raise NotImplementedError

    # --- the fused path ---------------------------------------------------------------------
    #
    # ``_draw`` issues the primitives; ``_ops`` DESCRIBES them. The two are the same geometry
    # written twice, which is a cost worth naming: the per-primitive path stays the readable
    # definition and the reference every fused frame is diffed against, and the description is what
    # a single per-pixel pass can consume. Neither is derived from the other at runtime, so a
    # divergence is a test failure rather than a silent agreement.

    #: radii this environment's discs use. Static geometry, so the disc tables may be cached on it.
    disc_radii: tuple[int, ...] = ()
    #: set False on a class whose frame the fused kernel does not describe yet.
    fused_capable: bool = False

    def _ops(self, s, sty):
        """The frame as a primitive list. Subclasses build on ``_ops_begin``."""
        raise NotImplementedError

    def _ops_begin(self, s, sty):
        from ..render import gfxdraw_classic as F

        H, W = self.native
        ops = F.Ops(s.shape[0], H, W, self.device)
        if self._disc_tables is None and self.disc_radii:
            self._disc_tables = F.DiscTables(self.disc_radii, self.device)
        ops.discs = self._disc_tables
        ops.aa = (sty["rendering.antialias"] > 0.5).to(torch.int32)
        ops.bg = _u8(sty["background.color"]).to(torch.int32)
        return ops

    # --- variation --------------------------------------------------------------------------

    @staticmethod
    def _colors(**kw) -> dict:
        return {k: RGBBox(init_value=v) for k, v in kw.items()}

    def _aa_space(self) -> Dict:
        """The one non-appearance knob: exact anti-aliasing on or off.

        Exposed rather than hidden because it is a real, measured trade and callers should be able to
        take it knowingly: dropping the Wu pass costs **0.055-0.13% of pixels** (all of them on shape
        boundaries, none in the interior). Only ``antialias=1`` is bit-identical to the reference; the
        default is 1. It is honoured PER ENVIRONMENT like every other knob, so a batch may hold both
        settings: the Wu pass runs and the environments that asked for no anti-aliasing discard their
        blended writes. The ~1.15x it used to buy is therefore only available to a batch that is
        uniformly off, which no longer skips the pass — correctness of the per-env contract first.
        """
        return Dict({"antialias": Discrete(2, init_value=1)})

    # --- reference bridge -------------------------------------------------------------------

    @classmethod
    def reference_env(cls, size: int | None = None):
        """The official gymnasium env. ``size`` is ignored — the canvas is not configurable."""
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

    @classmethod
    def reference_action(cls, a):
        """The action the official ``step`` expects, decoded from the same float ``step`` is given.

        Both sides must decode identically or the two simulators are being driven by different actions
        and every number that follows is meaningless — so this mirrors ``_discrete`` exactly rather
        than doing "the obvious thing" with a cast.
        """
        if cls.n_actions:
            return int(min(max(math.floor(float(a[0])), 0), cls.n_actions - 1))
        return np.asarray(a, np.float32).reshape(-1)

    @classmethod
    def reference_step(cls, ref, a) -> None:
        ref.step(cls.reference_action(a))

    @classmethod
    def reference_sim_state(cls, ref) -> np.ndarray:
        """``(len(sim_state_names),)`` float64, read off the official env."""
        raise NotImplementedError


# =============================================================================== CartPole
@register
class CartPole(_Classic):
    """CartPole-v1. Render state ``[x, theta]`` — 2 of the observation's 4 dims.

    Draw order, from the source: cart quad, pole quad, axle disc, then ``hline`` for the track, which
    is painted LAST and therefore over the cart. The cart is axis-aligned, so its ``aapolygon`` takes
    SDL_gfx's solid ``hline``/``vline`` short-circuit and is not anti-aliased at all; only the pole is.

    ``step`` takes 2 discrete actions — 0 push left, 1 push right — as one float per environment,
    floored. Dynamical state ``[x, x_dot, theta, theta_dot]``, Euler at ``tau = 0.02``; tracks the
    reference to **4.4e-14 over 100 steps, 1.1e-16 after one**.
    """

    name = "cartpole"
    state_dim = 2
    state_names = ("x", "theta")
    native = (400, 600)
    default_size = 600
    reference_id = "CartPole-v1"

    SCALE = 600 / 4.8  # screen_width / (2 * x_threshold)
    POLELEN = 125.0  # scale * 2 * length
    CARTY = 100.0
    AXLEOFF = 30.0 / 4.0
    AXLE_R = 5  # int(polewidth / 2)

    #: Physics, from ``CartPoleEnv.__init__``. Written as the same expressions, not as their decimal
    #: values: ``masspole + masscart`` and ``0.1 * 0.5`` are float64 results and a transcribed literal
    #: is a different number in the last bit, which is the whole margin this env is claiming.
    sim_state_names = ("x", "x_dot", "theta", "theta_dot")
    render_from_sim = (0, 2)
    n_actions = 2
    action_low, action_high = 0.0, 2.0
    GRAVITY = 9.8
    MASSCART = 1.0
    MASSPOLE = 0.1
    TOTAL_MASS = MASSPOLE + MASSCART
    LENGTH = 0.5  # actually half the pole's length
    POLEMASS_LENGTH = MASSPOLE * LENGTH
    FORCE_MAG = 10.0
    TAU = 0.02
    #: theta is unwrapped in the reference (the pole is allowed to go round), so 'abs' and not 'angle';
    #: measuring it wrapped would quietly forgive a full-turn disagreement.
    step_reference_groups = (
        ("x", (0,), "abs"),
        ("x_dot", (1,), "abs"),
        ("theta", (2,), "abs"),
        ("theta_dot", (3,), "abs"),
    )

    def _dynamics(self, s, a):
        """``CartPoleEnv.step``, term for term, in the same association order, in float64.

        ``kinematics_integrator == 'euler'``: every update uses the OLD velocity, so all four writes
        happen from the same state. Semi-implicit Euler is one line away in the reference and gives a
        visibly different trajectory, which is the sort of difference that survives every eyeball test.
        """
        x, x_dot, th, th_dot = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        force = torch.where(
            self._discrete(a) == 1, torch.full_like(x, self.FORCE_MAG), torch.full_like(x, -self.FORCE_MAG)
        )
        costheta, sintheta = torch.cos(th), torch.sin(th)
        temp = (force + self.POLEMASS_LENGTH * th_dot.square() * sintheta) / self.TOTAL_MASS
        thetaacc = (self.GRAVITY * sintheta - costheta * temp) / (
            self.LENGTH * (4.0 / 3.0 - self.MASSPOLE * costheta.square() / self.TOTAL_MASS)
        )
        xacc = temp - self.POLEMASS_LENGTH * thetaacc * costheta / self.TOTAL_MASS
        return torch.stack(
            [x + self.TAU * x_dot, x_dot + self.TAU * xacc, th + self.TAU * th_dot, th_dot + self.TAU * thetaacc], 1
        )

    def make_variation_space(self) -> Dict:
        return Dict(
            {
                "cart": Dict(self._colors(color=(0, 0, 0))),
                "pole": Dict(self._colors(color=(202, 152, 101))),
                "axle": Dict(self._colors(color=(129, 132, 203))),
                "track": Dict(self._colors(color=(0, 0, 0))),
                "background": Dict(self._colors(color=WHITE)),
                "rendering": self._aa_space(),
            }
        )

    #: Row windows. ``x`` moves the cart along the track and ``theta`` only rotates the pole, so the
    #: ROW extent of all three primitives is a constant of the environment for every state: the cart
    #: is 30 px tall about ``CARTY`` and the pole reaches ``hypot(5, POLELEN-5)`` from the axle
    #: whatever its angle. The column extent is not bounded -- the cart really does visit every
    #: column -- so these windows are rows only.
    WIN_CART = (85, 115)
    WIN_POLE = (
        int(CARTY + AXLEOFF - math.hypot(5.0, POLELEN - 5.0)),
        int(CARTY + AXLEOFF + math.hypot(5.0, POLELEN - 5.0)),
    )
    #: Per EDGE, because a 50x30 quad and a 10x125 one both have two long sides and two short ones,
    #: and the Wu pass costs its bound in steps whether the edge is that long or not.
    KMAX_CART = (32, 52, 32, 52)
    KMAX_POLE = (132, 16, 132, 16)

    def _draw(self, cv, s, sty) -> None:
        g, B = self.gfx, s.shape[0]
        H, W = self.native
        bi = g.batch_index(B)
        cartx = s[:, 0] * self.SCALE + 300.0
        carty = torch.full_like(cartx, self.CARTY)

        ctr = torch.stack([cartx, carty], -1).view(B, 1, 2)
        cart = g.verts([(-25.0, -15.0), (-25.0, 15.0), (25.0, 15.0), (25.0, -15.0)]).view(1, 4, 2) + ctr
        _poly(cv, cart, sty["cart.color"], bi, self.KMAX_CART, _box(H, W, *self.WIN_CART, 0, W - 1))

        ploc = (
            g.verts([(-5.0, -5.0), (-5.0, self.POLELEN - 5.0), (5.0, self.POLELEN - 5.0), (5.0, -5.0)])
            .view(1, 4, 2)
            .expand(B, 4, 2)
        )
        pole = G.rotate(ploc, (-s[:, 1]).view(B, 1)) + torch.stack([cartx, carty + self.AXLEOFF], -1).view(B, 1, 2)
        _poly(cv, pole, sty["pole.color"], bi, self.KMAX_POLE, _box(H, W, *self.WIN_POLE, 0, W - 1))

        cxi = torch.trunc(cartx).to(torch.int64)
        cyi = torch.full_like(cxi, int(self.CARTY + self.AXLEOFF))
        _circ(cv, cxi, cyi, self.AXLE_R, sty["axle.color"], bi, g)

        _row(cv, int(self.CARTY), sty["track.color"])  # gfxdraw.hline, drawn LAST, over the cart

    disc_radii = (AXLE_R,)
    fused_capable = True

    def _ops(self, s, sty):
        """``_draw``'s four primitives, described rather than issued. Same geometry, same order."""
        g, B = self.gfx, s.shape[0]
        ops = self._ops_begin(s, sty)
        cartx = s[:, 0] * self.SCALE + 300.0
        carty = torch.full_like(cartx, self.CARTY)
        ctr = torch.stack([cartx, carty], -1).view(B, 1, 2)
        ops.polygon(
            g.verts([(-25.0, -15.0), (-25.0, 15.0), (25.0, 15.0), (25.0, -15.0)]).view(1, 4, 2) + ctr,
            sty["cart.color"],
        )
        ploc = (
            g.verts([(-5.0, -5.0), (-5.0, self.POLELEN - 5.0), (5.0, self.POLELEN - 5.0), (5.0, -5.0)])
            .view(1, 4, 2)
            .expand(B, 4, 2)
        )
        ops.polygon(
            G.rotate(ploc, (-s[:, 1]).view(B, 1)) + torch.stack([cartx, carty + self.AXLEOFF], -1).view(B, 1, 2),
            sty["pole.color"],
        )
        cxi = torch.trunc(cartx).to(torch.int64)
        ops.disc(cxi, torch.full_like(cxi, int(self.CARTY + self.AXLEOFF)), self.AXLE_R, sty["axle.color"])
        ops.hline(int(self.CARTY), sty["track.color"])
        return ops

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """The region an EPISODE reaches, not the reset ball.

        ``reset`` draws U(-0.05, 0.05)^4, which leaves the cart within 6 px of centre and the pole
        within 3 degrees of vertical — a parity test there would never move the pole far enough to
        exercise a rotated quad, and never push the cart against the edge where SDL_gfx's line clipping
        engages. The termination bounds are the states the env actually visits.
        """
        rng = rng or self.rng
        return np.stack([rng.uniform(-2.4, 2.4, batch), rng.uniform(-0.2095, 0.2095, batch)], 1).astype(np.float32)

    def step(self, action) -> TensorDict:
        """One control step. ``action`` is ``(B,1)``: **0 push left, 1 push right**, floored.

        Declared per env rather than inherited because the number means something different in each of
        these five, and because the conformance suite requires the class that CLAIMS a batched
        simulator to declare one — an inherited ``step`` reached through a base class is exactly how a
        checker ends up driving an env it knows nothing about.
        """
        return super().step(action)

    @classmethod
    def reference_set_state(cls, env, st) -> None:
        env.state = np.array([st[0], 0.0, st[1], 0.0])

    @classmethod
    def reference_sim_state(cls, ref) -> np.ndarray:
        return np.asarray(ref.state, np.float64)

    @classmethod
    def reference_step(cls, ref, a) -> None:
        # The pole legitimately falls during a random rollout, and the reference then logs a warning on
        # every subsequent step ("you are calling step() even though this environment has already
        # returned terminated = True"). ``steps_beyond_terminated`` is reward bookkeeping and never
        # enters the dynamics — the integrator keeps running — so it is cleared rather than silenced.
        ref.steps_beyond_terminated = None
        super().reference_step(ref, a)


# =============================================================================== Pendulum
@register
class Pendulum(_Classic):
    """Pendulum-v1. Render state ``[theta, torque]``.

    ``torque`` is ``last_u``: not part of the observation, but it is drawn, so it is part of the render
    state. See the module docstring for why it reduces to a 228-entry sprite table and why ``u = 0``
    needs no separate flag.

    ``step`` takes one CONTINUOUS torque in [-2, 2]. Dynamical state ``[theta, theta_dot, torque]``, and
    stepping is what makes the third render dimension move on its own: the torque a policy asks for is
    the torque that gets drawn. Tracks the reference to **2.8e-13 over 100 steps, 2.2e-16 after one**.
    """

    name = "pendulum"
    state_dim = 2
    state_names = ("theta", "torque")
    native = (500, 500)
    default_size = 500
    reference_id = "Pendulum-v1"

    SCALE = 500 / 4.4
    OFF = 250
    ROD_R = 11  # int(0.2 * scale / 2)
    AXLE_R = 5  # int(0.05 * scale)
    MAX_SPRITE = 113  # int(scale * max_torque / 2), max_torque = 2
    WIN = 113  # sprite window: every size fits rows/cols 194..306

    #: Physics, from ``PendulumEnv.__init__``. ``g`` is 10.0 here, not 9.8: Pendulum is the one member
    #: of the family that does not use Earth gravity, and it is a constructor default rather than a
    #: class constant, so it is easy to carry over from CartPole and be wrong by 2%.
    #: ``torque`` is both the third dynamical variable and the second RENDER dimension — it is
    #: ``last_u``, which the reference draws — so ``render_from_sim`` is (0, 2) and not (0, 1).
    sim_state_names = ("theta", "theta_dot", "torque")
    render_from_sim = (0, 2)
    n_actions = 0  # continuous: one torque in [-2, 2]
    action_low, action_high = -2.0, 2.0
    G, M, L, DT = 10.0, 1.0, 1.0, 0.05
    MAX_SPEED, MAX_TORQUE = 8, 2.0
    #: theta is NOT wrapped by the reference — it accumulates, and the render is periodic in it — so
    #: the error is measured unwrapped. Wrapping it here would forgive a whole-turn disagreement.
    step_reference_groups = (("theta", (0,), "abs"), ("theta_dot", (1,), "abs"), ("torque", (2,), "abs"))

    def _dynamics(self, s, a):
        """``PendulumEnv.step``, in float64 — except for one term, which the reference computes in
        float32 and which therefore has to be computed in float32 here too.

        ``3.0 / (m * l**2) * u`` multiplies a python float by ``last_u``, and ``last_u`` is float32
        because it comes out of a float32 action array; under NEP 50 the python scalar is weak, so the
        product is float32 and is ROUNDED before it meets the float64 gravity term. Computing that term
        in float64 changes the trajectory in the last few bits from the first step, which is exactly
        the margin this env is claiming.
        """
        th, thdot = s[:, 0], s[:, 1]
        u = a[:, 0].clamp(-self.MAX_TORQUE, self.MAX_TORQUE)  # float32, as the ref
        drive = (3.0 / (self.M * self.L**2) * u).to(torch.float64)  # float32 mul, then widen
        newthdot = thdot + (3 * self.G / (2 * self.L) * torch.sin(th) + drive) * self.DT
        newthdot = newthdot.clamp(-self.MAX_SPEED, self.MAX_SPEED)
        return torch.stack([th + newthdot * self.DT, newthdot, u.to(torch.float64)], 1)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._sprite = _torque_sprites(self.device)

    def make_variation_space(self) -> Dict:
        return Dict(
            {
                "rod": Dict(self._colors(color=(204, 77, 77))),
                "axle": Dict(self._colors(color=(0, 0, 0))),
                "background": Dict(self._colors(color=WHITE)),
                "rendering": self._aa_space(),
            }
        )

    #: Windows, both axes: the rod pivots about the centre, so it reaches ``hypot(SCALE, 0.1*SCALE)``
    #: from it at any angle and its far cap is ``SCALE`` from it at any angle. Bounds on a rotation,
    #: so they hold for every state.
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
        """The torque arrow, gathered from the baked table and alpha-composited. Drawn under the axle.

        Every size lands inside rows/cols 194..306 (the arrow is centred on 250 and at most 113 px
        across), so one fixed window covers the batch and the per-env size becomes a gather offset
        rather than a variable-shape slice.
        """
        B = u.shape[0]
        n = torch.trunc(self.SCALE * u.abs() / 2).to(torch.int64).clamp(0, self.MAX_SPRITE)
        flip = (u > 0).to(torch.int64)
        w = self.WIN
        i = torch.arange(w, device=self.device)
        off = (w // 2) - torch.div(n, 2, rounding_mode="floor")  # 56 - n//2
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
        """``theta`` over a full turn (the render is periodic in it), ``torque`` over the action space."""
        rng = rng or self.rng
        return np.stack([rng.uniform(-np.pi, np.pi, batch), rng.uniform(-2.0, 2.0, batch)], 1).astype(np.float32)

    def step(self, action) -> TensorDict:
        """One control step. ``action`` is ``(B,1)``: **a continuous torque, clipped to [-2, 2]** —
        and the value that gets DRAWN, since the torque arrow is part of the render state."""
        return super().step(action)

    @classmethod
    def reference_set_state(cls, env, st) -> None:
        env.state = np.array([st[0], 0.0])
        # float32 on purpose: step() stores the clipped action, which comes from a float32 action
        # array, and the sprite size is trunc(scale*|last_u|/2) — a float64 last_u would land on a
        # different integer for values near a boundary.
        env.last_u = np.float32(st[1])

    @classmethod
    def reference_sim_state(cls, ref) -> np.ndarray:
        """``last_u`` is ``None`` until the first ``step``. Zero is the right reading and not a
        placeholder: a zero-size torque sprite blits nothing, so ``u = 0`` renders identically to
        ``last_u = None`` — measured over nine angles, see the module docstring."""
        u = 0.0 if getattr(ref, "last_u", None) is None else float(ref.last_u)
        return np.array([ref.state[0], ref.state[1], u], np.float64)


# =============================================================================== Acrobot
@register
class Acrobot(_Classic):
    """Acrobot-v1. Render state ``[theta1, theta2]`` — 2 of the observation's 6 dims.

    Two traps in the source, both silent. ``xys = np.array([[0,0], p1, p2])[:, ::-1]`` **swaps x and
    y**, so link 2 hangs off ``(p1_y, p1_x)``; and ``p2`` is computed but never drawn, because only two
    ``thetas`` are zipped against three points. The ground rule is ``pygame.draw.line``, which
    TRUNCATES its float y (363.6364 -> row 363) — rounding puts the whole line one row off, 1000 px a
    frame, which looks like a rasterizer bug rather than a coordinate one.

    ``step`` takes 3 discrete actions — torque -1, 0, +1 — as one floored float. Dynamical state
    ``[theta1, theta2, dtheta1, dtheta2]``, one RK4 step at ``dt = 0.2``; tracks the reference to
    **4.4e-13 over 100 steps, 1.1e-16 after one**. The widest spread in the family and the fewest
    bit-identical steps (15.4%), which is what a chaotic double pendulum is supposed to do to two
    trajectories that started one ulp apart.
    """

    name = "acrobot"
    state_dim = 2
    state_names = ("theta1", "theta2")
    native = (500, 500)
    default_size = 500
    reference_id = "Acrobot-v1"

    SCALE = 500 / 4.4
    OFF = 250.0
    JOINT_R = 11  # int(0.1 * scale)

    #: Physics, from ``AcrobotEnv``. ``book_or_nips == 'book'``, which selects the second of the two
    #: ``ddtheta2`` expressions in ``_dsdt`` — they differ by a real term and the paper version is
    #: reachable by changing one class attribute.
    sim_state_names = ("theta1", "theta2", "dtheta1", "dtheta2")
    render_from_sim = (0, 1)
    n_actions = 3
    action_low, action_high = 0.0, 3.0
    DT = 0.2
    M1 = M2 = 1.0
    L1 = 1.0
    LC1 = LC2 = 0.5
    I1 = I2 = 1.0
    G = 9.8
    MAX_VEL_1 = 4 * math.pi
    MAX_VEL_2 = 9 * math.pi
    AVAIL_TORQUE = (-1.0, 0.0, 1.0)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        #: on the device, once. Building it from the python tuple inside ``_dynamics`` is a host-to-
        #: device copy in the hot path, and under CUDA-graph capture it recorded NOTHING: the graph
        #: came back empty and every replay was a silent no-op. Constants belong in ``__init__``.
        self._torque = torch.tensor(self.AVAIL_TORQUE, dtype=torch.float64, device=self.device)

    #: theta1 and theta2 ARE wrapped to [-pi, pi] by the reference, so a hair's difference either side
    #: of pi is a 2*pi difference in the stored number and nothing at all in the physics. Measuring
    #: them wrapped is what separates "the integrator diverged" from "the representation flipped".
    step_reference_groups = (
        ("theta1", (0,), "angle"),
        ("theta2", (1,), "angle"),
        ("dtheta1", (2,), "abs"),
        ("dtheta2", (3,), "abs"),
    )

    def _dsdt(self, y):
        """``AcrobotEnv._dsdt``: ``(B,5)`` (theta1, theta2, dtheta1, dtheta2, torque) -> its derivative.

        Transcribed with the same parenthesisation and the same constants written as expressions, since
        every one of ``m1*lc1**2``, ``2*l1*lc2`` and ``m2*lc2*g`` is a float64 product whose decimal
        form is not the same double.
        """
        m1, m2, l1, lc1, lc2, i1, i2, g = (self.M1, self.M2, self.L1, self.LC1, self.LC2, self.I1, self.I2, self.G)
        theta1, theta2, dtheta1, dtheta2, a = (y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4])
        cos2, sin2 = torch.cos(theta2), torch.sin(theta2)
        d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos2) + i1 + i2
        d2 = m2 * (lc2**2 + l1 * lc2 * cos2) + i2
        phi2 = m2 * lc2 * g * torch.cos(theta1 + theta2 - math.pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2.square() * sin2
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin2
            + (m1 * lc1 + m2 * l1) * g * torch.cos(theta1 - math.pi / 2)
            + phi2
        )
        ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1.square() * sin2 - phi2) / (
            m2 * lc2**2 + i2 - d2**2 / d1
        )
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return torch.stack([dtheta1, dtheta2, ddtheta1, ddtheta2, torch.zeros_like(a)], 1)

    @staticmethod
    def _wrap(x, m: float, M: float, iters: int = 8):
        """``acrobot.wrap`` as masked subtraction. The reference's ``while`` loop cannot be vectorised,
        and it does not need to be: one control step moves theta1 by at most ``MAX_VEL_1 * dt`` = 2.5
        rad from a value already inside ``[-pi, pi]``, so the loop runs at most once. Eight passes is
        the same arithmetic, in the same order, with room for a state somebody set by hand."""
        diff = M - m
        for _ in range(iters):
            x = torch.where(x > M, x - diff, x)
        for _ in range(iters):
            x = torch.where(x < m, x + diff, x)
        return x

    def _dynamics(self, s, a):
        """One ``dt = 0.2`` RK4 step of the augmented state, then wrap and bound — ``AcrobotEnv.step``.

        The torque rides ALONG in the integrated vector with a zero derivative, which is the reference's
        own construction (``np.append(s, torque)``) and is why ``_dsdt`` returns five numbers.
        """
        y0 = torch.cat([s, self._torque[self._discrete(a)].unsqueeze(1)], 1)
        dt, dt2 = self.DT, self.DT / 2.0
        k1 = self._dsdt(y0)
        k2 = self._dsdt(y0 + dt2 * k1)
        k3 = self._dsdt(y0 + dt2 * k2)
        k4 = self._dsdt(y0 + dt * k3)
        ns = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return torch.stack(
            [
                self._wrap(ns[:, 0], -math.pi, math.pi),
                self._wrap(ns[:, 1], -math.pi, math.pi),
                ns[:, 2].clamp(-self.MAX_VEL_1, self.MAX_VEL_1),
                ns[:, 3].clamp(-self.MAX_VEL_2, self.MAX_VEL_2),
            ],
            1,
        )

    def make_variation_space(self) -> Dict:
        return Dict(
            {
                "link": Dict(self._colors(color=(0, 204, 204))),
                "joint": Dict(self._colors(color=(204, 204, 0))),
                "ground": Dict(self._colors(color=(0, 0, 0))),
                "background": Dict(self._colors(color=WHITE)),
                "rendering": self._aa_space(),
            }
        )

    #: Windows, per LINK, both axes. The first joint is pinned at the centre and the second is one
    #: link length from it in some direction, so ``|centre - OFF| <= i * SCALE`` for link ``i``
    #: whatever the angles are; a link then reaches ``hypot(SCALE, 0.1*SCALE)`` further. Both bounds
    #: hold for every state because they are bounds on a sine and a cosine.
    _REACH = math.hypot(SCALE, 0.1 * SCALE)
    WIN_LINK = ((int(OFF - _REACH), int(OFF + _REACH)), (int(OFF - SCALE - _REACH), int(OFF + SCALE + _REACH)))
    KMAX_LINK = (28, 120, 28, 120)  # the quad starts on its short side

    def _draw(self, cv, s, sty) -> None:
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

    disc_radii = (JOINT_R,)
    fused_capable = True

    def _ops(self, s, sty):
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

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Both links over a full turn — Acrobot's whole task is to swing through them."""
        rng = rng or self.rng
        return rng.uniform(-np.pi, np.pi, (batch, 2)).astype(np.float32)

    def step(self, action) -> TensorDict:
        """One control step. ``action`` is ``(B,1)``: **0 = -1 torque, 1 = 0, 2 = +1**, floored, applied
        to the ELBOW joint only — the shoulder is unactuated, which is the whole task."""
        return super().step(action)

    @classmethod
    def reference_set_state(cls, env, st) -> None:
        env.state = np.array([st[0], st[1], 0.0, 0.0])

    @classmethod
    def reference_sim_state(cls, ref) -> np.ndarray:
        return np.asarray(ref.state, np.float64)


# =============================================================================== MountainCar
@register
class MountainCar(_Classic):
    """MountainCar-v0. Render state ``[position]`` — one number for a 400x600 frame.

    The terrain is 99 ``pygame.draw.aalines`` segments, a sub-pixel AA path unrelated to gfxdraw's and
    not reimplemented; it is also a constant of the environment rather than of the state, so it is
    rasterised once by pygame and cached. That is exact by construction and free per frame. Everything
    the state moves — the car quad and its two wheels — is pure gfxdraw and exact by the rules in
    ``render/gfxdraw``. The flag is drawn AFTER the car in the reference, so it cannot join the baked
    layer; it is gfxdraw too, so it does not need to.

    ``step`` takes 3 discrete actions — 0 push left, 1 coast, 2 push right — as one floored float.
    Dynamical state ``[position, velocity]``; tracks the reference to **3.3e-16 over 100 steps,
    5.4e-20 after one**, the tightest in the family because one step is three float64 operations and a
    cosine.
    """

    name = "mountain-car"
    state_dim = 1
    state_names = ("position",)
    native = (400, 600)
    default_size = 600
    reference_id = "MountainCar-v0"

    MINP, MAXP = -1.2, 0.6
    #: ``screen_width / world_width``, and world_width MUST be computed as ``MAXP - MINP`` rather than
    #: written as the literal 1.8: ``0.6 - (-1.2)`` is 1.7999999999999998 in float64 and the literal is
    #: 1.8000000000000000, a 2-ULP gap that survives into ``int((goal - min) * scale)``. At goal 0.45
    #: that product is 550.0000000000001 one way and 549.9999999999999 the other, so the flag lands on
    #: a different column and the frame is off by ~35,800 px over 256 states -- entirely on edges, so
    #: interior parity stays at 100% and only an all-pixel diff catches it.
    SCALE = 600 / (0.6 - (-1.2))
    WHEEL_R = 8  # int(carheight / 2.5)
    #: where the flag stands. The ONLY thing that differs between the two mountain cars, and it is not
    #: in either render() -- see MountainCarContinuous below.
    GOALP = 0.5

    #: Physics, from ``MountainCarEnv``. The render state is ONE number and the dynamical state is two:
    #: velocity moves the car and never reaches the screen.
    sim_state_names = ("position", "velocity")
    render_from_sim = (0,)
    n_actions = 3  # 0 push left, 1 do nothing, 2 push right
    action_low, action_high = 0.0, 3.0
    FORCE = 0.001
    GRAVITY = 0.0025
    MAX_SPEED = 0.07
    step_reference_groups = (("position", (0,), "abs"), ("velocity", (1,), "abs"))

    def _dynamics(self, s, a):
        """``MountainCarEnv.step``, float64 throughout, which is what the reference's state is.

        The inelastic left wall is the interesting line: a car that is clipped to ``min_position`` while
        still moving left has its velocity ZEROED, and the test is an exact float equality against the
        clip bound. Written as a comparison rather than as "close to the wall" because that is what the
        reference does, and a tolerance here would zero the velocity of a car that the reference lets
        keep moving.
        """
        pos, vel = s[:, 0], s[:, 1]
        act = self._discrete(a).to(torch.float64)
        vel = vel + ((act - 1) * self.FORCE + torch.cos(3 * pos) * (-self.GRAVITY))
        vel = vel.clamp(-self.MAX_SPEED, self.MAX_SPEED)
        pos = (pos + vel).clamp(self.MINP, self.MAXP)
        vel = torch.where((pos == self.MINP) & (vel < 0), torch.zeros_like(vel), vel)
        return torch.stack([pos, vel], 1)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.flagx = int((self.GOALP - self.MINP) * self.SCALE)
        self.flagy1 = int(_mc_height(self.GOALP) * self.SCALE)
        self.flagy2 = self.flagy1 + 50

    def make_variation_space(self) -> Dict:
        return Dict(
            {
                "car": Dict(self._colors(color=(0, 0, 0))),
                "wheel": Dict(self._colors(color=(128, 128, 128))),
                "terrain": Dict(self._colors(color=(0, 0, 0))),
                "flag": Dict(self._colors(color=(204, 204, 0), pole_color=(0, 0, 0))),
                "background": Dict(self._colors(color=WHITE)),
                "rendering": self._aa_space(),
            }
        )

    def _terrain_base(self, bg, terrain) -> torch.Tensor:
        """``(B,400,600,3)`` static layer, one bake per DISTINCT (background, terrain) pair.

        The terrain is 99 ``pygame.draw.aalines`` segments, whose sub-pixel coverage rule is not
        reimplemented anywhere here — the layer is state-independent, so it is rasterised by pygame
        itself and that is exact by construction. A per-environment colour therefore costs a per-colour
        bake rather than a per-environment one: the batch is reduced to its distinct pairs on the host
        and the results are gathered back, so a batch that shares one appearance still pays for one.

        Recovering the pixels analytically instead was considered and is not possible bit-exactly:
        pygame writes ``trunc(a*src + (1-a)*dst)`` per channel, so a baked frame carries only
        ``floor(255a)`` and the coverage ``a`` cannot be recovered to the precision another colour pair
        would need.
        """
        key = torch.cat([bg, terrain], 1).to(torch.uint8).cpu().numpy()
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        # PLANAR, to match the canvas: the layer is the starting surface, so handing it over as
        # ``(3,H,W)`` makes the per-frame cost a broadcast copy when the batch shares one appearance
        # (which is every default render) instead of a transpose of the whole batch.
        imgs = torch.stack(
            [
                torch.as_tensor(
                    _bake_terrain(tuple(int(c) for c in u[:3]), tuple(int(c) for c in u[3:])),
                    dtype=torch.uint8,
                    device=self.device,
                ).permute(2, 0, 1)
                for u in uniq
            ]
        )  # (U,3,H,W)
        if len(uniq) == 1:
            return imgs[0].unsqueeze(1)  # (3,1,H,W)
        return imgs.index_select(0, torch.as_tensor(inv, device=self.device).reshape(-1)).transpose(0, 1)  # (3,B,H,W)

    #: The car's ROW window. ``position`` moves the car along the whole track, so
    #: its columns are unbounded -- but its row is ``10 + (sin(3p)*0.45 + 0.55) * SCALE``, which is
    #: bounded by the sine for every state, and the quad reaches ``hypot(20, 20)`` from that centre
    #: at any rotation. The flag is at a constant position, so both of its axes are bounded. The
    #: wheels need no window at all: a disc is drawn as its own offsets.
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
        ang = torch.cos(3 * pos).view(B, 1)  # the source rotates by cos(3*pos) IN RADIANS
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

        # gfxdraw.vline, the flag pole
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
        """The whole track. Near ``+0.6`` the car hangs off the right edge, which is the case that
        needs SDL_gfx's line clipping — sampling only the reset ball would never reach it."""
        rng = rng or self.rng
        return rng.uniform(self.MINP, self.MAXP, (batch, 1)).astype(np.float32)

    def step(self, action) -> TensorDict:
        """One control step. ``action`` is ``(B,1)``: **0 push left, 1 coast, 2 push right**, floored."""
        return super().step(action)

    @classmethod
    def reference_set_state(cls, env, st) -> None:
        env.state = np.array([st[0], 0.0])

    @classmethod
    def reference_sim_state(cls, ref) -> np.ndarray:
        return np.asarray(ref.state, np.float64).reshape(2)


@register
class MountainCarContinuous(MountainCar):
    """MountainCarContinuous-v0. Nearly two envs for the price of one — but not quite.

    ``continuous_mountain_car.render`` is byte-identical to ``mountain_car.render`` apart from one
    comment (checked by diffing the two sources), which makes it very tempting to reuse the renderer
    unchanged. That is wrong by 23,984 pixels across 64 frames, because the flag is drawn from
    ``self.goal_position`` and the two envs do not agree on it: **0.45 here, 0.5 for MountainCar-v0**,
    with a source comment attributing the difference to a third-party port. Identical draw code over a
    different constant is exactly the class of error a "looks the same" check misses and a pixel diff
    catches immediately.

    The same trap runs through the physics, one level deeper: ``step`` here takes one CONTINUOUS force
    in [-1, 1] scaled by ``power``, and it computes in **float32** where MountainCar-v0 computes in
    float64 (see ``_dynamics``). Seeded from a state the reference has already stored, agreement is
    **exact — exact_frac 1.0, max deviation 0.0 over 16 x 100 steps**; seeded from a freshly reset
    float64 state the reference's own first step is in a different dtype regime and the trajectory
    differs by up to one float32 ulp of position (1.2e-07).
    """

    name = "mountain-car-continuous"
    reference_id = "MountainCarContinuous-v0"
    GOALP = 0.45

    #: and the dynamics differ too, which the identical render code also hides: a continuous force
    #: scaled by ``power`` instead of a three-way push, and no ``-gravity*cos`` sign flip to spot.
    n_actions = 0
    action_low, action_high = -1.0, 1.0
    POWER = 0.0015

    def _dynamics(self, s, a):
        """``Continuous_MountainCarEnv.step`` — and this one is a **float32** simulator.

        ``self.state = np.array([position, velocity], dtype=np.float32)`` at the end of every step,
        while the discrete MountainCar keeps float64. Under NEP 50 every product in the update has a
        float32 operand and a weak python scalar, so ``force * power`` and the subtraction are float32
        as well; only the cosine is evaluated in double, on a float32 argument, and rounded on its way
        back. Computing the whole thing in float64 is the natural reading and diverges from the
        reference immediately, which is why the casts below are load-bearing rather than tidy-up.

        One consequence, stated because it bounds the fidelity claim: the reference's FIRST step after
        ``reset()`` still reads the float64 state that ``reset`` left behind, and only writes float32 on
        the way out. This implements the steady-state rule, so a step taken from a freshly-reset
        float64 state can differ in the last bits; every step from a state the reference has already
        stored is bit-identical.
        """
        force = a[:, 0].clamp(self.action_low, self.action_high)  # float32
        pos, vel = s[:, 0].to(torch.float32), s[:, 1].to(torch.float32)
        cos = torch.cos((3 * pos).to(torch.float64))  # float64 cos, f32 arg
        vel = vel + (force * self.POWER - (self.GRAVITY * cos).to(torch.float32))
        vel = vel.clamp(-self.MAX_SPEED, self.MAX_SPEED)
        pos = (pos + vel).clamp(self.MINP, self.MAXP)
        vel = torch.where((pos == np.float32(self.MINP)) & (vel < 0), torch.zeros_like(vel), vel)
        return torch.stack([pos.to(torch.float64), vel.to(torch.float64)], 1)

    def step(self, action) -> TensorDict:
        """One control step. ``action`` is ``(B,1)``: **a continuous force, clipped to [-1, 1]** — not
        MountainCar-v0's three-way push, which is the second thing these two envs do not share."""
        return super().step(action)


# =============================================================================== bakes
def _mc_height(x):
    """``MountainCarEnv._height``. ``np.sin``, not ``math.sin``, because that is what the reference
    calls and the result feeds an ``int()``."""
    return float(np.sin(3 * x) * 0.45 + 0.55)


#: Bounded because the key is a colour pair and a randomised style draws a new one every call; an
#: unbounded dict would grow by 720 kB per distinct appearance for the lifetime of the process.
_TERRAIN_CACHE_MAX = 512
_TERRAIN_CACHE = _RES.BoundedCache(_TERRAIN_CACHE_MAX, name="classic.terrain")
_SPRITE_CACHE: dict = {}


def _bake_terrain(bg, terrain) -> np.ndarray:
    """MountainCar's ``pygame.draw.aalines`` terrain, rasterised once. ``(400,600,3)`` uint8, pre-flip."""
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
    """Pendulum's torque arrow at every reachable size: ``(2, 114, 113, 113, 4)`` uint8 on device.

    Indexed ``[flip, size]``, zero-padded. ``smoothscale`` truncates its target size, so ``|u|`` in
    [0, 2] produces exactly 114 distinct sizes; both flip variants come from pygame so no flip is
    reimplemented. 11.6 MB resident, built once per process.
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
