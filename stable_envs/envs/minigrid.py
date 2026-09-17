"""MiniGrid and BabyAI — the family where the state-to-frame claim is exactly true, not approximately.

Reference: the ``minigrid`` package (3.1.0), covering **178 registered ids: 76 MiniGrid + 96 BabyAI +
6 WFC**. Every one of the 172 that instantiates is a stock ``MiniGridEnv`` holding a stock ``Grid``,
with ``get_full_render`` / ``get_pov_render`` / ``get_frame`` all the unoverridden base-class methods.
There is no per-env render code anywhere in the package and no ``WorldObj`` subclass outside
``core/world_object.py``, so nothing in the family breaks the tile model — no continuous positions, no
per-env art. (The 6 ``MiniGrid-WFC-*`` ids fail to construct, but for a packaging reason: the wheel
ships ``wfc/patterns.py`` without the ``patterns/*.png`` it loads. Their ``_gen_grid`` emits an
ordinary ``Wall``/``Goal`` grid.)

Because the frame is a gather from a precomputed atlas rather than a rasterisation, parity is not a
tuning exercise. Measured offline across the whole family: **21,216 frames — all 172 instantiable ids
x 3 render modes x 2 tile sizes, plus a stress pass over the full encode alphabet — every one
bit-identical to the official renderer, max absolute pixel difference 0.** The conformance suite
reproduces that on the classes below: every one reports ``exact_frac 1.0, max_abs 0``, so they hold
the default ``parity_interior_min = 1.0`` with nothing relaxed.

Throughput on one H200, through this class rather than a prototype: **6.55M agent-POV frames/s** at
224x224 (987 GB/s) against 2,378 f/s for the official CPU renderer, and 1.30M 256x256 top-down
frames/s. The top-down path is the slower one and not because of the blit: it must also compute the
field-of-view highlight, whose ~25 small kernels dominate at these frame sizes. Capturing the whole
call as a CUDA graph — which the index path is written to allow, see the scratch-slot scatter below —
measured 2-4x on top at small batch.

Compression, at the float32 the contract stores states in: a 8x8 top-down state is 268 bytes against
a 196,608-byte frame (**734x**), and the 7x7 POV state is 196 bytes against 150,528 (**768x**). Packed
to the one byte per cell the alphabet actually needs, the same states are 70 and 49 bytes — 2,809x and
3,072x — which is what a stored corpus would use.

What the state is, and what it is not:

  * **The full top-down frame does not depend on ``carrying``.** A picked-up object leaves the grid,
    and ``get_full_render`` touches ``carrying`` only through ``gen_obs_grid``, which writes it into
    the sliced view *after* ``process_vis`` — so it changes neither the visibility mask nor the world
    grid. The top-down classes therefore carry no such dimension, and that is checked, not asserted.
  * **``Box.contains`` is invisible.** ``Box.render`` ignores it and ``Box.encode`` drops it; verified
    at pixel level against a box with and without contents.
  * **This is the state sufficient to render, and — for the top-down classes — very nearly the state
    sufficient to simulate.** Those classes do also ship a ``step`` (see Physics below). It needs one
    thing this state does not carry, the object in the agent's hands, which the frame provably never
    shows; and it cannot express one transition, ``Box.toggle`` on a box with CONTENTS, which the
    reference bridge refuses rather than approximates. The mission string, ``step_count`` and the RNG
    govern the reward and the episode boundary, not the transition.

Cells are stored as one index into the 58 reachable ``(type, colour, state)`` encodings rather than as
the raw 3-channel encoding. That is not only 3x smaller — it is the only layout that can satisfy the
contract. In the raw encoding the state channel of a non-door cell provably cannot change the picture
(only ``Door`` reads it), so those dimensions would be dead, and a state with dead dimensions is either
not minimal or not fully rendered. One code per cell makes every dimension load-bearing.

Physics
-------

The top-down classes also ship ``step`` — a batched GPU port of ``MiniGridEnv.step`` — so
``env.sample(steps, batch, policy)`` runs on device. **This is the one family in the repo where the
simulator's claim is as strong as the renderer's**, and that is structural rather than a matter of
effort: the transition is a finite decision table over a finite alphabet (turn, move onto a cell that
can be overlapped, pick up, drop, toggle a door or a box), so a port either agrees on every
(state, action) pair or has a bug. There is no chaos to hide behind and no tolerance to negotiate.

Measured two ways, because a random rollout is a weak test of a decision table
(``python scratch/step_bench.py --what all`` reproduces both):

  * 16 episodes x 100 steps of uniform random actions per env, all six registered classes:
    **exact_frac 1.0, max deviation 0** on every cell, the agent, its direction and the carried object.
  * the whole cross product — **all 7 actions from each of 256 states, 1,792 transitions per env,
    10,752 in total, 0 mismatched**. Half the states are given a randomly chosen carried object,
    because a rollout that starts empty-handed never once exercises ``drop`` or a locked door, and
    those are exactly the transitions a decision table gets wrong.

Throughput on one H200, batched GPU step against one official CPU env (env-steps/s):

    env                       B=256      B=4,096     B=16,384    official CPU    step ms 256->16k
    8x8 (Empty/DoorKey/...)   1.69 M      23.3 M      80.0 M      8,100-10,700   0.152 -> 0.205
    19x19 (FourRooms)         1.67 M      19.4 M      48.4 M          8,700      0.153 -> 0.338

i.e. **158x the official env at B=256 and 7,500x at B=16,384** on the 8x8 grids. The step is
launch-bound at small batch — 0.152 ms whether it is 256 environments or 4,096 — which is why it is
captured into a CUDA graph (3.2-4.8x, measured; see ``_stepgraph``). The 19x19 grid is the one that
stops being launch-bound: 361 cells x 16,384 environments is real gather traffic and the step time
doubles.

The POV classes do NOT ship a step, and cannot: their state is an OBSERVATION — the 7x7 window after
occlusion — which does not determine the world it was cut out of, so there is nothing to step. That is
the same distinction the repo draws everywhere else, in its sharpest form: the state is sufficient to
RENDER and it is not sufficient to SIMULATE.

Two classes of environment are exposed, because they answer different questions:

  * ``MiniGrid*`` — the full top-down view, state ``[cell codes..., agent_x, agent_y, agent_dir]``.
  * ``MiniGridPOV*`` — the agent-centric 7x7 view, whose state is exactly MiniGrid's own observation
    (``gen_obs()["image"]``), 49 numbers. The carried object needs no dimension of its own: ``gen_obs``
    already writes it into the view's centre-bottom cell before encoding.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..env import Env
from ..registry import register
from ..render import tilemap as tm
from ..spaces import Box, Dict, Discrete, RGBBox
from . import _stepgraph

VIEW_SIZE = 7                      #: agent_view_size; 7 for every registered MiniGrid/BabyAI env
WALL_CODE = (tm.T_WALL, tm.C_GREY, 0)      #: Grid.slice pads out-of-bounds with a grey Wall()

#: ``Grid.encode`` writes this triple for a cell holding ``None``, so it doubles as "the agent is
#: carrying nothing": one alphabet for cells and for the carried object, and ``drop`` needs no special
#: case for the value it writes back.
EMPTY_CODE = tm.REACHABLE.index((tm.T_EMPTY, 0, 0))
CARRY_NONE = EMPTY_CODE
WALL_CODE_IDX = tm.REACHABLE.index(WALL_CODE)

_RULES: dict[str, dict] = {}


def _transition_tables(device) -> dict:
    """Per-code lookup tables for MiniGrid's transition rules, ASKED of the official classes.

    ``can_overlap`` and ``can_pickup`` are read off the objects ``make_obj`` builds, and the action ids
    off the ``Actions`` enum, rather than being re-derived from a list of type names here. The rules are
    one-line methods that would be trivial to copy, and copying them is exactly how a step drifts from
    its reference in a way no test names: upstream changes ``Door.can_overlap`` and a hardcoded table
    keeps saying what it said. This way the tables change with the package or fail loudly.
    """
    key = str(device)
    if key not in _RULES:
        from minigrid.core.actions import Actions
        from minigrid.core.constants import DIR_TO_VEC
        objs = [tm.make_obj(*c) for c in tm.REACHABLE]
        L = lambda v: torch.tensor(v, dtype=torch.long, device=device)
        B = lambda f: torch.tensor([bool(f(o)) for o in objs], dtype=torch.bool, device=device)
        _RULES[key] = {
            "type": L([c[0] for c in tm.REACHABLE]),
            "color": L([c[1] for c in tm.REACHABLE]),
            "state": L([c[2] for c in tm.REACHABLE]),
            # ``if fwd_cell is None or fwd_cell.can_overlap()`` — the three "no object" codes decode
            # to None, so they fold into the same table rather than needing a second test
            "overlap": B(lambda o: o is None or o.can_overlap()),
            "pickup": B(lambda o: o is not None and o.can_pickup()),
            "none": B(lambda o: o is None),
            #: (colour, state) -> code, for writing a door back after a toggle
            "door": L([[tm.REACHABLE.index((tm.T_DOOR, c, s)) for s in range(tm.N_STATE)]
                       for c in range(tm.N_COLOR)]),
            "dir_vec": L([list(v) for v in DIR_TO_VEC]),
            "actions": {n: int(getattr(Actions, n)) for n in
                        ("left", "right", "forward", "pickup", "drop", "toggle", "done")},
        }
    return _RULES[key]


def _code_table(device) -> torch.Tensor:
    """(58,3) int64: code index -> (type, colour, state)."""
    return torch.tensor(tm.REACHABLE, dtype=torch.long, device=device)


def _code_lookup(device) -> torch.Tensor:
    """(11,6,3) int64: (type, colour, state) -> code index. Unreachable triples map to 'empty'."""
    out = torch.full((tm.N_TYPE, tm.N_COLOR, tm.N_STATE), 0, dtype=torch.long, device=device)
    for i, (t, c, s) in enumerate(tm.REACHABLE):
        out[t, c, s] = i
        if t in (tm.T_UNSEEN, tm.T_EMPTY, tm.T_AGENT, tm.T_LAVA):
            out[t] = i                      # colour/state are not encoded for these types
    return out


class _TileEnv(Env):
    """Shared machinery: variation space, atlas plumbing, and the blit. Not registered itself."""

    grid_w: int = 8
    grid_h: int = 8
    see_through_walls: bool = False
    has_lava: bool = False
    tile_size_default: int = 32

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.atlas = tm.MiniGridAtlas.get(self._tile(self.size), device=self.device)
        self.code_table = _code_table(self.device)
        self.code_lookup = _code_lookup(self.device)
        self._pool: np.ndarray | None = None
        self._default_t: torch.Tensor | None = None

    # geometry

    def _tile(self, size: int) -> int:
        t, r = divmod(int(size), self.grid_h)
        if r or t < 3:
            raise ValueError(
                f"{type(self).__name__}: size {size} is not a whole tile size for a {self.grid_w}x"
                f"{self.grid_h} grid; use a multiple of {self.grid_h} that is at least {3*self.grid_h}")
        return t

    # variation

    def make_variation_space(self) -> Dict:
        """Appearance knobs, all defaulting to MiniGrid's own constants.

        ``tile_size`` is deliberately absent: it changes the output resolution rather than the
        appearance, and the contract already carries it as ``render(size=...)``. Declaring it twice
        would let the two disagree.

        ``lava_color`` appears only on envs that actually contain lava. MiniGrid hardcodes
        ``(255,128,0)`` in ``Lava.render`` instead of going through ``COLORS``, so it needs its own
        knob — but on a lava-free env it could not change any pixel, and a knob that cannot move the
        picture is exactly what the conformance suite exists to reject.

        **``chrome`` is one leaf and not three named RGB knobs, and that is a claim about MiniGrid
        rather than a packaging preference.** A MiniGrid frame is a palette applied to a label map and
        then composited with the field-of-view overlay, and ``get_full_render`` highlights the agent's
        own cell unconditionally (``process_vis`` seeds its mask there), while ``get_pov_render`` runs
        with an all-true visibility mask and therefore highlights **every** tile. So the agent colour
        is never painted anywhere, and on the POV view no colour is: what reaches the frame is
        ``floor(c + alpha*((highlight - c) mod 256))``. Declaring them as RGB knobs would advertise
        "this colour appears in the frame", which the conformance suite checks and which is not true
        here. They are palette entries, exactly like ``palette``'s six, and are declared as such.

        ``highlight_overlay`` is RGBA in one leaf for the same reason: the overlay colour is blended,
        never painted, so its colour and its opacity only mean anything together.
        """
        space = {
            "palette": Box(low=0, high=255, shape=(tm.N_COLOR, 3), dtype=np.uint8,
                           init_value=tm.official_colors()),
            #: rows: background, grid line, agent
            "chrome": Box(low=0, high=255, shape=(3, 3), dtype=np.uint8,
                          init_value=np.array([tm.DEFAULT_BG, tm.DEFAULT_GRIDLINE,
                                               tm.DEFAULT_AGENT], np.uint8)),
            "highlight": Discrete(2, init_value=1),
            "highlight_overlay": Box(low=np.zeros(4, np.float32),
                                     high=np.array([255.0, 255.0, 255.0, 1.0], np.float32),
                                     shape=(4,), dtype=np.float32,
                                     init_value=[*tm.DEFAULT_HIGHLIGHT,
                                                 tm.DEFAULT_HIGHLIGHT_ALPHA]),
        }
        if self.has_lava:
            space["lava_color"] = RGBBox(init_value=tm.DEFAULT_LAVA)
        return Dict(space)

    # style

    def _values(self, row: np.ndarray) -> dict:
        """One flat style row -> the ``{name: value}`` dict ``tilemap`` recolours an atlas from.

        Colours are truncated to integers here because that is what every other write of a style value
        into a frame does in this repo, and because ``label_palette`` then needs no second rounding
        rule for the half and 0.45 shades.
        """
        d = self.variation_space.unpack(row)
        chrome = np.trunc(np.asarray(d["chrome"], np.float64)).reshape(3, 3)
        ov = np.asarray(d["highlight_overlay"], np.float64).reshape(4)
        out = {"palette": np.trunc(np.asarray(d["palette"], np.float64)).reshape(tm.N_COLOR, 3),
               "background_color": chrome[0], "grid_line_color": chrome[1],
               "agent_color": chrome[2],
               "highlight_color": np.trunc(ov[:3]), "highlight_alpha": float(ov[3])}
        if self.has_lava:
            out["lava_color"] = np.trunc(np.asarray(d["lava_color"], np.float64))
        return out

    def _shared_style(self, sty) -> bool:
        """True when every environment asked for the same appearance, so one atlas serves the batch.

        Answered on the device and read back as a single value: the alternative, copying the whole
        ``(B, style_dim)`` tensor to the host to compare it there, is half a megabyte of traffic per
        call at B=4096 for a question with a one-bit answer.
        """
        v = sty.values
        return v.shape[0] == 1 or bool((v == v[:1]).all())

    def _is_default_style(self, sty) -> bool:
        if self._default_t is None:
            self._default_t = torch.as_tensor(self.default_style(), device=self.device,
                                              dtype=torch.float32)
        return bool((sty.values[0] == self._default_t).all())

    def _label_palettes(self, sty) -> torch.Tensor:
        """``(B, 2*N_LABEL, 3)`` float64: ``label_palette`` and ``highlight_palette``, per environment.

        The same arithmetic as ``tilemap.label_palette`` / ``highlight_palette``, kept on the device
        and batched, so a 4096-environment batch of different palettes is one tensor rather than 4096
        recoloured atlases — an atlas is 1980 tiles and a frame uses at most H*W of them, so
        recolouring per environment would do far more work than drawing.
        """
        B, dev = sty.batch, self.device
        pal = torch.trunc(sty["palette"].reshape(B, tm.N_COLOR, 3).double())
        chrome = torch.trunc(sty["chrome"].reshape(B, 3, 3).double())
        p = torch.zeros(B, tm.N_LABEL, 3, dtype=torch.float64, device=dev)
        p[:, tm.LBL_BG] = chrome[:, 0]
        p[:, tm.LBL_GRID] = chrome[:, 1]
        p[:, tm.LBL_AGENT] = chrome[:, 2]
        p[:, tm.LBL_LAVA] = (torch.trunc(sty["lava_color"].double()) if "lava_color" in sty
                             else torch.tensor(tm.DEFAULT_LAVA, dtype=torch.float64,
                                               device=dev).expand(B, 3))
        p[:, tm.LBL_PAL:tm.LBL_PAL + tm.N_COLOR] = pal
        p[:, tm.LBL_PAL_HALF:tm.LBL_PAL_HALF + tm.N_COLOR] = torch.trunc(pal / 2)
        p[:, tm.LBL_PAL_45:tm.LBL_PAL_45 + tm.N_COLOR] = torch.trunc(0.45 * pal)
        ov = sty["highlight_overlay"].double()
        # the reference's uint8 subtraction, so a highlight darker than the pixel wraps exactly as
        # ``highlight_img`` makes it wrap rather than as it arguably meant to
        diff = torch.remainder(torch.trunc(ov[:, :3]).unsqueeze(1) - p, 256.0)
        ph = torch.floor(torch.clamp(p + ov[:, 3].view(B, 1, 1) * diff, 0, 255))
        return torch.cat([p, ph], 1)

    # rendering

    @torch.no_grad()
    def render(self, states=None, size: int | None = None, supersample: int | None = None,
               variation: dict | None = None, style=None, out=None) -> torch.Tensor:
        """``(...,state_dim)`` -> ``(...,H,W,3)`` uint8 on device, for arbitrary leading dims."""
        if supersample not in (None, 1):
            raise ValueError(
                f"{type(self).__name__}: supersample={supersample} is not available. MiniGrid's own "
                f"3x supersampling is already baked into the atlas by Grid.render_tile; sampling "
                f"again on top of it would render something the reference never draws.")
        s = self._state if states is None else torch.as_tensor(states, device=self.device,
                                                               dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        size = int(size or self.size)
        tile = self._tile(size)
        atlas = (self.atlas if tile == self.atlas.T
                 else tm.MiniGridAtlas.get(tile, device=self.device))
        sty = self.resolve_style(style, variation, batch=s.shape[0])
        idx = self._tile_indices(s, sty)
        flat_out = None
        if out is not None:
            if not out.is_contiguous():
                raise ValueError(f"{type(self).__name__}.render: `out` must be contiguous — the "
                                 f"kernel writes through a flat view of it")
            # flatten the leading dims rather than dropping `out`; a reshape of a contiguous tensor
            # is a view, so the kernel still writes into the caller's buffer
            flat_out = out.reshape(-1, self.grid_h * tile, self.grid_w * tile, 3)
        if self._shared_style(sty):
            # One appearance for the whole batch: recolour the 1980-tile atlas once and blit. This is
            # the fast path the throughput numbers were measured on, and the default style hits the
            # official atlas outright rather than reconstructing it.
            rgb = (atlas.official if self._is_default_style(sty)
                   else atlas.rgb(self._values(sty.values[0].detach().cpu().numpy())))
            img = tm.blit(rgb, idx, tile, out=flat_out)
        else:
            img = self._blit_styled(atlas, idx, sty, tile, flat_out)
        return img.reshape(*lead, *img.shape[1:]) if lead else img

    def _blit_styled(self, atlas, idx: torch.Tensor, sty, tile: int, out) -> torch.Tensor:
        """Per-environment palettes: gather LABELS, colour them per environment, then downsample.

        The shared-atlas path cannot express this — an atlas is keyed on a palette, so B palettes mean
        B atlases (97 MB at B=16, 25 GB at B=4096). Colouring at blit time instead costs one pass over
        the 3x supersampled frame and is arithmetically identical to ``tilemap.recolour`` followed by
        the gather: every subpixel has exactly one label, the palette lookup is per subpixel either
        way, and the box filter is the same two-stage ``mean`` in the same order and dtype.
        """
        B, H, W = idx.shape
        dev = self.device
        f = tm.SUBDIVS
        px = f * tile
        pal = self._label_palettes(sty)                                   # (B, 2*N_LABEL, 3)
        geom = idx.long() % tm.N_GEOM
        hl = idx.long() // tm.N_GEOM
        if out is None:
            out = torch.empty(B, H * tile, W * tile, 3, dtype=torch.uint8, device=dev)
        # chunk the batch so the float64 working set stays bounded; a 19x19 grid at 32 px is a
        # 1824x1824 subpixel frame, which is 80 MB of float64 per environment
        step = max(1, int((256 << 20) // (px * W * px * 3 * 8)))
        for b0 in range(0, B, step):
            b1 = min(B, b0 + step)
            bi = torch.arange(b0, b1, device=dev).view(-1, 1, 1)
            for j in range(H):
                lab = atlas.labels[geom[b0:b1, j]].long()                 # (b, W, px, px)
                key = lab + hl[b0:b1, j].view(-1, W, 1, 1) * tm.N_LABEL
                key = key.permute(0, 2, 1, 3).reshape(b1 - b0, px, W * px)
                col = pal[bi, key]                                        # (b, px, W*px, 3) f64
                col = col.reshape(b1 - b0, tile, f, W * tile, f, 3).mean(4).mean(2)
                out[b0:b1, j * tile:(j + 1) * tile] = col.to(torch.uint8)
        return out

    def _tile_indices(self, s: torch.Tensor, sty) -> torch.Tensor:
        raise NotImplementedError

    def _decode_cells(self, codes: torch.Tensor) -> torch.Tensor:
        """Float cell codes -> (…,3) int64 (type, colour, state).

        Wrapping rather than clamping: the conformance suite proves each dimension reaches the render
        by adding a constant to it, and a clamp would silently pin an already-large code in place and
        report a live dimension as dead.
        """
        return self.code_table[torch.round(codes).long() % tm.N_CODE]

    # states

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """States drawn from real rollouts of the official env, not from a synthetic prior.

        Layouts here are generated by procedural code with strong invariants (a wall ring, doors in
        walls, one goal), so uniform noise over the code alphabet would exercise a region of state
        space no episode ever visits and prove nothing about the frames anyone would actually store.
        """
        rng = rng or self.rng
        if self._pool is None:
            self._pool = self._build_pool(max(256, batch))
        if batch <= len(self._pool):
            return self._pool[rng.choice(len(self._pool), batch, replace=False)].copy()
        return self._pool[rng.integers(0, len(self._pool), batch)].copy()

    def _build_pool(self, n: int) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def _official(cls):
        import gymnasium as gym
        import minigrid  # noqa: F401  (registers the ids)
        e = gym.make(cls.reference_id)
        e.reset(seed=0)
        cls._check_source(e.unwrapped)
        return e

    @classmethod
    def _check_source(cls, u) -> None:
        """Re-derive from the official env what this class declared, rather than trusting the table.

        Grid sizes and ``see_through_walls`` are hardcoded per id so that ``state_dim`` exists before
        an env is ever built, which the base class needs. Hardcoded means it can drift when the
        upstream package changes a level generator, and a wrong value here is silent: the frame comes
        out the wrong shape, or the highlight is computed from the wrong visibility rule.
        """
        if bool(u.see_through_walls) != cls.see_through_walls:
            raise AssertionError(
                f"{cls.__name__} declares see_through_walls={cls.see_through_walls} but "
                f"{cls.reference_id} uses {bool(u.see_through_walls)}; the visibility mask, and so "
                f"the highlight, would be wrong")


# ---------------------------------------------------------------- full top-down view

class MiniGridTopDown(_TileEnv):
    """``get_full_render``: the whole layout, with the agent's field of view highlighted."""

    # ------------------------------------------------------------------ physics
    #
    # The transition is EXACT, not approximate, and that is a structural difference from every
    # continuous env in this repo rather than a matter of effort. MiniGrid's step is a finite decision
    # table over a finite alphabet — turn, step onto a cell if it can be overlapped, take an object,
    # put it down, toggle a door or a box — so a batched GPU port either agrees with the reference on
    # every (state, action) pair or has a bug. Measured against the official ``MiniGridEnv.step``:
    # exact_frac 1.0 over 16 x 100 random steps AND over all 7 actions from each of 256 states, on all
    # six registered classes (see the module docstring). ``exact_frac`` is the only fidelity statistic
    # worth quoting here — a "median error" for a grid would be a category mistake.
    #
    # ``carrying`` is the one piece of dynamical state the compact state does not carry, for the reason
    # the module docstring gives: ``get_full_render`` provably never reads it, so it is not part of the
    # RENDER state, and putting it there would add a dimension the conformance suite would rightly
    # report as dead. It therefore lives beside the state the way PushT keeps its agent's velocity, and
    # ``set_state`` empties the agent's hands — the condition the reference is in after ``reset``.

    #: MiniGrid's action alphabet, in ``Actions`` order: 0 turn left, 1 turn right, 2 forward,
    #: 3 pick up, 4 drop, 5 toggle, 6 done. A policy emits ONE float per environment and ``step``
    #: floors it, so ``RandomPolicy`` over the half-open ``[0, 7)`` bounds below is uniform over the
    #: seven actions and an integer-valued tensor works unchanged.
    N_ACTIONS = 7
    action_dim = 1
    action_low = 0.0
    action_high = 7.0
    use_step_graph = True
    step_reference_checkable = True
    step_reference_exact = True
    #: 1, not the PushT default: a turn held for eight steps just spins the agent twice, and what this
    #: alphabet needs is coverage of the transitions, not sustained pressure on a solver.
    step_reference_hold = 1

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._rules = _transition_tables(self.device)
        self._A = self._rules["actions"]
        z = lambda v: torch.tensor(int(v), dtype=torch.long, device=self.device)
        self._empty, self._wall, self._none = z(EMPTY_CODE), z(WALL_CODE_IDX), z(CARRY_NONE)
        self._carry = torch.full((self.B,), CARRY_NONE, dtype=torch.long, device=self.device)
        self._act = torch.zeros(self.B, self.action_dim, device=self.device, dtype=torch.float32)
        self._step_graph = None
        self._step_graph_error = ""

    def set_state(self, states, style=None) -> "MiniGridTopDown":
        """As the base class, and empty the agent's hands — the compact state does not carry them."""
        super().set_state(states, style=style)
        self._carry.fill_(CARRY_NONE)
        return self

    def sim_state(self) -> torch.Tensor:
        """``(B, state_dim + 1)`` — the compact state with ``carrying`` appended."""
        return torch.cat([self._state.to(torch.float64),
                          self._carry.to(torch.float64).unsqueeze(1)], 1)

    def set_sim_state(self, v) -> "MiniGridTopDown":
        v = torch.as_tensor(v, device=self.device, dtype=torch.float64).reshape(self.B, -1)
        self.set_state(v[:, :self.state_dim])
        self._carry.copy_(torch.round(v[:, self.state_dim]).long())
        return self

    def _advance(self) -> None:
        """``MiniGridEnv.step`` for the whole batch, as one straight-line sequence of masked writes.

        Every branch of the reference's if/elif chain is evaluated for every environment and selected
        by a mask, because a batch holds all seven actions at once; the chain is mutually exclusive on
        ``action``, so the masks cannot overlap and the order of the writes below does not matter.

        Out-of-bounds is the one place this must decide something the reference never does: ``Grid.get``
        asserts, and cannot be called off the grid because every MiniGrid layout has a wall ring, so no
        agent can face outward from a legal state. A state assembled by hand can, and the conservative
        reading is taken — the cell beyond the edge behaves as a wall, and nothing is written there.
        """
        R, A, s = self._rules, self._A, self._state
        W, H, N = self.grid_w, self.grid_h, self.grid_w * self.grid_h
        a = self._act[:, 0].floor().clamp(0, self.N_ACTIONS - 1).long()
        codes = torch.round(s[:, :N]).long() % tm.N_CODE
        ax = torch.round(s[:, N]).long() % W
        ay = torch.round(s[:, N + 1]).long() % H
        ad = torch.round(s[:, N + 2]).long() % 4
        carry = self._carry

        d = R["dir_vec"][ad]                                        # front_pos = agent_pos + dir_vec
        fx, fy = ax + d[:, 0], ay + d[:, 1]
        inb = (fx >= 0) & (fx < W) & (fy >= 0) & (fy < H)
        fi = (fx.clamp(0, W - 1) * H + fy.clamp(0, H - 1)).unsqueeze(1)
        held = codes.gather(1, fi).squeeze(1)
        cell = torch.where(inb, held, self._wall)
        ctype, ccol, cstate = R["type"][cell], R["color"][cell], R["state"][cell]

        ad_new = torch.where(a == A["left"], (ad - 1) % 4,
                             torch.where(a == A["right"], (ad + 1) % 4, ad))
        move = (a == A["forward"]) & R["overlap"][cell]
        ax_new, ay_new = torch.where(move, fx, ax), torch.where(move, fy, ay)

        pick = (a == A["pickup"]) & R["pickup"][cell] & (carry == CARRY_NONE)
        drop = (a == A["drop"]) & R["none"][cell] & (carry != CARRY_NONE) & inb
        tog = a == A["toggle"]
        # Door.toggle: a locked door opens only for a key of ITS OWN colour, and an unlocked one just
        # flips. A locked door that is toggled without the key is left exactly as it was.
        keyed = (R["type"][carry] == tm.T_KEY) & (R["color"][carry] == ccol)
        unlock = tog & (ctype == tm.T_DOOR) & (cstate == 2) & keyed
        flip = tog & (ctype == tm.T_DOOR) & (cstate != 2)
        openbox = tog & (ctype == tm.T_BOX)

        new = torch.where(pick, self._empty, cell)
        new = torch.where(drop, carry, new)
        new = torch.where(unlock, R["door"][ccol, 0], new)
        new = torch.where(flip, R["door"][ccol, 1 - cstate], new)
        new = torch.where(openbox, self._empty, new)
        new = torch.where(inb, new, held)              # off-grid: write the cell back unchanged

        s[:, :N] = codes.scatter(1, fi, new.unsqueeze(1)).to(torch.float32)
        s[:, N] = ax_new.to(torch.float32)
        s[:, N + 1] = ay_new.to(torch.float32)
        s[:, N + 2] = ad_new.to(torch.float32)
        self._carry.copy_(torch.where(pick, cell, torch.where(drop, self._none, carry)))

    @torch.no_grad()
    def step(self, action) -> torch.Tensor:
        """Advance every environment one step. ``action`` is ``(B, 1)``, floored to ``[0, 7)``.

        Returns the new ``(B, state_dim)`` compact state. Termination is not applied here and the reward
        is not computed: the reference reports both as flags beside a state it integrates regardless,
        so an env that froze or reset itself would be simulating something else. ``carrying`` is not in
        the returned state — see ``sim_state()`` — which means a stored trajectory does not determine
        its own continuation, exactly as it does not for PushT's agent velocity. It does determine every
        FRAME, which is the claim this repo makes.
        """
        self._act.copy_(torch.as_tensor(action, device=self.device, dtype=torch.float32
                                        ).reshape(self.B, self.action_dim))
        if self.use_step_graph and self.device.type == "cuda":
            if self._step_graph is None and not self._step_graph_error:
                self._step_graph, self._step_graph_error = _stepgraph.capture(
                    self._advance, (self._state, self._carry))
                if self._step_graph is None:
                    self.use_step_graph = False
            if self._step_graph is not None:
                self._step_graph.replay()
                return self._state
        self._advance()
        return self._state

    @classmethod
    def reference_sim_state(cls, ref) -> np.ndarray:
        """The reference's grid, agent and carried object in this class's own alphabet."""
        u = ref.unwrapped
        lut = _code_lookup(torch.device("cpu")).numpy()
        enc = u.grid.encode()
        codes = lut[enc[:, :, 0], enc[:, :, 1], enc[:, :, 2]].reshape(-1)
        carry = CARRY_NONE
        if u.carrying is not None:
            e = u.carrying.encode()
            carry = int(lut[e[0], e[1], e[2]])
        return np.concatenate([codes, u.agent_pos, [u.agent_dir], [carry]]).astype(np.float64)

    @classmethod
    def reference_step(cls, ref, a) -> None:
        """Drive the UNWRAPPED env, so ``TimeLimit`` cannot end an episode part-way through a
        comparison. Truncation is bookkeeping and the transition keeps being defined after it."""
        ref.unwrapped.step(int(min(max(math.floor(float(a[0])), 0), cls.N_ACTIONS - 1)))

    @classmethod
    def step_reference_reset(cls, ref, seed: int) -> np.ndarray:
        """Reset, and refuse to compare a layout this state cannot represent.

        ``Box.toggle`` replaces the box with its CONTENTS, and the contents are not in the state:
        ``Box.render`` ignores them and ``Box.encode`` drops them, which is exactly why the state does
        not carry them and why the render claim is unaffected. Every registered env here builds its
        boxes empty (BabyAI's distractors are ``Box(color)``), so toggling one empties the cell and the
        step stays exact — but a level that put something inside a box would step to a cell this
        alphabet cannot name, and saying so is better than quietly reporting a mismatch as drift.
        """
        ref.reset(seed=seed)
        u = ref.unwrapped
        if any(o is not None and o.type == "box" and o.contains is not None for o in u.grid.grid):
            raise NotImplementedError(
                f"{cls.__name__}: {cls.reference_id} builds a Box with contents, and Box.toggle "
                f"replaces the box by them. The contents are invisible to the renderer and so are not "
                f"in the state, which makes that one transition unrepresentable here.")
        return cls.reference_sim_state(ref)

    @classmethod
    def _check_source(cls, u) -> None:
        super()._check_source(u)
        if (u.width, u.height) != (cls.grid_w, cls.grid_h):
            raise AssertionError(
                f"{cls.__name__} declares a {cls.grid_w}x{cls.grid_h} grid but {cls.reference_id} "
                f"builds {u.width}x{u.height}; every rendered frame would be the wrong shape")

    def _tile_indices(self, s: torch.Tensor, sty) -> torch.Tensor:
        B, W, H = s.shape[0], self.grid_w, self.grid_h
        dev = self.device
        tcs = self._decode_cells(s[:, :W * H]).reshape(B, W, H, 3)          # [b, i, j, 3]
        ax = torch.round(s[:, W * H + 0]).long() % W
        ay = torch.round(s[:, W * H + 1]).long() % H
        ad = torch.round(s[:, W * H + 2]).long() % 4

        vis = self._visibility(tcs, ax, ay, ad)                            # (B,V,V) [b,vi,vj]
        offs = tm.view_offsets(VIEW_SIZE, dev)[ad]                         # (B,V,V,2) [vj,vi]
        wx = ax.view(B, 1, 1) + offs[..., 0]
        wy = ay.view(B, 1, 1) + offs[..., 1]
        keep = (wx >= 0) & (wx < W) & (wy >= 0) & (wy < H) & vis.permute(0, 2, 1)
        flat = (torch.arange(B, device=dev).view(B, 1, 1) * (H * W)
                + wy.clamp(0, H - 1) * W + wx.clamp(0, W - 1))
        # Off-grid and occluded view cells are routed to a scratch slot instead of being
        # compacted, so the scatter needs no device-to-host sync and the pipeline stays
        # CUDA-graph capturable.
        scratch = B * H * W
        flat = torch.where(keep, flat, torch.full_like(flat, scratch))
        acc = torch.zeros(scratch + 1, dtype=torch.long, device=dev)
        acc.scatter_(0, flat.reshape(-1), torch.ones_like(flat.reshape(-1)))
        # the highlight toggle is a per-environment mask rather than a python branch: a batch may
        # legitimately hold both settings, and masking is what makes that one launch
        hl = acc[:scratch].reshape(B, H, W) * (sty["highlight"] > 0.5).long().view(B, 1, 1)

        img = tcs.permute(0, 2, 1, 3)                                      # [b, j, i, 3]
        d = torch.full((B, H, W), tm.NO_AGENT, dtype=torch.long, device=dev)
        d[torch.arange(B, device=dev), ay, ax] = ad
        return tm.atlas_flat_index(hl, d, img[..., 0], img[..., 1], img[..., 2]).to(torch.int32)

    def _visibility(self, tcs, ax, ay, ad):
        """The agent's view, sliced/rotated by gather, then swept by ``Grid.process_vis``."""
        B, W, H, V = tcs.shape[0], self.grid_w, self.grid_h, VIEW_SIZE
        dev = self.device
        offs = tm.view_offsets(V, dev)[ad]
        wx = ax.view(B, 1, 1) + offs[..., 0]
        wy = ay.view(B, 1, 1) + offs[..., 1]
        oob = (wx < 0) | (wx >= W) | (wy < 0) | (wy >= H)
        flat = (torch.arange(B, device=dev).view(B, 1, 1) * (W * H)
                + wx.clamp(0, W - 1) * H + wy.clamp(0, H - 1))
        view = tcs.reshape(B * W * H, 3)[flat.reshape(-1)].reshape(B, V, V, 3)
        wall = torch.tensor(WALL_CODE, dtype=view.dtype, device=dev)
        view = torch.where(oob.unsqueeze(-1), wall, view).permute(0, 2, 1, 3)   # [b, vi, vj]
        if self.see_through_walls:
            return torch.ones(B, V, V, dtype=torch.bool, device=dev)
        t, st = view[..., 0], view[..., 2]
        transp = ~((t == tm.T_WALL) | ((t == tm.T_DOOR) & (st != 0)))           # see_behind()
        return tm.visibility_scan(transp, V)

    def _build_pool(self, n: int) -> np.ndarray:
        env = self._official()
        u = env.unwrapped
        lut = _code_lookup(torch.device("cpu")).numpy()
        out, ep = [], 0
        while len(out) < n:
            env.reset(seed=ep)
            rng = np.random.default_rng(ep)
            for _ in range(32):
                enc = u.grid.encode()
                codes = lut[enc[:, :, 0], enc[:, :, 1], enc[:, :, 2]].reshape(-1)
                out.append(np.concatenate([codes, u.agent_pos, [u.agent_dir]]))
                if len(out) >= n:
                    break
                _, _, te, tr, _ = env.step(int(rng.integers(0, 7)))
                if te or tr:
                    break
            ep += 1
        env.close()
        return np.stack(out).astype(np.float32)

    # reference bridge

    @classmethod
    def reference_env(cls, size: int | None = None):
        env = cls._official()
        env.unwrapped._se_tile = (int(size or cls.default_size)) // cls.grid_h
        return env

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Official frames, by writing each state into the reference env's own ``Grid``.

        This drives the real ``get_full_render``, so the whole official path runs: ``gen_obs_grid``,
        ``Grid.slice``, ``rotate_left``, ``process_vis``, the highlight scatter, and ``render_tile``.
        """
        from minigrid.core.grid import Grid
        u = env.unwrapped
        out = []
        for st in np.asarray(states):
            codes = np.round(st[:cls.grid_w * cls.grid_h]).astype(np.int64) % tm.N_CODE
            grid = Grid(cls.grid_w, cls.grid_h)
            for k, code in enumerate(codes):
                grid.set(k // cls.grid_h, k % cls.grid_h, tm.make_obj(*tm.REACHABLE[code]))
            u.grid, u.width, u.height = grid, cls.grid_w, cls.grid_h
            u.agent_pos = (int(round(st[-3])) % cls.grid_w, int(round(st[-2])) % cls.grid_h)
            u.agent_dir = int(round(st[-1])) % 4
            u.carrying = None                     # provably not read by get_full_render
            out.append(u.get_full_render(True, u._se_tile))
        return np.stack(out)


# ---------------------------------------------------------------- agent-centric view

class MiniGridPOVBase(_TileEnv):
    """``get_pov_render``: the agent's 7x7 window, state = MiniGrid's own observation.

    The highlight is not a separate input. ``gen_obs_grid`` blanks occluded cells and ``grid.encode``
    writes type ``unseen`` exactly there, while ``get_pov_render`` highlights exactly the visible
    cells — so ``highlight == (type != unseen)`` identically, and the visibility sweep has already
    happened by the time this state exists.

    A consequence worth stating, because it is what shaped the variation space: the source env must
    have ``see_through_walls=True`` (see ``pov``), so its visibility mask is all-true, so **every tile
    of a POV frame is highlighted** and every pixel is a blend of a palette colour with the overlay.
    No declared colour is ever painted literally here, which is why the label colours are one
    ``chrome`` leaf rather than named RGB knobs.

    No ``step``, and there cannot be one. This state is an OBSERVATION — the 7x7 window, after
    occlusion — and a window does not determine the world it was cut out of: the next observation
    depends on cells outside it, and on which of the objects inside it the agent may pick up next.
    ``MiniGridTopDown`` steps exactly the same dynamics because it holds the whole grid. The split is
    the repo's own distinction at its sharpest: this state is sufficient to RENDER, and it is not
    sufficient to SIMULATE.
    """

    grid_w = grid_h = VIEW_SIZE
    see_through_walls = True

    def _tile_indices(self, s: torch.Tensor, sty) -> torch.Tensor:
        B, V = s.shape[0], VIEW_SIZE
        dev = self.device
        tcs = self._decode_cells(s).reshape(B, V, V, 3).permute(0, 2, 1, 3)   # [b, vj, vi, 3]
        t, c, st = tcs[..., 0], tcs[..., 1], tcs[..., 2]
        hl = (t != tm.T_UNSEEN).long() * (sty["highlight"] > 0.5).long().view(B, 1, 1)
        d = torch.full((B, V, V), tm.NO_AGENT, dtype=torch.long, device=dev)
        d[:, V - 1, V // 2] = 3                        # the POV agent always points up
        return tm.atlas_flat_index(hl, d, t, c, st).to(torch.int32)

    def _build_pool(self, n: int) -> np.ndarray:
        env = self._official()
        lut = _code_lookup(torch.device("cpu")).numpy()
        out, ep = [], 0
        while len(out) < n:
            obs, _ = env.reset(seed=ep)
            rng = np.random.default_rng(ep)
            for _ in range(32):
                im = obs["image"]
                out.append(lut[im[:, :, 0], im[:, :, 1], im[:, :, 2]].reshape(-1))
                if len(out) >= n:
                    break
                obs, _, te, tr, _ = env.step(int(rng.integers(0, 7)))
                if te or tr:
                    break
            ep += 1
        env.close()
        return np.stack(out).astype(np.float32)

    @classmethod
    def reference_env(cls, size: int | None = None):
        env = cls._official()
        env.unwrapped._se_tile = (int(size or cls.default_size)) // VIEW_SIZE
        return env

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Official frames through the real ``get_pov_render``.

        The observation is written back as the env's whole grid with the agent at the view's
        centre-bottom facing up, which is a fixed point of the view transform: ``get_view_exts`` then
        returns ``(0,0)``, so ``Grid.slice`` is the identity and ``rotate_left`` runs four times on a
        square grid, also the identity. The env's source is chosen with ``see_through_walls=True`` so
        that ``gen_obs_grid`` does not re-run ``process_vis`` over an already-processed view — an
        occluded cell decodes to ``None``, which is transparent, so a second sweep would mark it
        visible and highlight a cell the reference leaves dark. The carried object is restored from
        the view's centre cell because ``gen_obs_grid`` overwrites that cell with ``self.carrying``.
        """
        from minigrid.core.grid import Grid
        u = env.unwrapped
        V = VIEW_SIZE
        out = []
        for st in np.asarray(states):
            codes = np.round(st).astype(np.int64) % tm.N_CODE
            grid = Grid(V, V)
            for k, code in enumerate(codes):
                grid.set(k // V, k % V, tm.make_obj(*tm.REACHABLE[code]))
            u.grid, u.width, u.height = grid, V, V
            u.agent_pos, u.agent_dir = (V // 2, V - 1), 3
            u.carrying = grid.get(V // 2, V - 1)
            out.append(u.get_pov_render(u._se_tile))
        return np.stack(out)


# ---------------------------------------------------------------- concrete registrations

def _cell_names(w: int, h: int) -> tuple[str, ...]:
    return tuple(f"cell[{i},{j}]" for i in range(w) for j in range(h))


def topdown(class_name: str, env_id: str, w: int, h: int, stw: bool, lava: bool = False,
            tile: int = 32, doc: str = "") -> type:
    """Build and register a top-down class for one official env id.

    A class per id rather than one parameterised env: the registry is the conformance suite's
    parameter list, so an id that is only reachable through a constructor argument is an id nothing
    tests. Declared grid size and ``see_through_walls`` are re-checked against the official env every
    time one is built.
    """
    n = w * h
    cls = type(class_name, (MiniGridTopDown,), {
        "__doc__": doc or f"{env_id}: {w}x{h} top-down view.",
        "__module__": __name__,
        "name": env_id, "reference_id": env_id,
        "grid_w": w, "grid_h": h, "see_through_walls": stw, "has_lava": lava,
        "state_dim": n + 3,
        "state_names": _cell_names(w, h) + ("agent_x", "agent_y", "agent_dir"),
        "default_size": h * tile,
        # the dynamical state is the render state plus the carried object; see MiniGridTopDown
        "sim_state_names": _cell_names(w, h) + ("agent_x", "agent_y", "agent_dir", "carrying"),
        # Reported as four groups rather than as w*h+4 one-dimensional ones: "which cell disagrees"
        # is not a useful readout for a 361-cell grid, "does any cell disagree" is. The agent's
        # position is euclidean so that a diagonal disagreement counts once.
        "step_reference_groups": (("cells", tuple(range(n)), "abs"),
                                  ("agent", (n, n + 1), "euclidean"),
                                  ("dir", (n + 2,), "abs"),
                                  ("carrying", (n + 3,), "abs")),
        # The conformance suite requires the class that CLAIMS a batched simulator to DECLARE one,
        # rather than to reach an unrelated ``step`` through a base class — that guard is what stops a
        # checker driving an env it knows nothing about. These classes are built by a factory, so the
        # declaration is made here explicitly; the seven actions are identical across every grid, so
        # there is nothing per-class to say about them beyond ``MiniGridTopDown.step``'s docstring.
        "step": MiniGridTopDown.step,
    })
    return register(cls)


def pov(class_name: str, env_id: str, tile: int = 32, doc: str = "") -> type:
    """Build and register an agent-POV class. The source env must have ``see_through_walls=True``.

    Not a restriction on what the renderer handles — the offline validation covers occluded views on
    all 172 ids — but on what can be *checked against the official renderer from a state alone*: with
    occlusion on, an observation cannot be written back into a live env without ``gen_obs_grid``
    recomputing visibility over it and disagreeing.
    """
    cls = type(class_name, (MiniGridPOVBase,), {
        "__doc__": doc or f"{env_id}: agent-centric {VIEW_SIZE}x{VIEW_SIZE} view.",
        "__module__": __name__,
        "name": env_id + "-pov", "reference_id": env_id,
        "state_dim": VIEW_SIZE * VIEW_SIZE,
        "state_names": tuple(f"view[{i},{j}]" for i in range(VIEW_SIZE) for j in range(VIEW_SIZE)),
        "default_size": VIEW_SIZE * tile,
    })
    return register(cls)


MiniGridEmpty = topdown(
    "MiniGridEmpty", "MiniGrid-Empty-8x8-v0", 8, 8, stw=True,
    doc="Empty room, 8x8. The simplest layout in the family: walls, one goal, one agent.")
MiniGridDoorKey = topdown(
    "MiniGridDoorKey", "MiniGrid-DoorKey-8x8-v0", 8, 8, stw=False,
    doc="Locked door and key, 8x8. Exercises all three door states and occluded visibility.")
MiniGridFourRooms = topdown(
    "MiniGridFourRooms", "MiniGrid-FourRooms-v0", 19, 19, stw=False,
    doc="Four rooms joined by gaps, 19x19 — the large-grid case, 608x608 at the default tile size.")
MiniGridLavaCrossing = topdown(
    "MiniGridLavaCrossing", "MiniGrid-LavaCrossingS9N2-v0", 9, 9, stw=False, lava=True,
    doc="Lava rivers with crossings, 9x9. The only registered env carrying the lava_color knob.")
MiniGridKeyCorridor = topdown(
    "MiniGridKeyCorridor", "MiniGrid-KeyCorridorS4R3-v0", 10, 10, stw=False,
    doc="Locked rooms off a corridor, 10x10: locked doors, keys and a ball behind them.")
BabyAIGoToLocal = topdown(
    "BabyAIGoToLocal", "BabyAI-GoToLocal-v0", 8, 8, stw=False,
    doc="BabyAI's instruction-following room, 8x8: keys, balls and boxes in six colours.")
MiniGridPOVFetch = pov(
    "MiniGridPOVFetch", "MiniGrid-Fetch-8x8-N3-v0",
    doc="Agent-centric 7x7 view of the Fetch room — MiniGrid's standard partial observation, "
        "224x224 from 49 numbers.")


def register_all(ids=None, tile: int = 32) -> list[str]:
    """Register a top-down class for every instantiable MiniGrid/BabyAI id (or a given subset).

    Not called at import. All 172 pass the same checks offline, but registering them here would put
    172 envs into a conformance run whose reference leg renders on the CPU, and a suite that takes ten
    minutes stops being run. The curated classes above cover the distinct cases: empty rooms, doors in
    every state, lava, large grids, BabyAI, and the partial view.

    One known false positive if you do register everything and run the suite. Agent coordinates are
    integers wrapped modulo the grid width, and ``check_state_dims_matter`` probes a dimension by
    adding a fixed ``eps=25.0``; on a 25-wide grid that is congruent to zero, so the probe cannot move
    the agent and reports ``agent_x``/``agent_y`` as dead. Measured: MultiRoom-N4-S5 (25x25) reports
    both dead, BossLevel (22x22) and every curated class report none. The dimensions are live — the
    probe is blind at that one period. Fixing it belongs in the suite (vary eps per dimension, or take
    it from the env) rather than in a wrap that is otherwise correct, so the curated set simply avoids
    grid widths dividing 25.
    """
    import gymnasium as gym
    import minigrid  # noqa: F401
    from minigrid.minigrid_env import MiniGridEnv as _MGE

    if ids is None:
        ids = sorted(k for k in gym.registry
                     if k.startswith("MiniGrid-") or k.startswith("BabyAI-"))
    added = []
    for eid in ids:
        cname = "MG_" + eid.replace("-", "_").replace(".", "_")
        try:
            e = gym.make(eid)
            e.reset(seed=0)
            u = e.unwrapped
            if not isinstance(u, _MGE):
                continue
            w, h, stw = u.width, u.height, bool(u.see_through_walls)
            lava = bool((u.grid.encode()[:, :, 0] == tm.T_LAVA).any())
            e.close()
        except Exception:                                        # noqa: BLE001  e.g. missing WFC assets
            continue
        topdown(cname, eid, w, h, stw, lava=lava, tile=tile)
        added.append(cname)
    return added
