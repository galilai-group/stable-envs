"""Overcooked — JaxMARL's renderer, which is a per-cell gather, and NOT the ``overcooked_ai_py`` one.

.. warning::

   **THIS FAMILY NEEDS ITS OWN INTERPRETER, and it is the only one that does.** Every other env in
   this library — all 191 of them — installs into a single environment on current torch/jax. These
   five cannot join it, and the reason is a hard resolver conflict rather than a missing package:

       jaxmarl 0.1.0 requires ``scipy<=1.12``   |   jax 0.10.2 requires ``scipy>=1.14``

   Nothing satisfies both. Installing jaxmarl with ``--no-deps`` to step over the pin does not end
   there: it walks into ``brax``, then ``jaxopt``, then the rest of the 2023 JAX ecosystem, and lands
   in an interpreter where nothing has verified that Overcooked still renders correctly on modern jax.
   The atlas here IS jaxmarl's own ``_render_tile`` output and there is no degraded mode, so a
   silently-wrong tile would be a silently-wrong frame.

   Install these five on their own, into a CLEAN venv (``python -m venv --system-site-packages``
   breaks jax's CUDA plugin discovery)::

       python -m venv ~/venvs/se-oc
       ~/venvs/se-oc/bin/pip install -e ".[overcooked]"

   Measured against jaxmarl 0.1.0 / jax 0.4.38 in ``~/venvs/se-oc``. The parity numbers below were
   taken there and are unaffected by this note — the env is fully supported, it simply does not share
   an interpreter.

   **Consequently this family is excluded from the release gate**, declared in
   ``stable_envs/testing/gate.py::EXCLUDED`` with the same reason, so its absence from a normal run is
   a stated carve-out rather than one more skip nobody reads. The permanent fix is to embed the atlas
   the way ``craftax``/``crafter``/``minihack`` embed theirs, which would make jaxmarl build-time-only
   and fold these five back in; ``_build_atlas`` below is what would be baked.

Reference: ``jaxmarl`` 0.1.0, ``jaxmarl.viz.overcooked_visualizer.OvercookedVisualizer._render_grid``,
driven the way ``OvercookedVisualizer.animate`` drives it — the whole ``maze_map`` window with
``highlight_mask=None``. ``reference_id`` is ``jaxmarl/Overcooked-V1`` and that is a load-bearing
declaration rather than a formality:

    THIS IS A DIFFERENT PICTURE FROM ``overcooked_ai_py``. Same board, different render. JaxMARL draws
    ``cramped_room`` at 450x525 with ``tile_size=75`` where ``StateVisualizer`` draws 300x375 (JaxMARL
    renders one extra ring of wall cells), it uses 48 distinct colours against 33, and it has ZERO
    sprite assets — every tile is MiniGrid's ``point_in_rect`` / ``point_in_circle`` /
    ``point_in_triangle`` primitive set at 3x supersampling, where ``overcooked_ai_py`` blits 93 named
    15x15 sprite frames. Nothing here reproduces the original Overcooked-AI picture, no pixel
    comparison between the two is meaningful, and this module does not claim otherwise.

Why the JAX one is the one that could be built at all is a single measured fact. ``overcooked_ai_py``
draws the pot's cooking timer as **font-rendered Roboto text at a fixed 20 px, applied AFTER the x5
tile scale**, so its pot tiles are ``tile_size``-specific and cannot be composed from the 15x15 source
art (see ``docs/research/overcooked.md``). JaxMARL draws the same information as a bar in NORMALISED
tile coordinates — ``point_in_rect(0.1, 0.9-(0.9-0.1)/20*pot_status, 0.83, 0.88)`` filled green — so
the tile composes at any size. Rendered at six tile sizes, **24/24 pot statuses stay distinct with zero
duplicates at T in {8, 12, 20, 32, 64, 75}**, and a token scan of all four jaxmarl viz modules for
``freetype|font|SysFont|Roboto|pygame|ImageFont|putText|cv2`` comes back empty. MEAL (arXiv 2506.14990)
does NOT restore the option: it replaces JaxMARL's renderer with a *pygame* reimplementation of the
``overcooked_ai_py`` look, Roboto digit timer included.

The frame is then a pure per-cell gather, which is the same shape as MiniGrid and is why
``tilemap.blit`` runs unmodified. Measured through these classes on one H200: **interior 100 %, all
pixels 100 %, max abs diff 0** against ``_render_grid`` on all five layouts, at **3.56-7.90 M frames/s
at batch 8,192** against 1,078-1,888 f/s for jaxmarl's own renderer with a warm tile cache — i.e.
**(x3,303-x5,713 at batch 8,192), ahead from batch 1** (7,192-7,496 f/s), replacing **1,351-1,997 CPU
cores** of JPEG decode of the same frames.

What the state IS
-----------------

One index per cell of the rendered grid, into a **64-entry alphabet** that is enumerated from jaxmarl's
own constants rather than harvested:

    1  empty  +  7 objects (wall, onion, onion_pile, plate, plate_pile, goal, dish)
    24 pot statuses  +  32 agents (2 hat colours x 4 facings x 4 held items)

The grid is ``maze_map[3:-3, 3:-3]`` — the layout plus the one-cell wall ring jaxmarl adds — so a
``cramped_room`` state is 42 numbers against a 192x224x3 = 129,024-byte frame at the default tile size.

Two facts make the per-cell key exactly right, and both are properties of the reference rather than
simplifications:

  * **the held item is a property of the CELL, not of the frame.** ``_render_tile`` resolves the
    agent's inventory as ``agent_inv[COLOR_TO_AGENT_INDEX[obj[1]]]`` — it looks the item up by the hat
    colour standing in that cell — so the tile is a function of (colour, facing, held item) alone;
  * **``agent_dir_idx`` is not read for a two-agent layout.** ``_render_tile`` overwrites the facing
    with ``agent_dir_idx[0]`` only ``if len(agent_dir_idx) == 1``, which is the single-agent minimax
    path. Every shipped Overcooked layout has two agents, so the facing that reaches the frame is the
    one in ``maze_map``, which is per cell.

No ``step``. The transition is discrete, but jaxmarl's own ``Overcooked.step`` is already a jitted,
``vmap``-able JAX function on the same GPU — there is no CPU simulator to replace, and a torch port
would be a second implementation of something that is not the bottleneck. What jaxmarl does NOT have on
the GPU is the renderer: ``_render_grid`` is a Python double loop over cells that reaches 1,078-1,888
f/s with a warm tile cache. That is the gap this module closes, and it is the only claim it makes.

Variations recolour through a label map, exactly as ``tilemap.py`` does for MiniGrid: alongside the RGB
atlas we record, at jaxmarl's own 3x supersampled resolution, which of ten semantic slots painted each
subpixel. That is not a convenience — the grid lines are literally ``(100, 100, 100)`` and so is
``COLORS['grey']``, so no colour-matching recolour can separate the wall from the line it is drawn
over, and ``downsample`` averages 3x3 subpixels so a recolour applied to the FINISHED tile is not the
recolour of the tile. The build asserts that ``palette[label]``, downsampled, reproduces jaxmarl's own
atlas byte for byte at the default palette, so a wrong constant cannot survive it.

``purple`` is deliberately absent from the palette knobs: it is the one entry of jaxmarl's ``COLORS``
that no Overcooked tile paints, and a knob that cannot move the picture is what
``check_variations_do_something`` exists to reject.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from ..env import Env
from ..registry import register
from ..render import tilemap as tm
from ..spaces import Dict, RGBBox

#: ``OvercookedVisualizer.TILE_PIXELS``; also the size the atlas is measured at
TILE_PIXELS = 32
#: ``_render_tile``'s supersampling factor
SUBDIVS = 3
#: ``animate``'s crop: ``agent_view_size - 2`` with jaxmarl's fixed ``agent_view_size = 5``
PAD = 3
#: ``pot_status`` values: 23 is an empty pot, 20..22 are filling, 19..1 are cooking, 0 is done
N_POT_STATUS = 24

#: label ids for the recolouring decomposition: two chrome slots, then jaxmarl's ``COLORS`` in
#: ``COLOR_TO_INDEX`` order
LBL_BG, LBL_GRID, LBL_COLOR = 0, 1, 2
N_COLOR = 9
N_LABEL = LBL_COLOR + N_COLOR
#: ``COLOR_TO_INDEX`` order, restated so ``state_names``-like ordering exists without importing jaxmarl
COLOR_NAMES = ("red", "green", "blue", "purple", "yellow", "grey", "white", "black", "orange")
#: the one ``COLORS`` entry no Overcooked tile paints; asserted, not assumed (see ``_build_atlas``)
UNPAINTED = ("purple",)
DEFAULT_GRIDLINE = (100, 100, 100)
DEFAULT_BG = (0, 0, 0)


def _require_jaxmarl():
    """Probe the PATH THAT WOULD FAIL, not just the top-level import.

    ``import jaxmarl`` is not the thing this module needs. It needs (a) the visualiser module, which
    imports ``jaxmarl.viz.window`` and that module calls ``sys.exit(-1)`` when matplotlib is missing —
    an ImportError is not even raised, the interpreter is killed; (b) ``_render_tile`` actually
    producing a tile, which is what the atlas is built from and has no fallback; and (c) the layout
    table. Each fails differently and none of them is checked by importing the package.

    Raising ``ImportError`` is what makes the env cleanly ABSENT rather than present and broken:
    ``stable_envs/envs/__init__.py`` records it and ``import_errors()`` names it.
    """
    try:
        import jaxmarl  # noqa: F401
    except Exception as e:                                              # noqa: BLE001
        raise ImportError(
            f"jaxmarl is not importable ({type(e).__name__}: {e}). The atlas IS jaxmarl's own "
            f"`_render_tile` output — nothing here re-derives it — so there is no degraded mode. "
            f"`pip install jaxmarl` (measured against jaxmarl 0.1.0 / jax 0.4.38) INTO A CLEAN VENV: "
            f"a `python -m venv --system-site-packages` venv breaks jax's CUDA plugin discovery.") from e
    try:
        import matplotlib  # noqa: F401
    except Exception as e:                                              # noqa: BLE001
        raise ImportError(
            f"matplotlib is missing ({type(e).__name__}: {e}), and jaxmarl.viz.window — which "
            f"overcooked_visualizer imports at module scope — responds to that by calling "
            f"sys.exit(-1) rather than by raising. Importing the visualiser would kill the "
            f"interpreter, so it is checked here instead.") from e
    try:
        from jaxmarl.environments.overcooked.common import OBJECT_TO_INDEX
        from jaxmarl.environments.overcooked.layouts import overcooked_layouts
        from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
        tile = OvercookedVisualizer._render_tile(
            np.array([OBJECT_TO_INDEX["pot"], 7, 5], np.uint8), tile_size=8)
    except SystemExit as e:                                             # noqa: BLE001
        raise ImportError(f"importing jaxmarl's Overcooked visualiser called sys.exit({e.code})") from e
    except Exception as e:                                              # noqa: BLE001
        raise ImportError(
            f"jaxmarl is installed but its Overcooked visualiser does not render: "
            f"{type(e).__name__}: {e}") from e
    if np.asarray(tile).shape != (8, 8, 3):
        raise ImportError(f"jaxmarl's _render_tile returned {np.asarray(tile).shape}, not (8, 8, 3); "
                          f"the atlas would be the wrong shape")
    if not overcooked_layouts:
        raise ImportError("jaxmarl ships no Overcooked layouts")


_require_jaxmarl()


def _constants():
    from jaxmarl.environments.overcooked.common import (COLOR_TO_INDEX, COLORS, OBJECT_TO_INDEX,
                                                        OBJECT_INDEX_TO_VEC)
    if tuple(sorted(COLOR_TO_INDEX, key=COLOR_TO_INDEX.get)) != COLOR_NAMES:
        raise ImportError(
            f"jaxmarl's COLOR_TO_INDEX is {sorted(COLOR_TO_INDEX, key=COLOR_TO_INDEX.get)}, not "
            f"{list(COLOR_NAMES)} — the palette knobs are named and ordered by that table, so a "
            f"changed one would silently rename every colour")
    return COLOR_TO_INDEX, COLORS, OBJECT_TO_INDEX, np.asarray(OBJECT_INDEX_TO_VEC, np.uint8)


def official_colors() -> np.ndarray:
    """``COLORS`` as a (9,3) uint8 array in ``COLOR_TO_INDEX`` order, read from the package."""
    C2I, COLORS, _, _ = _constants()
    out = np.zeros((N_COLOR, 3), np.uint8)
    for name, i in C2I.items():
        out[i] = COLORS[name]
    return out


def cell_alphabet() -> list[tuple[int, int, int, int]]:
    """Every ``(obj_type, colour, extra, held)`` a rendered Overcooked cell can hold. 64 of them.

    Enumerated from jaxmarl's own ``OBJECT_INDEX_TO_VEC`` rather than from what a rollout happens to
    reach: a random policy almost never fills a pot, so 20 of the 24 pot tiles would be missing and the
    atlas would be a lower bound. ``unseen`` is excluded because ``_render_grid`` folds it into
    ``empty``; ``empty`` is the ``obj is None`` tile.
    """
    C2I, _, O, vec = _constants()
    held = (O["empty"], O["onion"], O["plate"], O["dish"])          # what ``_render_inv`` can draw
    out = [(O["empty"], 0, 0, O["empty"])]
    for name in ("wall", "onion", "onion_pile", "plate", "plate_pile", "goal", "dish"):
        out.append((O[name], int(vec[O[name]][1]), 0, O["empty"]))
    out += [(O["pot"], int(vec[O["pot"]][1]), s, O["empty"]) for s in range(N_POT_STATUS)]
    out += [(O["agent"], c, d, h) for c in (C2I["red"], C2I["blue"]) for d in range(4) for h in held]
    return out


ALPHABET = cell_alphabet()
N_CODE = len(ALPHABET)
CODE_OF = {c: i for i, c in enumerate(ALPHABET)}


def _label_of(color, painted: dict) -> int:
    C2I, COLORS, _, _ = _constants()
    v = np.asarray(color, np.float64).ravel()
    for name, i in C2I.items():
        if np.array_equal(v, np.asarray(COLORS[name], np.float64)):
            painted[name] = True
            return LBL_COLOR + i
    raise ValueError(f"jaxmarl painted {v}, which is not one of its own COLORS; the label "
                     f"decomposition is incomplete and every recoloured frame would be wrong")


def _build_atlas(tile_size: int):
    """``(64,T,T,3)`` uint8 official tiles and ``(64,3T,3T)`` uint8 label maps.

    The RGB tiles come back from ``_render_tile`` and are truncated to uint8 exactly the way
    ``_render_grid`` truncates when it assigns a float64 tile into its uint8 frame, so exactness is
    structural. The label maps are produced by the SAME call with ``grid_rendering.fill_coords``
    shimmed to write a label id instead of a colour — every pixel is still placed by jaxmarl's own
    predicates, so nothing about the geometry is restated here. The two grid-line rectangles are the
    first two fills ``_render_tile`` issues, unconditionally, which is how they are told apart from
    ``COLORS['grey']``: the two are the same RGB value and no colour match can separate them.
    """
    import jaxmarl.viz.grid_rendering as gr
    from jaxmarl.environments.overcooked.common import OBJECT_TO_INDEX
    from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer as V

    T, f = int(tile_size), SUBDIVS
    AGENT = OBJECT_TO_INDEX["agent"]

    def draw(code, out=None):
        t, c, e, h = code
        V.tile_cache = {}
        obj = None if t == OBJECT_TO_INDEX["empty"] else np.array([t, c, e], np.uint8)
        inv = np.array([h, h], np.int32) if t == AGENT else None
        return V._render_tile(obj, highlight=False, agent_dir_idx=np.array([0, 1]),
                              agent_inv=inv, tile_size=T, subdivs=f)

    rgb = np.stack([np.asarray(draw(c)).astype(np.uint8) for c in ALPHABET])

    labels = np.zeros((N_CODE, T * f, T * f), np.uint8)
    painted: dict = {}
    orig = gr.fill_coords
    for i, code in enumerate(ALPHABET):
        lab = np.zeros((T * f, T * f, 3), np.uint8)                 # LBL_BG == 0
        n = [0]

        def shim(img, fn, color, _lab=lab, _n=n, _o=orig):
            lbl = LBL_GRID if _n[0] < 2 else _label_of(color, painted)
            _n[0] += 1
            return _o(_lab, fn, (lbl,) * 3)

        gr.fill_coords = shim
        try:
            draw(code)
        finally:
            gr.fill_coords = orig
        if n[0] < 2:
            raise AssertionError(
                f"_render_tile issued {n[0]} fills for {code}; the first two are assumed to be the "
                f"grid lines and that is how they are separated from COLORS['grey']")
        labels[i] = lab[:, :, 0]
    V.tile_cache = {}

    missing = [c for c in COLOR_NAMES if c not in painted and c not in UNPAINTED]
    extra = [c for c in UNPAINTED if c in painted]
    if missing or extra:
        raise AssertionError(
            f"the colours Overcooked actually paints changed: {missing} are declared as knobs and "
            f"never reach a pixel, {extra} are declared unpainted and do. A knob that cannot move the "
            f"picture is what check_variations_do_something rejects")
    rec = _recolour_np(labels, _label_palette({"palette": official_colors(),
                                               "background": DEFAULT_BG,
                                               "grid_line": DEFAULT_GRIDLINE}), T)
    bad = int((rec != rgb).any(axis=(1, 2, 3)).sum())
    if bad:
        raise AssertionError(
            f"the label reconstruction differs from jaxmarl's own atlas in {bad}/{N_CODE} tiles at the "
            f"default palette — the decomposition is wrong, so every recoloured frame would be wrong "
            f"in a way no parity test would catch")
    return rgb, labels


def _label_palette(values: dict) -> np.ndarray:
    p = np.zeros((N_LABEL, 3), np.float64)
    p[LBL_BG] = np.asarray(values["background"], np.float64)
    p[LBL_GRID] = np.asarray(values["grid_line"], np.float64)
    p[LBL_COLOR:LBL_COLOR + N_COLOR] = np.asarray(values["palette"], np.float64).reshape(N_COLOR, 3)
    return np.trunc(p)


def _recolour_np(labels: np.ndarray, pal: np.ndarray, T: int) -> np.ndarray:
    """``downsample`` applied to ``palette[label]``, in jaxmarl's own axis order and dtype."""
    f = SUBDIVS
    x = pal[labels.astype(np.int64)]                                # (N, fT, fT, 3) float64
    x = x.reshape(labels.shape[0], T, f, T, f, 3).mean(axis=4).mean(axis=2)
    return x.astype(np.uint8)


def _cache_path(tile_size: int) -> str:
    root = os.environ.get("STABLE_ENVS_CACHE") or os.path.expanduser("~/.cache/stable-envs")
    return os.path.join(root, "tilemap", f"overcooked_jaxmarl_t{tile_size}.npz")


class OvercookedAtlas:
    """The 64-tile atlas for one tile size, plus its label decomposition. Cached in process and on disk.

    Building costs a few seconds of pure-Python scanline fills per tile size (``fill_coords`` is a
    double loop over ``3T x 3T`` subpixels), and the conformance suite builds several sizes, so the
    result is memoised. The disk write is atomic because a test session may build several envs at once.
    """

    _mem: dict[tuple[int, str], "OvercookedAtlas"] = {}

    def __init__(self, tile_size: int, device="cuda"):
        self.T = int(tile_size)
        self.device = torch.device(device)
        rgb, labels = self._load_or_build(self.T)
        self.official = torch.from_numpy(rgb).to(self.device)
        self.labels = torch.from_numpy(labels).to(self.device)
        self._rgb_cache: dict[bytes, torch.Tensor] = {}

    @classmethod
    def get(cls, tile_size: int, device="cuda") -> "OvercookedAtlas":
        key = (int(tile_size), str(device))
        if key not in cls._mem:
            cls._mem[key] = cls(tile_size, device)
        return cls._mem[key]

    @staticmethod
    def _load_or_build(T: int):
        path = _cache_path(T)
        if os.path.exists(path):
            try:
                z = np.load(path)
                if int(z["n_code"]) == N_CODE:
                    return z["atlas"], z["labels"]
            except Exception:                                       # noqa: BLE001  corrupt/partial
                pass
        rgb, labels = _build_atlas(T)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, atlas=rgb, labels=labels, n_code=N_CODE)
        os.replace(tmp, path)
        return rgb, labels

    def rgb(self, values: dict) -> torch.Tensor:
        key = b"|".join(np.ascontiguousarray(np.asarray(values[k], np.float64)).tobytes()
                        for k in sorted(values))
        hit = self._rgb_cache.get(key)
        if hit is not None:
            return hit
        pal = torch.as_tensor(_label_palette(values), device=self.device)
        out = self._recolour(pal)
        if len(self._rgb_cache) > 8:
            self._rgb_cache.clear()
        self._rgb_cache[key] = out
        return out

    def _recolour(self, pal: torch.Tensor) -> torch.Tensor:
        f, T = SUBDIVS, self.T
        x = pal[self.labels.long()]                                 # (N, fT, fT, 3) float64
        x = x.reshape(N_CODE, T, f, T, f, 3).mean(4).mean(2)
        return x.to(torch.uint8).contiguous()


class _OvercookedLayout(Env):
    """One JaxMARL Overcooked layout, rendered as a gather from the 64-tile atlas."""

    #: jaxmarl layout key; the class name is the registry key
    layout: str = ""
    reference_id = "jaxmarl/Overcooked-V1"
    #: PADDED grid — the layout plus jaxmarl's one-cell wall ring, which IS what gets drawn
    grid_w: int = 0
    grid_h: int = 0

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.atlas = OvercookedAtlas.get(self._tile(self.size), device=self.device)
        self._pool: np.ndarray | None = None
        self._pool_labels: np.ndarray | None = None
        self._default_t: torch.Tensor | None = None

    # geometry

    def _tile(self, size: int) -> int:
        t, r = divmod(int(size), self.grid_h)
        if r or t < 4:
            raise ValueError(
                f"{type(self).__name__}: size {size} is not a whole tile size for a {self.grid_w}x"
                f"{self.grid_h} grid; use a multiple of {self.grid_h} that is at least {4*self.grid_h}")
        return t

    # variation

    def make_variation_space(self) -> Dict:
        """One RGB knob per colour Overcooked actually paints, plus the two chrome slots.

        Per-colour leaves rather than one packed palette, and that is a claim about this renderer:
        every subpixel has exactly one writer and ``downsample`` averages 3x3 of them, so a region
        whose subpixels share a label lands on the requested colour EXACTLY — which is what
        ``check_colors_apply_exactly`` measures and what a blended renderer (MiniGrid's POV view) could
        not offer.

        ``purple`` is absent because no Overcooked tile paints it; ``_build_atlas`` asserts that,
        against jaxmarl, every build.
        """
        pal = official_colors()
        C2I, _, _, _ = _constants()
        space = {"background": RGBBox(init_value=DEFAULT_BG),
                 "grid_line": RGBBox(init_value=DEFAULT_GRIDLINE)}
        for name in COLOR_NAMES:
            if name not in UNPAINTED:
                space[name] = RGBBox(init_value=tuple(int(v) for v in pal[C2I[name]]))
        return Dict(space)

    def _values(self, row: np.ndarray) -> dict:
        d = self.variation_space.unpack(row)
        C2I, _, _, _ = _constants()
        pal = official_colors().astype(np.float64)
        for name in COLOR_NAMES:
            if name not in UNPAINTED:
                pal[C2I[name]] = np.trunc(np.asarray(d[name], np.float64))
        return {"palette": pal, "background": np.trunc(np.asarray(d["background"], np.float64)),
                "grid_line": np.trunc(np.asarray(d["grid_line"], np.float64))}

    def _label_palettes(self, sty) -> torch.Tensor:
        """``(B, N_LABEL, 3)`` float64 — the colour each label paints, per environment."""
        B, dev = sty.batch, self.device
        C2I, _, _, _ = _constants()
        p = torch.zeros(B, N_LABEL, 3, dtype=torch.float64, device=dev)
        p[:, LBL_BG] = torch.trunc(sty["background"].double())
        p[:, LBL_GRID] = torch.trunc(sty["grid_line"].double())
        base = torch.as_tensor(official_colors(), dtype=torch.float64, device=dev)
        p[:, LBL_COLOR:LBL_COLOR + N_COLOR] = base.unsqueeze(0)
        for name in COLOR_NAMES:
            if name not in UNPAINTED:
                p[:, LBL_COLOR + C2I[name]] = torch.trunc(sty[name].double())
        return p

    def _is_default_style(self, sty) -> bool:
        if self._default_t is None:
            self._default_t = torch.as_tensor(self.default_style(), device=self.device,
                                              dtype=torch.float32)
        return bool((sty.values[0] == self._default_t).all())

    # rendering

    @torch.no_grad()
    def render(self, states=None, size: int | None = None, supersample: int | None = None,
               variation: dict | None = None, style=None, out=None) -> torch.Tensor:
        """``(...,state_dim)`` -> ``(...,grid_h*T,grid_w*T,3)`` uint8 on device."""
        if supersample not in (None, 1):
            raise ValueError(
                f"{type(self).__name__}: supersample={supersample} is not available. jaxmarl's own 3x "
                f"supersampling is already baked into the atlas by _render_tile; sampling again on top "
                f"of it would render something the reference never draws.")
        s = self._state if states is None else torch.as_tensor(states, device=self.device,
                                                               dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        tile = self._tile(int(size or self.size))
        atlas = self.atlas if tile == self.atlas.T else OvercookedAtlas.get(tile, device=self.device)
        sty = self.resolve_style(style, variation, batch=int(s.shape[0]))
        idx = torch.round(s).long().remainder(N_CODE).reshape(
            -1, self.grid_h, self.grid_w).to(torch.int32)
        flat = None
        if out is not None:
            flat = out.reshape(-1, self.grid_h * tile, self.grid_w * tile, 3)
        if sty.shared:
            rgb = (atlas.official if self._is_default_style(sty)
                   else atlas.rgb(self._values(sty.values[0].detach().cpu().numpy())))
            img = tm.blit(rgb, idx, tile, out=flat)
        else:
            img = self._blit_styled(atlas, idx, sty, tile, flat)
        return img.reshape(*lead, *img.shape[1:]) if lead else img

    def _blit_styled(self, atlas, idx: torch.Tensor, sty, tile: int, out) -> torch.Tensor:
        """Per-environment palettes: gather LABELS, colour them per environment, then downsample.

        Arithmetically identical to recolouring the atlas and gathering — every subpixel has exactly
        one label, the lookup is per subpixel either way, and the box filter is the same two-stage
        ``mean`` in the same order and dtype — but it does not need B atlases for B appearances.
        """
        B, H, W = idx.shape
        dev, f, px = self.device, SUBDIVS, SUBDIVS * tile
        pal = self._label_palettes(sty)
        if out is None:
            out = torch.empty(B, H * tile, W * tile, 3, dtype=torch.uint8, device=dev)
        step = max(1, int((256 << 20) // (px * W * px * 3 * 8)))
        for b0 in range(0, B, step):
            b1 = min(B, b0 + step)
            bi = torch.arange(b0, b1, device=dev).view(-1, 1, 1)
            for j in range(H):
                lab = atlas.labels[idx[b0:b1, j].long()].long()      # (b, W, px, px)
                lab = lab.permute(0, 2, 1, 3).reshape(b1 - b0, px, W * px)
                col = pal[bi, lab]                                  # (b, px, W*px, 3) f64
                col = col.reshape(b1 - b0, tile, f, W * tile, f, 3).mean(4).mean(2)
                out[b0:b1, j * tile:(j + 1) * tile] = col.to(torch.uint8)
        return out

    # states

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Rollout states, DIALLED so that every tile the atlas holds actually occurs, and stratified.

        Uniform random joint actions never cook. Measured on ``cramped_room``, 2,000 random states
        reach **4 of the 24 pot codes and no dish at all**, so 20 of the atlas's 64 tiles — including
        every cooking-timer tile, which is the entire reason this family is buildable — would never
        appear in a parity comparison, and the ``orange`` knob would be reported inert because no state
        in the batch paints it.

        The fix is not a synthetic prior. Each dialled row is a real row with the substitutions
        ``make_overcooked_map`` ITSELF performs: a pot cell moved to another ``pot_status``, a counter
        cell holding a placed ``onion``/``plate``/``dish``, and an agent cell holding another item.
        Those are the map writes the env makes when a chef cooks, puts something down, or picks
        something up. The batch is then stratified over the atlas's LABELS — the semantic colour slots
        — so every declared colour knob provably has something to paint, which is what
        ``check_colors_apply_exactly`` measures and what a lucky draw would otherwise decide.
        """
        rng = rng or self.rng
        if self._pool is None:
            self._pool, self._pool_labels = self._build_pool(max(256, batch))
        pool, has = self._pool, self._pool_labels
        idx = rng.choice(len(pool), batch, replace=batch > len(pool))
        if batch >= N_LABEL:
            for _ in range(N_LABEL + 1):
                missing = [l for l in range(N_LABEL) if has[:, l].any() and not has[idx, l].any()]
                if not missing:
                    break
                for l in missing:
                    cand = np.flatnonzero(has[:, l])
                    idx[rng.integers(0, batch)] = cand[rng.integers(0, len(cand))]
        return pool[idx].astype(np.float32)

    def _code_labels(self) -> np.ndarray:
        """``(N_CODE, N_LABEL)`` bool: which semantic slots each tile of the atlas paints."""
        lab = self.atlas.labels.reshape(N_CODE, -1)
        out = np.zeros((N_CODE, N_LABEL), bool)
        for l in range(N_LABEL):
            out[:, l] = (lab == l).any(1).cpu().numpy()
        return out

    def _build_pool(self, n: int):
        import jax
        import jaxmarl
        from jaxmarl.environments.overcooked.layouts import overcooked_layouts

        _, _, O, vec = _constants()
        env = jaxmarl.make("overcooked", layout=overcooked_layouts[self.layout])
        key = jax.random.PRNGKey(0)
        base = []
        while len(base) < max(64, n // 3):
            key, k = jax.random.split(key)
            _obs, st = env.reset(k)
            for _ in range(48):
                base.append(self.encode_state(st))
                if len(base) >= max(64, n // 3):
                    break
                key, k = jax.random.split(key)
                acts = {a: jax.random.randint(jax.random.fold_in(k, i), (), 0, 6)
                        for i, a in enumerate(env.agents)}
                key, ks = jax.random.split(key)
                _obs, st, _r, _d, _i = env.step(ks, st, acts)

        pot = [CODE_OF[(O["pot"], int(vec[O["pot"]][1]), s, O["empty"])]
               for s in range(N_POT_STATUS)]
        placed = [CODE_OF[(O[t], int(vec[O[t]][1]), 0, O["empty"])]
                  for t in ("onion", "plate", "dish")]
        wall = CODE_OF[(O["wall"], int(vec[O["wall"]][1]), 0, O["empty"])]
        held = (O["empty"], O["onion"], O["plate"], O["dish"])
        agent_of = {c: i for i, c in enumerate(ALPHABET) if c[0] == O["agent"]}

        # the dial. Deterministic and coprime-ish in its periods (24 statuses, 3 placeable items, 4
        # held items) so a short run of rows covers every one of them rather than a lucky subset.
        out = list(base)
        for j in range(max(n - len(base), 3 * N_POT_STATUS)):
            row = base[j % len(base)].copy()
            row[np.isin(row, pot)] = pot[j % N_POT_STATUS]
            walls = np.flatnonzero(row == wall)
            if len(walls):
                row[walls[j % len(walls)]] = placed[j % len(placed)]
            for i, c in enumerate(row.astype(np.int64)):
                code = ALPHABET[int(c)]
                if code[0] == O["agent"]:
                    row[i] = agent_of[(code[0], code[1], code[2], held[j % len(held)])]
            out.append(row)
        pool = np.stack(out[:max(n, len(base) + 3 * N_POT_STATUS)]).astype(np.float32)
        codes = np.round(pool).astype(np.int64) % N_CODE
        cl = self._code_labels()
        has = np.stack([cl[codes[i]].any(0) for i in range(len(codes))])
        return pool, has

    @classmethod
    def encode_state(cls, st) -> np.ndarray:
        """A jaxmarl ``State`` -> this class's compact state, through the bytes the renderer reads."""
        _, _, O, _ = _constants()
        grid = np.asarray(st.maze_map[PAD:-PAD, PAD:-PAD, :], np.int64)
        inv = np.asarray(st.agent_inv, np.int64).reshape(-1)
        return cls.encode_grid(grid, inv)

    @classmethod
    def encode_grid(cls, grid: np.ndarray, agent_inv: np.ndarray) -> np.ndarray:
        from jaxmarl.viz.overcooked_visualizer import COLOR_TO_AGENT_INDEX
        _, _, O, _ = _constants()
        H, W = grid.shape[:2]
        out = np.zeros(H * W, np.float32)
        for y in range(H):
            for x in range(W):
                t, c, e = (int(v) for v in grid[y, x])
                if t in (O["unseen"], O["empty"]) and e == 0:
                    key = (O["empty"], 0, 0, O["empty"])
                elif t == O["agent"]:
                    key = (t, c, e, int(agent_inv[COLOR_TO_AGENT_INDEX[c]]))
                else:
                    key = (t, c, e, O["empty"])
                if key not in CODE_OF:
                    raise ValueError(
                        f"{cls.__name__}: jaxmarl produced the cell {key}, which is not in the "
                        f"64-entry alphabet enumerated from its own constants")
                out[y * W + x] = CODE_OF[key]
        return out

    # reference bridge

    @classmethod
    def reference_env(cls, size: int | None = None):
        """jaxmarl's visualiser plus the tile size to drive it at.

        Returned as a tuple rather than a gym env because ``_render_grid`` is a classmethod on the
        visualiser and needs no instance — and constructing an ``OvercookedVisualizer`` would build a
        matplotlib ``Window``, which is the interactive path and has nothing to do with the frame.
        """
        from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
        return (OvercookedVisualizer, int(size or cls.default_size) // cls.grid_h)

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Official frames through the real ``_render_grid``, with the state round-trip asserted.

        ``agent_dir_idx`` is passed with length 2 deliberately: at length 1 ``_render_tile`` overwrites
        every agent's facing with ``agent_dir_idx[0]``, which is jaxmarl's single-agent minimax path
        and not what a two-agent layout draws.
        """
        from jaxmarl.environments.overcooked.common import OBJECT_TO_INDEX
        viz, tile = env
        _, _, O, _ = _constants()
        H, W = cls.grid_h, cls.grid_w
        out = []
        for st in np.asarray(states):
            codes = np.round(np.asarray(st, np.float64)).astype(np.int64) % N_CODE
            grid = np.zeros((H, W, 3), np.uint8)
            inv = np.array([O["empty"], O["empty"]], np.int64)
            seen: dict[int, int] = {}
            for k, code in enumerate(codes):
                t, c, e, h = ALPHABET[int(code)]
                grid[k // W, k % W] = (t, c, e)
                if t == OBJECT_TO_INDEX["agent"]:
                    slot = 0 if c == 0 else 1
                    if slot in seen and seen[slot] != h:
                        raise AssertionError(
                            f"{cls.__name__}: this state puts two agents of the same colour on the "
                            f"board holding different items. jaxmarl's renderer looks the held item up "
                            f"by hat colour in a two-entry array, so it cannot draw that state and a "
                            f"parity number taken against it would compare two different scenes")
                    seen[slot] = h
                    inv[slot] = h
            fr = viz._render_grid(grid, tile_size=tile, highlight_mask=None,
                                  agent_dir_idx=np.array([0, 1]), agent_inv=inv)
            back = cls.encode_grid(grid.astype(np.int64), inv)
            if not np.array_equal(np.round(back).astype(np.int64), codes):
                raise AssertionError(
                    f"{cls.__name__}: the state written into the reference does not read back as the "
                    f"state requested; parity measured against this frame would compare two scenes")
            out.append(np.asarray(fr, np.uint8))
        return np.stack(out)


def _cell_names(h: int, w: int) -> tuple[str, ...]:
    return tuple(f"cell[{r},{c}]" for r in range(h) for c in range(w))


def layout(class_name: str, key: str, doc: str = "", tile: int = TILE_PIXELS) -> type:
    """Build and register a class for one jaxmarl layout.

    A class per layout rather than one parameterised env: the registry is the conformance suite's
    parameter list, so a layout reachable only through a constructor argument is a layout nothing
    tests. The padded grid size is read from jaxmarl's own layout table at import, so it cannot drift
    from the reference the way a hardcoded one would.
    """
    from jaxmarl.environments.overcooked.layouts import overcooked_layouts
    if key not in overcooked_layouts:
        raise ImportError(f"jaxmarl has no Overcooked layout {key!r}; it ships "
                          f"{sorted(overcooked_layouts)}")
    lay = overcooked_layouts[key]
    w, h = int(lay["width"]) + 2, int(lay["height"]) + 2        # + jaxmarl's one-cell wall ring
    cls = type(class_name, (_OvercookedLayout,), {
        "__doc__": doc or f"jaxmarl Overcooked, layout {key}: {w}x{h} padded grid.",
        "__module__": __name__,
        "name": "overcooked-" + key.replace("_", "-"),
        "layout": key,
        "grid_w": w, "grid_h": h,
        "state_dim": w * h,
        "state_names": _cell_names(h, w),
        "default_size": h * tile,
    })
    return register(cls)


OvercookedCrampedRoom = layout(
    "OvercookedCrampedRoom", "cramped_room",
    doc="Overcooked cramped_room: two chefs, one pot, one counter run — the layout every Overcooked "
        "paper reports first. 42 numbers against a 192x224 frame.")
OvercookedAsymmAdvantages = layout(
    "OvercookedAsymmAdvantages", "asymm_advantages",
    doc="Overcooked asymm_advantages: the two chefs have different access to the pots, so the "
        "layout is not left-right symmetric and neither is any state in it.")
OvercookedCoordRing = layout(
    "OvercookedCoordRing", "coord_ring",
    doc="Overcooked coord_ring: a ring corridor, the layout where the two chefs must pass each other.")
OvercookedForcedCoord = layout(
    "OvercookedForcedCoord", "forced_coord",
    doc="Overcooked forced_coord: a dividing wall neither chef can cross, so every soup needs a "
        "hand-off across the counter.")
OvercookedCounterCircuit = layout(
    "OvercookedCounterCircuit", "counter_circuit",
    doc="Overcooked counter_circuit: the widest shipped layout, a 9x7 padded grid with a long "
        "counter loop.")
