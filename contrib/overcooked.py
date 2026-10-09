from __future__ import annotations

import os

import numpy as np
import torch

from contrib.render import tilemap as tm
from contrib.render._cache import cache_path
from stable_envs.env import Env
from stable_envs.registry import register
from stable_envs.render import _resources as _RES
from stable_envs.spaces import Dict, RGBBox

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
    """Probe visualizer imports, tile rendering and layouts before registration.

    Check matplotlib first: JaxMARL’s window module can call sys.exit when it is
    unavailable. Convert that failure to ImportError for registry discovery."""
    try:
        import jaxmarl  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            f"jaxmarl is not importable ({type(e).__name__}: {e}). The atlas IS jaxmarl's own "
            f"`_render_tile` output — nothing here re-derives it — so there is no degraded mode. "
            f"`pip install jaxmarl` (measured against jaxmarl 0.1.0 / jax 0.4.38) INTO A CLEAN VENV: "
            f"a `python -m venv --system-site-packages` venv breaks jax's CUDA plugin discovery."
        ) from e
    try:
        import matplotlib  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            f"matplotlib is missing ({type(e).__name__}: {e}), and jaxmarl.viz.window — which "
            f"overcooked_visualizer imports at module scope — responds to that by calling "
            f"sys.exit(-1) rather than by raising. Importing the visualiser would kill the "
            f"interpreter, so it is checked here instead."
        ) from e
    try:
        from jaxmarl.environments.overcooked.common import OBJECT_TO_INDEX
        from jaxmarl.environments.overcooked.layouts import overcooked_layouts
        from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer

        tile = OvercookedVisualizer._render_tile(np.array([OBJECT_TO_INDEX["pot"], 7, 5], np.uint8), tile_size=8)
    except SystemExit as e:  # noqa: BLE001
        raise ImportError(f"importing jaxmarl's Overcooked visualiser called sys.exit({e.code})") from e
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            f"jaxmarl is installed but its Overcooked visualiser does not render: {type(e).__name__}: {e}"
        ) from e
    if np.asarray(tile).shape != (8, 8, 3):
        raise ImportError(
            f"jaxmarl's _render_tile returned {np.asarray(tile).shape}, not (8, 8, 3); "
            f"the atlas would be the wrong shape"
        )
    if not overcooked_layouts:
        raise ImportError("jaxmarl ships no Overcooked layouts")


_require_jaxmarl()


def _constants():
    from jaxmarl.environments.overcooked.common import COLOR_TO_INDEX, COLORS, OBJECT_INDEX_TO_VEC, OBJECT_TO_INDEX

    if tuple(sorted(COLOR_TO_INDEX, key=COLOR_TO_INDEX.get)) != COLOR_NAMES:
        raise ImportError(
            f"jaxmarl's COLOR_TO_INDEX is {sorted(COLOR_TO_INDEX, key=COLOR_TO_INDEX.get)}, not "
            f"{list(COLOR_NAMES)} — the palette knobs are named and ordered by that table, so a "
            f"changed one would silently rename every colour"
        )
    return COLOR_TO_INDEX, COLORS, OBJECT_TO_INDEX, np.asarray(OBJECT_INDEX_TO_VEC, np.uint8)


def official_colors() -> np.ndarray:
    """Read source colours as ``(9, 3)`` uint8 in COLOR_TO_INDEX order."""
    C2I, COLORS, _, _ = _constants()
    out = np.zeros((N_COLOR, 3), np.uint8)
    for name, i in C2I.items():
        out[i] = COLORS[name]
    return out


def cell_alphabet() -> list[tuple[int, int, int, int]]:
    """Enumerate all 64 renderable object/colour/status/inventory combinations.

    Include every pot status and held item even if random play rarely reaches it.
    The source renderer maps unseen cells to empty."""
    C2I, _, O, vec = _constants()
    held = (O["empty"], O["onion"], O["plate"], O["dish"])  # what ``_render_inv`` can draw
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
    raise ValueError(
        f"jaxmarl painted {v}, which is not one of its own COLORS; the label "
        f"decomposition is incomplete and every recoloured frame would be wrong"
    )


def _build_atlas(tile_size: int):
    """Build source RGB tiles and labels at the source’s 3× supersampling resolution.

    Label the first two fills as grid lines to distinguish them from identically
    coloured walls. Verify that palette lookup and downsampling reproduce RGB.
    Construction temporarily patches source fill_coords and must be serialized."""
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
        return V._render_tile(
            obj, highlight=False, agent_dir_idx=np.array([0, 1]), agent_inv=inv, tile_size=T, subdivs=f
        )

    rgb = np.stack([np.asarray(draw(c)).astype(np.uint8) for c in ALPHABET])

    labels = np.zeros((N_CODE, T * f, T * f), np.uint8)
    painted: dict = {}
    orig = gr.fill_coords
    for i, code in enumerate(ALPHABET):
        lab = np.zeros((T * f, T * f, 3), np.uint8)  # LBL_BG == 0
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
                f"grid lines and that is how they are separated from COLORS['grey']"
            )
        labels[i] = lab[:, :, 0]
    V.tile_cache = {}

    missing = [c for c in COLOR_NAMES if c not in painted and c not in UNPAINTED]
    extra = [c for c in UNPAINTED if c in painted]
    if missing or extra:
        raise AssertionError(
            f"the colours Overcooked actually paints changed: {missing} are declared as knobs and "
            f"never reach a pixel, {extra} are declared unpainted and do. A knob that cannot move the "
            f"picture is what check_variations_do_something rejects"
        )
    rec = _recolour_np(
        labels,
        _label_palette({"palette": official_colors(), "background": DEFAULT_BG, "grid_line": DEFAULT_GRIDLINE}),
        T,
    )
    bad = int((rec != rgb).any(axis=(1, 2, 3)).sum())
    if bad:
        raise AssertionError(
            f"the label reconstruction differs from jaxmarl's own atlas in {bad}/{N_CODE} tiles at the "
            f"default palette — the decomposition is wrong, so every recoloured frame would be wrong "
            f"in a way no parity test would catch"
        )
    return rgb, labels


def _label_palette(values: dict) -> np.ndarray:
    p = np.zeros((N_LABEL, 3), np.float64)
    p[LBL_BG] = np.asarray(values["background"], np.float64)
    p[LBL_GRID] = np.asarray(values["grid_line"], np.float64)
    p[LBL_COLOR : LBL_COLOR + N_COLOR] = np.asarray(values["palette"], np.float64).reshape(N_COLOR, 3)
    return np.trunc(p)


def _recolour_np(labels: np.ndarray, pal: np.ndarray, T: int) -> np.ndarray:
    """Downsample palette labels with source axis order, float64 means and uint8 truncation."""
    f = SUBDIVS
    x = pal[labels.astype(np.int64)]  # (N, fT, fT, 3) float64
    x = x.reshape(labels.shape[0], T, f, T, f, 3).mean(axis=4).mean(axis=2)
    return x.astype(np.uint8)


def _cache_path(tile_size: int) -> str:
    return str(cache_path("tilemap", f"overcooked_jaxmarl_t{tile_size}.npz"))


class OvercookedAtlas:
    """Cache source RGB and label atlases per tile size, with device-specific tensors.

    Disk writes replace the final file atomically. In-process construction is not
    synchronized; the source renderer’s global patch requires exclusive access."""

    _mem: dict[tuple[int, str], OvercookedAtlas] = {}

    def __init__(self, tile_size: int, device="cuda"):
        self.T = int(tile_size)
        self.device = torch.device(device)
        rgb, labels = self._load_or_build(self.T)
        self.official = torch.from_numpy(rgb).to(self.device)
        self.labels = torch.from_numpy(labels).to(self.device)
        self._rgb_cache = _RES.BoundedCache(8, name="overcooked.rgb")

    @classmethod
    def get(cls, tile_size: int, device="cuda") -> OvercookedAtlas:
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
            except Exception:  # noqa: BLE001  corrupt/partial
                pass
        rgb, labels = _build_atlas(T)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, atlas=rgb, labels=labels, n_code=N_CODE)
        os.replace(tmp, path)
        return rgb, labels

    def rgb(self, values: dict) -> torch.Tensor:
        key = b"|".join(np.ascontiguousarray(np.asarray(values[k], np.float64)).tobytes() for k in sorted(values))
        hit = self._rgb_cache.get(key)
        if hit is not None:
            return hit
        pal = torch.as_tensor(_label_palette(values), device=self.device)
        return self._rgb_cache.put(key, self._recolour(pal))

    def _recolour(self, pal: torch.Tensor) -> torch.Tensor:
        f, T = SUBDIVS, self.T
        x = pal[self.labels.long()]  # (N, fT, fT, 3) float64
        x = x.reshape(N_CODE, T, f, T, f, 3).mean(4).mean(2)
        return x.to(torch.uint8).contiguous()


class _OvercookedLayout(Env):
    """Render a two-agent JaxMARL Overcooked layout from compact cell codes.

    The padded grid includes the layout’s outer wall ring. Codes determine pixels
    but do not restore full simulator State, and this adapter does not implement step.
    Use a separate environment for the pinned JaxMARL dependency profile."""

    family = "overcooked"

    #: jaxmarl layout key; the class name is the registry key
    layout: str = ""
    reference_id = "jaxmarl/Overcooked-V1"
    #: PADDED grid — the layout plus jaxmarl's one-cell wall ring, which IS what gets drawn
    grid_w: int = 0
    grid_h: int = 0

    @property
    def frame_shape(self) -> tuple[int, int]:
        """Return grid dimensions multiplied by the tile edge derived from ``size``."""
        t = self._tile(int(self.size))
        return self.grid_h * t, self.grid_w * t

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.atlas = OvercookedAtlas.get(self._tile(self.size), device=self.device)
        self._pool: np.ndarray | None = None
        self._pool_labels: np.ndarray | None = None
        self._default_t: torch.Tensor | None = None

    def _close(self) -> None:
        """Drop instance state/style caches; shared atlases remain available to other instances."""
        self._pool = None
        self._pool_labels = None
        self._default_t = None

    def _tile(self, size: int) -> int:
        t, r = divmod(int(size), self.grid_h)
        if r or t < 4:
            raise ValueError(
                f"{type(self).__name__}: size {size} is not a whole tile size for a {self.grid_w}x"
                f"{self.grid_h} grid; use a multiple of {self.grid_h} that is at least {4 * self.grid_h}"
            )
        return t

    def make_variation_space(self) -> Dict:
        """Expose painted source colours, background and grid-line colours.

        Purple is excluded because no atlas tile paints it. Recolour before
        downsampling so antialiased boundaries retain their semantic labels."""
        pal = official_colors()
        C2I, _, _, _ = _constants()
        space = {"background": RGBBox(init_value=DEFAULT_BG), "grid_line": RGBBox(init_value=DEFAULT_GRIDLINE)}
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
        return {
            "palette": pal,
            "background": np.trunc(np.asarray(d["background"], np.float64)),
            "grid_line": np.trunc(np.asarray(d["grid_line"], np.float64)),
        }

    def _label_palettes(self, sty) -> torch.Tensor:
        """Return per-row ``(B, N_LABEL, 3)`` float64 palettes."""
        B, dev = sty.batch, self.device
        C2I, _, _, _ = _constants()
        p = torch.zeros(B, N_LABEL, 3, dtype=torch.float64, device=dev)
        p[:, LBL_BG] = torch.trunc(sty["background"].double())
        p[:, LBL_GRID] = torch.trunc(sty["grid_line"].double())
        base = torch.as_tensor(official_colors(), dtype=torch.float64, device=dev)
        p[:, LBL_COLOR : LBL_COLOR + N_COLOR] = base.unsqueeze(0)
        for name in COLOR_NAMES:
            if name not in UNPAINTED:
                p[:, LBL_COLOR + C2I[name]] = torch.trunc(sty[name].double())
        return p

    def _is_default_style(self, sty) -> bool:
        if self._default_t is None:
            self._default_t = torch.as_tensor(self.default_style(), device=self.device, dtype=torch.float32)
        return bool((sty.values[0] == self._default_t).all())

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
        """Gather compact cell rows into uint8 frames with the layout’s rectangular shape."""
        if supersample not in (None, 1):
            raise ValueError(
                f"{type(self).__name__}: supersample={supersample} is not available. jaxmarl's own 3x "
                f"supersampling is already baked into the atlas by _render_tile; sampling again on top "
                f"of it would render something the reference never draws."
            )
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        tile = self._tile(int(size or self.size))
        atlas = self.atlas if tile == self.atlas.T else OvercookedAtlas.get(tile, device=self.device)
        sty = self.resolve_style(style, variation, batch=int(s.shape[0]))
        idx = torch.round(s).long().remainder(N_CODE).reshape(-1, self.grid_h, self.grid_w).to(torch.int32)
        flat = None
        if out is not None:
            flat = out.reshape(-1, self.grid_h * tile, self.grid_w * tile, 3)
        if sty.shared:
            rgb = (
                atlas.official
                if self._is_default_style(sty)
                else atlas.rgb(self._values(sty.values[0].detach().cpu().numpy()))
            )
            img = tm.blit(rgb, idx, tile, out=flat)
        else:
            img = self._blit_styled(atlas, idx, sty, tile, flat)
        return img.reshape(*lead, *img.shape[1:]) if lead else img

    def _blit_styled(self, atlas, idx: torch.Tensor, sty, tile: int, out) -> torch.Tensor:
        """Gather labels and apply per-row palettes before downsampling.

        Preserve float64 and the two means’ axis order to match shared-atlas recolouring."""
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
                lab = atlas.labels[idx[b0:b1, j].long()].long()  # (b, W, px, px)
                lab = lab.permute(0, 2, 1, 3).reshape(b1 - b0, px, W * px)
                col = pal[bi, lab]  # (b, px, W*px, 3) f64
                col = col.reshape(b1 - b0, tile, f, W * tile, f, 3).mean(4).mean(2)
                out[b0:b1, j * tile : (j + 1) * tile] = col.to(torch.uint8)
        return out

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample source rollout grids augmented with pot, placed-item and inventory variants.

        Augmented grids exercise renderer codes but are not unchanged rollout states.
        For sufficiently large batches, try to cover every painted palette label."""
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
        """Return a boolean table of palette labels painted by each cell code."""
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
                acts = {a: jax.random.randint(jax.random.fold_in(k, i), (), 0, 6) for i, a in enumerate(env.agents)}
                key, ks = jax.random.split(key)
                _obs, st, _r, _d, _i = env.step(ks, st, acts)

        pot = [CODE_OF[(O["pot"], int(vec[O["pot"]][1]), s, O["empty"])] for s in range(N_POT_STATUS)]
        placed = [CODE_OF[(O[t], int(vec[O[t]][1]), 0, O["empty"])] for t in ("onion", "plate", "dish")]
        wall = CODE_OF[(O["wall"], int(vec[O["wall"]][1]), 0, O["empty"])]
        held = (O["empty"], O["onion"], O["plate"], O["dish"])
        agent_of = {c: i for i, c in enumerate(ALPHABET) if c[0] == O["agent"]}

        # Cycle through pot statuses, placed items and held items for deterministic coverage.
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
        pool = np.stack(out[: max(n, len(base) + 3 * N_POT_STATUS)]).astype(np.float32)
        codes = np.round(pool).astype(np.int64) % N_CODE
        cl = self._code_labels()
        has = np.stack([cl[codes[i]].any(0) for i in range(len(codes))])
        return pool, has

    @classmethod
    def encode_state(cls, st) -> np.ndarray:
        """Extract compact render codes from the source map crop and agent inventory."""
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
                        f"64-entry alphabet enumerated from its own constants"
                    )
                out[y * W + x] = CODE_OF[key]
        return out

    @classmethod
    def reference_env(cls, size: int | None = None):
        """Return the source visualizer class and tile size without opening an interactive window."""
        from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer

        return (OvercookedVisualizer, int(size or cls.default_size) // cls.grid_h)

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Restore render inputs, verify their round trip, and call the source grid renderer.

        Pass two agent directions: a one-element array makes the source overwrite all
        facings. Agents with the same colour must share the same held item."""
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
                            f"parity number taken against it would compare two different scenes"
                        )
                    seen[slot] = h
                    inv[slot] = h
            fr = viz._render_grid(
                grid, tile_size=tile, highlight_mask=None, agent_dir_idx=np.array([0, 1]), agent_inv=inv
            )
            back = cls.encode_grid(grid.astype(np.int64), inv)
            if not np.array_equal(np.round(back).astype(np.int64), codes):
                raise AssertionError(
                    f"{cls.__name__}: the state written into the reference does not read back as the "
                    f"state requested; parity measured against this frame would compare two scenes"
                )
            out.append(np.asarray(fr, np.uint8))
        return np.stack(out)


def _cell_names(h: int, w: int) -> tuple[str, ...]:
    return tuple(f"cell[{r},{c}]" for r in range(h) for c in range(w))


def layout(class_name: str, key: str, doc: str = "", tile: int = TILE_PIXELS) -> type:
    """Register a layout with padded dimensions read from the source layout table."""
    from jaxmarl.environments.overcooked.layouts import overcooked_layouts

    if key not in overcooked_layouts:
        raise ImportError(f"jaxmarl has no Overcooked layout {key!r}; it ships {sorted(overcooked_layouts)}")
    lay = overcooked_layouts[key]
    w, h = int(lay["width"]) + 2, int(lay["height"]) + 2  # + jaxmarl's one-cell wall ring
    cls = type(
        class_name,
        (_OvercookedLayout,),
        {
            "__doc__": doc or f"jaxmarl Overcooked, layout {key}: {w}x{h} padded grid.",
            "__module__": __name__,
            "name": "overcooked-" + key.replace("_", "-"),
            "layout": key,
            "grid_w": w,
            "grid_h": h,
            "state_dim": w * h,
            "state_names": _cell_names(h, w),
            "default_size": h * tile,
        },
    )
    return register(cls)


OvercookedCrampedRoom = layout(
    "OvercookedCrampedRoom",
    "cramped_room",
    doc="Overcooked cramped_room: two chefs, one pot, one counter run — the layout every Overcooked "
    "paper reports first. 42 numbers against a 192x224 frame.",
)
OvercookedAsymmAdvantages = layout(
    "OvercookedAsymmAdvantages",
    "asymm_advantages",
    doc="Overcooked asymm_advantages: the two chefs have different access to the pots, so the "
    "layout is not left-right symmetric and neither is any state in it.",
)
OvercookedCoordRing = layout(
    "OvercookedCoordRing",
    "coord_ring",
    doc="Overcooked coord_ring: a ring corridor, the layout where the two chefs must pass each other.",
)
OvercookedForcedCoord = layout(
    "OvercookedForcedCoord",
    "forced_coord",
    doc="Overcooked forced_coord: a dividing wall neither chef can cross, so every soup needs a "
    "hand-off across the counter.",
)
OvercookedCounterCircuit = layout(
    "OvercookedCounterCircuit",
    "counter_circuit",
    doc="Overcooked counter_circuit: the widest shipped layout, a 9x7 padded grid with a long counter loop.",
)
