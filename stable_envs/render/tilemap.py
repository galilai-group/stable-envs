"""Tile-atlas rendering: frames as a pure gather, for environments drawn as a grid of tiles.

A MiniGrid frame is a grid of tiles, and ``minigrid.core.grid.Grid.render_tile`` caches its output
under the key ``obj.encode() + (agent_dir, highlight, tile_size)`` — upstream therefore already
asserts that a tile image is a function of nothing else. That makes the appearance set finite: 11
object types x 6 colours x 3 states x 5 agent-direction slots x 2 highlight states = 1980 entries, of
which 492 are distinct images and 580 are reachable from a real ``encode()``. Precompute them once and
rendering is a gather with no rasterisation at all, so it is exact by construction rather than by
matching a rasterizer pixel for pixel.

Measured on this path (see ``docs/research/minigrid.md``): **21,216 frames across all 172 instantiable
MiniGrid/BabyAI ids at two tile sizes, every one bit-identical to the official renderer, max absolute
pixel difference 0**, and **6.55M agent-POV frames/s** at 224x224 on one H200 against 2,378 f/s for
the official CPU renderer, sustaining 987 GB/s — memory-bandwidth-bound, not compute-bound. The blit
in isolation reaches 1.1 TB/s, so the gather is already at the roofline and the only way to go faster
is to emit fewer bytes.

Two things here are deliberately not what they first look like.

**The atlas is built by calling the official renderer, never by reimplementing it.** The tiles come
back from ``Grid.render_tile`` and are cast to uint8 exactly the way ``Grid.render`` casts when it
assigns a float tile into a uint8 buffer. Exactness is then structural. In particular the atlas must
*not* be built through ``WorldObj.decode``: decode throws colour away for ``Goal`` and ``Lava`` — it
always returns a green goal — while ``Goal.render`` reads ``self.color``. An atlas built that way
renders a purple goal green and nothing anywhere would flag it, because the encoding round-trips and
the picture looks entirely plausible.

**Variations recolour through a label map, not through pixel edits.** Alongside the RGB atlas we build,
at MiniGrid's own 3x supersampled resolution, a map saying *which* of 22 semantic slots painted each
subpixel (background, grid line, agent, lava, and the 6 palette colours in their full, half and 0.45
shades — ``Floor`` paints ``COLORS[c]/2`` and a locked ``Door``'s inner panel ``0.45*COLORS[c]``).
Every subpixel has exactly one writer, so an atlas for any palette is ``palette[label]`` reduced by the
same two-stage box filter ``minigrid.utils.rendering.downsample`` applies. This matters because a
naive recolour is wrong in two ways that are invisible by eye: grey walls and the grid lines are both
literally ``(100,100,100)``, so colour matching cannot separate them; and ``highlight_img`` truncates
to uint8 *before* the downsample, so a highlighted tile is not the alpha blend of the downsampled tile.
The label decomposition handles both — highlight becomes an effective palette
``floor(p + alpha*(hc - p))`` applied per label, which is exact because the blend is per subpixel and
each subpixel has a single label. ``build()`` asserts the reconstruction reproduces the official atlas
byte for byte at the default palette, so a wrong constant cannot survive the build.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

from . import _resources as _RES

try:  # triton is optional; the torch path is the readable one
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # noqa: BLE001
    HAVE_TRITON = False

# ---------------------------------------------------------------- MiniGrid's alphabet

N_TYPE, N_COLOR, N_STATE, N_DIR, N_HL = 11, 6, 3, 5, 2
NO_AGENT = 4  #: agent_dir slot meaning "no agent stands on this tile"
N_GEOM = N_DIR * N_TYPE * N_COLOR * N_STATE  # 990 distinct tile geometries
N_SLOT = N_HL * N_GEOM  # 1980 atlas entries
SUBDIVS = 3  #: Grid.render_tile's supersampling factor

T_UNSEEN, T_EMPTY, T_WALL, T_FLOOR, T_DOOR = 0, 1, 2, 3, 4
T_KEY, T_BALL, T_BOX, T_GOAL, T_LAVA, T_AGENT = 5, 6, 7, 8, 9, 10
C_GREY = 5

#: label ids for the recolouring decomposition
LBL_BG, LBL_GRID, LBL_AGENT, LBL_LAVA = 0, 1, 2, 3
LBL_PAL, LBL_PAL_HALF, LBL_PAL_45 = 4, 10, 16
N_LABEL = 22

DEFAULT_LAVA = (255, 128, 0)
DEFAULT_GRIDLINE = (100, 100, 100)
DEFAULT_AGENT = (255, 0, 0)
DEFAULT_BG = (0, 0, 0)
DEFAULT_HIGHLIGHT = (255, 255, 255)
DEFAULT_HIGHLIGHT_ALPHA = 0.30


def official_colors() -> np.ndarray:
    """``COLORS`` as a (6,3) uint8 array in ``COLOR_TO_IDX`` order, read from the package."""
    from minigrid.core.constants import COLOR_TO_IDX, COLORS

    out = np.zeros((N_COLOR, 3), np.uint8)
    for name, i in COLOR_TO_IDX.items():
        out[i] = COLORS[name]
    return out


def reachable_codes() -> list[tuple[int, int, int]]:
    """Every ``(type, colour, state)`` an ``encode()`` can produce. 58 of the 198 combinations.

    ``unseen``/``empty``/``agent`` all mean "no object" and get one slot each; ``door`` is the only
    type whose state channel is used; ``Lava.__init__`` hardcodes red so lava has a single encoding.
    """
    out = []
    for t in range(N_TYPE):
        if t in (T_UNSEEN, T_EMPTY, T_AGENT):
            out.append((t, 0, 0))
        elif t == T_DOOR:
            out += [(t, c, s) for c in range(N_COLOR) for s in range(N_STATE)]
        elif t == T_LAVA:
            out.append((t, 0, 0))
        else:
            out += [(t, c, 0) for c in range(N_COLOR)]
    return out


REACHABLE = reachable_codes()
N_CODE = len(REACHABLE)  # 58


def atlas_flat_index(hl, d, t, c, s):
    """Index into the atlas. ``hl*990 + geometry``, so labels[geom] lines up with both halves."""
    return (((hl * N_DIR + d) * N_TYPE + t) * N_COLOR + c) * N_STATE + s


def make_obj(t: int, c: int, s: int):
    """The object with encoding ``(t, c, s)``, built faithfully rather than through ``decode``."""
    from minigrid.core.constants import IDX_TO_COLOR
    from minigrid.core.world_object import Ball, Box, Door, Floor, Goal, Key, Lava, Wall

    col = IDX_TO_COLOR[c]
    if t in (T_UNSEEN, T_EMPTY, T_AGENT):
        return None
    return {
        T_WALL: lambda: Wall(col),
        T_FLOOR: lambda: Floor(col),
        T_DOOR: lambda: Door(col, is_open=(s == 0), is_locked=(s == 2)),
        T_KEY: lambda: Key(col),
        T_BALL: lambda: Ball(col),
        T_BOX: lambda: Box(col),
        T_GOAL: lambda: Goal(col),
        T_LAVA: Lava,
    }[t]()


# ---------------------------------------------------------------- building the atlas


def _official_atlas(tile_size: int) -> np.ndarray:
    """(1980,T,T,3) uint8, straight out of ``Grid.render_tile``."""
    from minigrid.core.grid import Grid

    out = np.zeros((N_SLOT, tile_size, tile_size, 3), np.uint8)
    for hl in range(N_HL):
        for d in range(N_DIR):
            for t in range(N_TYPE):
                for c in range(N_COLOR):
                    for s in range(N_STATE):
                        img = Grid.render_tile(
                            make_obj(t, c, s),
                            agent_dir=(d if d < 4 else None),
                            highlight=bool(hl),
                            tile_size=tile_size,
                        )
                        # Grid.render assigns this float array into a uint8 frame; same cast here
                        out[atlas_flat_index(hl, d, t, c, s)] = img.astype(np.uint8)
    return out


def _label_for(color, color_idx: int) -> int:
    """Which semantic slot a colour passed to ``fill_coords`` belongs to.

    Dispatch is on the value *before* it is truncated into the uint8 image, because ``Floor`` passes
    ``COLORS[c]/2`` and a locked ``Door`` passes ``0.45*COLORS[c]`` — both non-integral.
    """
    v = np.asarray(color, np.float64).ravel()
    if np.allclose(v, 0.0):
        return LBL_BG
    if np.allclose(v, DEFAULT_LAVA):
        return LBL_LAVA
    base = official_colors()[color_idx].astype(np.float64)
    for lbl, cand in (
        (LBL_PAL + color_idx, base),
        (LBL_PAL_HALF + color_idx, base / 2),
        (LBL_PAL_45 + color_idx, 0.45 * base),
    ):
        if np.allclose(v, cand):
            return lbl
    raise ValueError(
        f"unrecognised fill colour {v} for colour index {color_idx}; the label "
        f"decomposition is incomplete and a recoloured atlas would be wrong"
    )


def _label_maps(tile_size: int) -> np.ndarray:
    """(990, 3T, 3T) uint8: which label painted each supersampled subpixel.

    Mirrors ``Grid.render_tile``'s call sequence, but every pixel is still set by the official
    ``fill_coords`` and the official predicates — objects draw through a recording shim installed over
    ``minigrid.core.world_object.fill_coords``, which is the only way a ``WorldObj`` can paint. The two
    grid-line rectangles and the agent triangle are the sole constants restated here, and ``build()``
    catches any divergence by diffing the reconstruction against the official atlas.
    """
    import minigrid.core.world_object as wo
    from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_triangle, rotate_fn

    n = tile_size * SUBDIVS
    out = np.zeros((N_GEOM, n, n), np.uint8)
    scratch = np.zeros((n, n, 3), np.uint8)
    cache: dict[tuple, np.ndarray] = {}
    for d in range(N_DIR):
        for t in range(N_TYPE):
            for c in range(N_COLOR):
                for s in range(N_STATE):
                    obj = make_obj(t, c, s)
                    key = (obj.encode() if obj is not None else None, d if d < 4 else None)
                    if key not in cache:
                        lab = np.zeros((n, n, 3), np.uint8)  # LBL_BG == 0
                        fill_coords(lab, point_in_rect(0, 0.031, 0, 1), (LBL_GRID,) * 3)
                        fill_coords(lab, point_in_rect(0, 1, 0, 0.031), (LBL_GRID,) * 3)
                        if obj is not None:
                            orig = wo.fill_coords
                            wo.fill_coords = lambda img, fn, color, _l=lab, _c=c, _o=orig: _o(
                                _l, fn, (_label_for(color, _c),) * 3
                            )
                            try:
                                obj.render(scratch)
                            finally:
                                wo.fill_coords = orig
                        if d < 4:
                            tri = rotate_fn(
                                point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81)),
                                cx=0.5,
                                cy=0.5,
                                theta=0.5 * math.pi * d,
                            )
                            fill_coords(lab, tri, (LBL_AGENT,) * 3)
                        cache[key] = lab[:, :, 0].copy()
                    out[atlas_flat_index(0, d, t, c, s)] = cache[key]
    return out


def label_palette(values: dict) -> np.ndarray:
    """(22,3) float64: the colour each label paints, for a set of variation values.

    Half and 0.45 shades are truncated because ``fill_coords`` assigns the float colour straight into
    a uint8 image, so those are the values that actually land on a subpixel.
    """
    pal = np.asarray(values["palette"], np.float64).reshape(N_COLOR, 3)
    p = np.zeros((N_LABEL, 3), np.float64)
    p[LBL_BG] = np.asarray(values.get("background_color", DEFAULT_BG), np.float64)
    p[LBL_GRID] = np.asarray(values.get("grid_line_color", DEFAULT_GRIDLINE), np.float64)
    p[LBL_AGENT] = np.asarray(values.get("agent_color", DEFAULT_AGENT), np.float64)
    p[LBL_LAVA] = np.asarray(values.get("lava_color", DEFAULT_LAVA), np.float64)
    p[LBL_PAL : LBL_PAL + N_COLOR] = pal
    p[LBL_PAL_HALF : LBL_PAL_HALF + N_COLOR] = np.trunc(pal / 2)
    p[LBL_PAL_45 : LBL_PAL_45 + N_COLOR] = np.trunc(0.45 * pal)
    return p


def highlight_palette(p: np.ndarray, color, alpha: float) -> np.ndarray:
    """``highlight_img`` applied per label.

    Reproduces the reference arithmetic including its uint8 subtraction: ``highlight_img`` computes
    ``img + alpha*(np.array(color, uint8) - img)``, so the difference wraps mod 256 when the highlight
    colour is darker than the pixel. At MiniGrid's own ``(255,255,255)`` nothing wraps and this is a
    plain blend; the wrap is reproduced anyway so that a user-chosen dark highlight matches what the
    reference would have done rather than what it arguably meant.
    """
    diff = (np.asarray(color, np.int64) - p.astype(np.int64)) % 256
    return np.floor(np.clip(p + float(alpha) * diff, 0, 255))


def recolour(labels: torch.Tensor, pal_lo, pal_hi, tile_size: int, chunk: int = 128) -> torch.Tensor:
    """(1980,T,T,3) uint8 atlas from label maps and two effective palettes (no-highlight, highlight).

    The reduction is ``downsample``'s two-stage box filter in its original axis order, in float64, so
    the last-ULP behaviour matches and the truncation to uint8 lands on the same integer.
    """
    N, n, _ = labels.shape
    T, f = tile_size, SUBDIVS
    out = torch.empty(N_SLOT, T, T, 3, dtype=torch.uint8, device=labels.device)
    for hl, pal in ((0, pal_lo), (1, pal_hi)):
        pt = torch.as_tensor(np.asarray(pal, np.float64), device=labels.device)
        for i in range(0, N, chunk):
            lab = labels[i : i + chunk].long()
            sub = pt[lab]  # (n_chunk, 3T, 3T, 3) float64
            x = sub.reshape(lab.shape[0], T, f, T, f, 3).mean(4).mean(2)
            out[hl * N_GEOM + i : hl * N_GEOM + i + lab.shape[0]] = x.to(torch.uint8)
    return out


def _cache_path(tile_size: int) -> str:
    root = os.environ.get("STABLE_ENVS_CACHE") or os.path.expanduser("~/.cache/stable-envs")
    return os.path.join(root, "tilemap", f"minigrid_t{tile_size}.npz")


def _resolve(device) -> torch.device:
    """``"cuda"`` -> the CURRENT cuda device, with its index filled in.

    Any cache keyed by the string a caller passed cannot tell ``"cuda"`` on one GPU from ``"cuda"`` on
    another, because the string is identical and the device is not.
    """
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        d = torch.device("cuda", torch.cuda.current_device())
    return d


class MiniGridAtlas:
    """The tile atlas for one tile size, plus recoloured variants cached by variation values.

    Building costs ~75 s of pure-Python scanline fills, so the result is memoised in-process and on
    disk. The disk write is atomic because a test session may build several envs at once.
    """

    _mem: dict[tuple[int, str], MiniGridAtlas] = {}

    def __init__(self, tile_size: int, device="cuda", verify: bool = True):
        self.T = int(tile_size)
        self.device = torch.device(device)
        atlas, labels = self._load_or_build(self.T, verify)
        self.official = torch.from_numpy(atlas).to(self.device)
        self.labels = torch.from_numpy(labels).to(self.device)
        self._rgb_cache = _RES.BoundedCache(8, name="tilemap.rgb")

    @classmethod
    def get(cls, tile_size: int, device="cuda") -> MiniGridAtlas:
        # RESOLVE the device before keying on it. ``str("cuda")`` is the same key on every GPU, while
        # ``torch.device("cuda")`` means whichever one is current -- so a process that renders on
        # cuda:0 and then on cuda:1 was handed cuda:0's atlas the second time and Triton refused the
        # launch. ``_resolve`` turns both "cuda" and "cuda:1" into a device carrying an explicit index.
        key = (int(tile_size), str(_resolve(device)))
        if key not in cls._mem:
            cls._mem[key] = cls(tile_size, _resolve(device))
        return cls._mem[key]

    @staticmethod
    def _load_or_build(T: int, verify: bool):
        path = _cache_path(T)
        if os.path.exists(path):
            try:
                z = np.load(path)
                return z["atlas"], z["labels"]
            except Exception:  # noqa: BLE001  corrupt/partial cache
                pass
        atlas, labels = _official_atlas(T), _label_maps(T)
        if verify:
            pal = {"palette": official_colors()}
            p = label_palette(pal)
            rec = recolour(
                torch.from_numpy(labels), p, highlight_palette(p, DEFAULT_HIGHLIGHT, DEFAULT_HIGHLIGHT_ALPHA), T
            ).numpy()
            bad = int((rec != atlas).any(axis=(1, 2, 3)).sum())
            if bad:
                raise AssertionError(
                    f"label-map reconstruction differs from the official atlas in {bad}/{N_SLOT} "
                    f"tiles at the default palette — the decomposition is wrong, so every recoloured "
                    f"frame would be wrong in a way no parity test would catch"
                )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:  # a handle, not a name: savez appends .npz to names
            np.savez_compressed(fh, atlas=atlas, labels=labels)
        os.replace(tmp, path)
        return atlas, labels

    def rgb(self, values: dict) -> torch.Tensor:
        """(1980,T,T,3) uint8 atlas for these variation values. Default values return the official one."""
        key = b"|".join(np.ascontiguousarray(np.asarray(values[k], np.float64)).tobytes() for k in sorted(values))
        hit = self._rgb_cache.get(key)
        if hit is not None:
            return hit
        if self._is_default(values):
            out = self.official
        else:
            p = label_palette(values)
            hp = highlight_palette(
                p,
                values.get("highlight_color", DEFAULT_HIGHLIGHT),
                float(np.asarray(values.get("highlight_alpha", DEFAULT_HIGHLIGHT_ALPHA)).ravel()[0]),
            )
            out = recolour(self.labels, p, hp, self.T)
        return self._rgb_cache.put(key, out)

    @staticmethod
    def _is_default(values: dict) -> bool:
        def same(k, ref):
            return np.array_equal(np.asarray(values.get(k, ref), np.float64), np.asarray(ref, np.float64))

        return (
            same("palette", official_colors())
            and same("background_color", DEFAULT_BG)
            and same("grid_line_color", DEFAULT_GRIDLINE)
            and same("agent_color", DEFAULT_AGENT)
            and same("lava_color", DEFAULT_LAVA)
            and same("highlight_color", DEFAULT_HIGHLIGHT)
            and float(np.asarray(values.get("highlight_alpha", DEFAULT_HIGHLIGHT_ALPHA)).ravel()[0])
            == DEFAULT_HIGHLIGHT_ALPHA
        )


# ---------------------------------------------------------------- the gather itself


def word_size(tile_size: int) -> int:
    """Widest word (bytes) that evenly divides a tile row of ``T*3`` bytes.

    4 for the usual T=32; rjepa renders MiniGrid at T=14 where ``T*3 = 42`` and the widest safe word
    is 2. Every offset the kernel forms is a whole number of tile rows on both sides — the atlas
    strides by ``T*T*3`` and the output by ``W*T*3`` — so alignment follows.
    """
    row = tile_size * 3
    return 4 if row % 4 == 0 else (2 if row % 2 == 0 else 1)


_WORD_DTYPE = {4: torch.int32, 2: torch.int16, 1: torch.int8}

if HAVE_TRITON:

    @triton.jit
    def _blit_kernel(
        atlas_ptr,
        idx_ptr,
        out_ptr,
        H: tl.constexpr,
        W: tl.constexpr,
        T: tl.constexpr,
        WPTR: tl.constexpr,
        WPR: tl.constexpr,
        WPT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)  # flattened (batch, output row)
        b = row // (H * T)
        y = row % (H * T)
        tj, py = y // T, y % T
        w = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = w < WPR
        tile = tl.load(idx_ptr + b * (H * W) + tj * W + w // WPTR, mask=m, other=0)
        src = tile.to(tl.int64) * WPT + py * WPTR + (w % WPTR)
        tl.store(out_ptr + row.to(tl.int64) * WPR + w, tl.load(atlas_ptr + src, mask=m, other=0), mask=m)


def blit(atlas_rgb: torch.Tensor, idx: torch.Tensor, tile_size: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """``idx`` (B,H,W) int32 of atlas slots -> ``(B, H*T, W*T, 3)`` uint8, in one pass.

    One program per (batch, output row, column block); each lane resolves its own tile and copies a
    word. Writing the final layout directly matters: the obvious ``atlas[idx].permute(...).reshape(...)``
    materialises a full-size tiled buffer and then transposes it, and measures 2.5-3.5x slower.
    """
    B, H, W = idx.shape
    if not HAVE_TRITON:
        return blit_torch(atlas_rgb, idx, out=out)
    T = tile_size
    w = word_size(T)
    wpr = W * T * 3 // w
    if out is None:
        out = torch.empty(B, H * T, W * T, 3, dtype=torch.uint8, device=idx.device)
    atlas_w = atlas_rgb.reshape(-1).view(_WORD_DTYPE[w]).view(atlas_rgb.shape[0], T * T * 3 // w)
    BLOCK = 256 if wpr >= 256 else triton.next_power_of_2(wpr)
    # Launch on the tensors' device rather than the process's current one -- see the same guard in
    # ``render/raster2d.py:render_triton`` for the failure it prevents.
    with torch.cuda.device(idx.device):
        _blit_kernel[(B * H * T, triton.cdiv(wpr, BLOCK))](
            atlas_w,
            idx.contiguous(),
            out.view(-1).view(_WORD_DTYPE[w]),
            H=H,
            W=W,
            T=T,
            WPTR=T * 3 // w,
            WPR=wpr,
            WPT=T * T * 3 // w,
            BLOCK=BLOCK,
            num_warps=4,
        )
    return out


def blit_torch(atlas_rgb: torch.Tensor, idx: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """The readable reference for :func:`blit`: gather whole tiles, then untile."""
    B, H, W = idx.shape
    T = atlas_rgb.shape[1]
    img = atlas_rgb[idx.long()].permute(0, 1, 3, 2, 4, 5).reshape(B, H * T, W * T, 3)
    if out is not None:
        out.copy_(img)
        return out
    return img


# ---------------------------------------------------------------- the visibility scan

if HAVE_TRITON:

    @triton.jit
    def _vis_kernel(t_ptr, m_ptr, n, V: tl.constexpr, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = off < n
        t = tl.load(t_ptr + off, mask=ok, other=0)
        m = tl.zeros((BLOCK,), tl.int64) + (1 << ((V // 2) * V + (V - 1)))
        for j in range(V - 1, -1, -1):
            for i in range(0, V - 1):
                k = i * V + j
                c = (m >> k) & (t >> k) & 1
                m = m | (c << ((i + 1) * V + j))
                if j > 0:
                    m = m | (c << ((i + 1) * V + j - 1))
                    m = m | (c << (i * V + j - 1))
            for i in range(V - 1, 0, -1):
                k = i * V + j
                c = (m >> k) & (t >> k) & 1
                m = m | (c << ((i - 1) * V + j))
                if j > 0:
                    m = m | (c << ((i - 1) * V + j - 1))
                    m = m | (c << (i * V + j - 1))
        tl.store(m_ptr + off, m, mask=ok)


def visibility_scan(transparent: torch.Tensor, view_size: int) -> torch.Tensor:
    """``Grid.process_vis``, batched. ``transparent`` (B,V,V) bool indexed [b,i,j] -> visible mask.

    The sweep is genuinely sequential — the left-to-right pass reads cells it just wrote, and the
    right-to-left pass reads what the left-to-right pass wrote — so it cannot be reordered or merged.
    Transcribed elementwise it costs ~670 tiny kernel launches for V=7 and dominated everything else in
    the pipeline (0.71M f/s). Packing the whole V*V mask into one int64, bit ``k = i*V + j``, folds all
    ``2*V*(V-1)`` steps into a single launch and measured 9.24M f/s — a 13x index-path speedup.
    """
    B = transparent.shape[0]
    V = view_size
    if not HAVE_TRITON or V * V > 63:
        return visibility_scan_torch(transparent, V)
    p2 = torch.ones(V * V, dtype=torch.int64, device=transparent.device) << torch.arange(
        V * V, device=transparent.device
    )
    bits = (transparent.reshape(B, -1).to(torch.int64) * p2).sum(1).contiguous()
    m = torch.empty_like(bits)
    BLOCK = 128
    _vis_kernel[(triton.cdiv(B, BLOCK),)](bits, m, B, V=V, BLOCK=BLOCK, num_warps=4)
    return ((m.unsqueeze(1) >> torch.arange(V * V, device=m.device)) & 1).bool().reshape(B, V, V)


def visibility_scan_torch(transparent: torch.Tensor, view_size: int) -> torch.Tensor:
    """Literal transcription of ``Grid.process_vis``; the definition the kernel is checked against."""
    B, V = transparent.shape[0], view_size
    mask = torch.zeros(B, V, V, dtype=torch.bool, device=transparent.device)
    mask[:, V // 2, V - 1] = True
    for j in reversed(range(V)):
        for i in range(0, V - 1):
            c = mask[:, i, j] & transparent[:, i, j]
            mask[:, i + 1, j] |= c
            if j > 0:
                mask[:, i + 1, j - 1] |= c
                mask[:, i, j - 1] |= c
        for i in reversed(range(1, V)):
            c = mask[:, i, j] & transparent[:, i, j]
            mask[:, i - 1, j] |= c
            if j > 0:
                mask[:, i - 1, j - 1] |= c
                mask[:, i, j - 1] |= c
    return mask


def view_offsets(view_size: int, device) -> torch.Tensor:
    """``offs[d, vj, vi] = (dx, dy)`` of the world cell shown at agent-view cell ``(vi, vj)``.

    This is ``Grid.slice`` composed with ``rotate_left**(d+1)``, in closed form. The form is the one
    ``MiniGridEnv.get_full_render`` uses for its highlight mask; it was validated against a
    tagged-object trace of the real ``slice``/``rotate_left`` for every view size MiniGrid permits
    (V in 3,5,7,9,11), and found translation-invariant, so it is a gather rather than a copy.
    """
    dirs = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=device)
    V = view_size
    vj = torch.arange(V, device=device).view(V, 1)
    vi = torch.arange(V, device=device).view(1, V)
    offs = torch.empty(4, V, V, 2, dtype=torch.long, device=device)
    for d in range(4):
        f = dirs[d]
        r = torch.tensor([-f[1], f[0]], device=device)
        top = f * (V - 1) - r * (V // 2)
        for k in range(2):
            offs[d, :, :, k] = top[k] - f[k] * vj + r[k] * vi
    return offs
