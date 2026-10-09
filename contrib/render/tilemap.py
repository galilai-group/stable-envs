from __future__ import annotations

import math
import os
import tempfile
import threading
import warnings
from importlib.metadata import version
from zipfile import BadZipFile

import numpy as np
import torch

from stable_envs.render import _resources as _RES

from ._cache import cache_path

try:  # triton is optional; the torch path is the readable one
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

N_TYPE, N_COLOR, N_STATE, N_DIR, N_HL = 11, 6, 3, 5, 2
NO_AGENT = 4  #: agent_dir slot meaning "no agent stands on this tile"
N_GEOM = N_DIR * N_TYPE * N_COLOR * N_STATE  # 990 distinct tile geometries
N_SLOT = N_HL * N_GEOM  # 1980 atlas entries
SUBDIVS = 3  #: Grid.render_tile's supersampling factor
CACHE_FORMAT = 1

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
    """Return the type, colour, and state codes reachable through MiniGrid encoding."""
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
    """Build by constructor: ``decode`` discards non-default Goal colours."""
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
    """Identify a fill before uint8 conversion, preserving fractional Floor/Door shades."""
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
    """(990, 3T, 3T) uint8 labels painted by MiniGrid predicates.

    Temporarily replaces the upstream fill function to record semantic labels;
    unrelated upstream rendering must not run concurrently. Reconstruction is
    checked against official tiles before the atlas is cached.
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
                            # This global patch is not safe alongside unrelated upstream rendering.
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
    """Apply MiniGrid highlighting per label before downsampling.

    Subtraction wraps modulo 256, matching upstream uint8 arithmetic even when
    the highlight is darker than the pixel. Truncate before the box filter.
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


def _cache_path(tile_size: int):
    return cache_path("tilemap", f"minigrid_t{tile_size}.npz")


def _resolve(device) -> torch.device:
    """Resolve an unspecified CUDA index so cache keys distinguish physical devices."""
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        d = torch.device("cuda", torch.cuda.current_device())
    return d


class MiniGridAtlas:
    """Official MiniGrid tiles and recolouring labels, cached by tile size and device.

    Atlas construction is serialized within this class. It temporarily patches
    MiniGrid's object renderer, so do not render upstream objects during a build.
    """

    _mem: dict[tuple[int, str], MiniGridAtlas] = {}
    _lock = threading.RLock()

    def __init__(self, tile_size: int, device="cuda", verify: bool = True):
        self.T = int(tile_size)
        self.device = torch.device(device)
        with self._lock:
            atlas, labels = self._load_or_build(self.T, verify)
        self.official = torch.from_numpy(atlas).to(self.device)
        self.labels = torch.from_numpy(labels).to(self.device)
        self._rgb_cache = _RES.BoundedCache(8, name="tilemap.rgb")

    @classmethod
    def get(cls, tile_size: int, device="cuda") -> MiniGridAtlas:
        device = _resolve(device)
        key = (int(tile_size), str(device))
        with cls._lock:
            if key not in cls._mem:
                cls._mem[key] = cls(tile_size, device)
            return cls._mem[key]

    @staticmethod
    def _load_or_build(T: int, verify: bool):
        path = _cache_path(T)
        source_version = version("minigrid")
        try:
            with np.load(path, allow_pickle=False) as z:
                if z["format"].item() != CACHE_FORMAT or z["minigrid_version"].item() != source_version:
                    raise ValueError("atlas format or MiniGrid version mismatch")
                atlas, labels = z["atlas"], z["labels"]
                if (
                    atlas.shape != (N_SLOT, T, T, 3)
                    or labels.shape != (N_GEOM, T * SUBDIVS, T * SUBDIVS)
                    or atlas.dtype != np.uint8
                    or labels.dtype != np.uint8
                    or labels.max() >= N_LABEL
                ):
                    raise ValueError("invalid atlas or label array")
                return atlas, labels
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
            warnings.warn(f"Rebuilding MiniGrid cache {path}: {exc}", RuntimeWarning, stacklevel=2)
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
        tmp = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
            ) as fh:
                tmp = fh.name
                np.savez_compressed(
                    fh, atlas=atlas, labels=labels, format=CACHE_FORMAT, minigrid_version=source_version
                )
            os.replace(tmp, path)
        except OSError as exc:
            warnings.warn(f"MiniGrid cache is memory-only; cannot write {path}: {exc}", RuntimeWarning, stacklevel=2)
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    warnings.warn(f"Cannot remove temporary MiniGrid cache {tmp}: {exc}", RuntimeWarning, stacklevel=2)
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


def word_size(tile_size: int) -> int:
    """Return the widest aligned copy width for a tile row of T * 3 bytes."""
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
    """Gather (B,H,W) tile indices into (B,H*T,W*T,3) uint8 frames.

    CUDA writes the final layout directly, avoiding an intermediate tiled image.
    CPU or installations without Triton use the torch implementation.
    """
    B, H, W = idx.shape
    if not HAVE_TRITON or idx.device.type != "cuda":
        return blit_torch(atlas_rgb, idx, out=out)
    T = tile_size
    w = word_size(T)
    wpr = W * T * 3 // w
    if out is None:
        out = torch.empty(B, H * T, W * T, 3, dtype=torch.uint8, device=idx.device)
    atlas_w = atlas_rgb.reshape(-1).view(_WORD_DTYPE[w]).view(atlas_rgb.shape[0], T * T * 3 // w)
    BLOCK = 256 if wpr >= 256 else triton.next_power_of_2(wpr)
    # The tensor's device can differ from the process's current CUDA device.
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
    """Return the (B,V,V) visibility mask for a boolean transparency grid.

    Each left-to-right sweep reads its own writes; the reverse sweep reads those
    results. Preserve that order. CUDA packs grids up to 63 cells into int64.
    """
    B = transparent.shape[0]
    V = view_size
    if not HAVE_TRITON or transparent.device.type != "cuda" or V * V > 63:
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
    """Return world-cell offsets indexed by (direction, view_row, view_column).

    Composes Grid.slice with rotate_left**(direction+1) without copying grids.
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
