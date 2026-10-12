from __future__ import annotations

import importlib.util
import os

import numpy as np
import torch

from stable_envs.env import Env
from stable_envs.registry import register
from stable_envs.spaces import Box, Dict, Discrete

_SPEC = importlib.util.find_spec("procgen")
if _SPEC is None or _SPEC.origin is None:  # optional dependency: lose this env only
    raise ImportError(
        "procgen is not installed; `pip install procgen` provides both the assets this "
        "renderer packs and the reference env parity is measured against"
    )
ASSET_ROOT = os.path.join(os.path.dirname(_SPEC.origin), "data", "assets")
if not os.path.isdir(ASSET_ROOT):
    raise ImportError(f"procgen is importable but its asset tree is missing at {ASSET_ROOT}")

try:  # triton is optional; the torch path is the readable one
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # noqa: BLE001
    HAVE_TRITON = False

F = np.float32
PI_F = F(3.14159265358979323846264338327950288)  # cpp-utils.h's `const float PI`
DEG2RAD = 0.017453292519943295769  # QTransform::rotate's own constant
RENDER_EPS = 0.02  # basic-abstract-game.cpp
P_SPACE, P_WALL = 100, 51  # object-ids.h


def _qround(v):
    """Round halves away from zero, matching Qt qRound."""
    v = np.asarray(v, np.float64)
    return np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5)).astype(np.int64)


def _to_rect(x, y, w, h):
    """Convert floating edges to integer (left, top, width, height) as QRectF::toRect does."""
    l, t = _qround(x), _qround(y)
    return l, t, _qround(x + w) - l, _qround(y + h) - t


def _prepare(mw, mh, rect_h):
    """Compute the fixed full-world camera in source float32 operation order.

    Widening intermediate arithmetic changes rectangle edges before Qt samples them."""
    cx, cy = F(mw) * F(0.5), F(mh) * F(0.5)
    vis = F(max(mw, mh))
    raw_unit = F(F(64) / vis)
    unit = F(np.float64(raw_unit) * (rect_h / 64.0))
    view_dim = F(64.0 / np.float64(raw_unit))
    x_off = F(unit * F(cx - F(view_dim / F(2))))
    y_off = F(unit * F(cy - F(view_dim / F(2))))
    return unit, view_dim, x_off, y_off


def _screen_rect(x, y, dx, dy, g, eps=0.0):
    """Map world rectangles through (unit, view_dim, x_off, y_off) in float32."""
    unit, view_dim, x_off, y_off = g
    e = F(eps)
    return (
        float(F(F(F(x) - e) * unit - x_off)),
        float(F(F(F(view_dim) - F(y) - e) * unit + y_off)),
        float(F(F(F(dx) + F(2) * e) * unit)),
        float(F(F(F(dy) + F(2) * e) * unit)),
    )


def _premultiply(rgba: np.ndarray) -> np.ndarray:
    """Convert RGBA to Qt’s integer premultiplied representation."""
    a = rgba[..., 3].astype(np.uint32)
    out = rgba.copy()
    for c in range(3):
        t = rgba[..., c].astype(np.uint32) * a
        out[..., c] = (((t + (t >> 8) + 0x80) >> 8) & 0xFF).astype(np.uint8)
    return out


def _over(dst: torch.Tensor, src: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Qt SourceOver on premultiplied integer RGB.

    Keep the reciprocal-rounding BYTE_MUL expression; integer division by 255
    changes pixels. Zero source and alpha preserve the destination exactly."""
    t = dst * (255 - a)
    return (src + (((t + (t >> 8) + 0x80) >> 8) & 0xFF)).clamp_(0, 255)


def _qt_sincos(deg: np.ndarray):
    """Evaluate Qt’s exact quadrant branches and remaining trig on the host in float64.

    Keep the asymmetric ±180° cases. Device trig can round differently before
    fixed-point truncation, changing the sampled texel."""
    deg = np.asarray(deg, np.float64)
    rad = deg * DEG2RAD
    sina = np.where(
        deg == 90.0,
        1.0,
        np.where(
            deg == -270.0,
            1.0,
            np.where((deg == 270.0) | (deg == -90.0), -1.0, np.where(deg == 180.0, 0.0, np.sin(rad))),
        ),
    )
    cosa = np.where(
        (deg == 90.0) | (deg == -270.0) | (deg == 270.0) | (deg == -90.0),
        0.0,
        np.where(deg == 180.0, -1.0, np.cos(rad)),
    )
    return sina, cosa


class _EntSlot:
    """Entity sprite geometry; position, visibility and optional sprite choice come from rows."""

    def __init__(
        self, key, sprite, rx, ry, z, rotated=False, reflect=False, abs_coords=False, fixed_xy=None, rotation=None
    ):
        self.key, self.sprite = key, sprite
        self.rx, self.ry, self.z = F(rx), F(ry), z
        self.rotated, self.reflect, self.abs_coords = rotated, reflect, abs_coords
        self.fixed_xy, self.rotation = fixed_xy, rotation


class _Spec:
    """Per-game render description, independent of simulation.

    ``fills`` entries are (class_index, RGB, cell_fraction), for solid grid
    sub-rectangles such as Chaser orbs rather than sprite gathers."""

    def __init__(self, name, dim, classes, sprites, backgrounds, ent_slots, grid_slot, fills=()):
        self.name, self.dim, self.classes = name, dim, classes
        self.sprites, self.backgrounds, self.ent_slots = sprites, backgrounds, ent_slots
        self.grid_slot = np.asarray(grid_slot, np.int64)
        self.fills = tuple(fills)


def _aspect_ry(rx, w, h):
    """Match sprite aspect ratio with both division steps rounded to float32."""
    return float(F(F(rx) / F(w * 1.0 / h)))


# Chaser has one background, so its index is constant and omitted from render rows.
TOPDOWN_SIMPLE_BG = ["topdown_backgrounds/floortiles.png"]
# Non-square source backgrounds require a horizontal crop percentage.
WATER_BG = [
    f"water_backgrounds/{n}"
    for n in [
        "water1.png",
        "water2.png",
        "water3.png",
        "water4.png",
        "underwater1.png",
        "underwater2.png",
        "underwater3.png",
    ]
]
TOPDOWN_BG = [
    f"topdown_backgrounds/{n}"
    for n in [
        "floortiles.png",
        "backgrounddetailed1.png",
        "backgrounddetailed2.png",
        "backgrounddetailed3.png",
        "backgrounddetailed4.png",
        "backgrounddetailed5.png",
        "backgrounddetailed6.png",
        "backgrounddetailed7.png",
        "backgrounddetailed8.png",
    ]
]
#: `platform_backgrounds` in `resources.cpp` order, then every space background appended -- which is
#: what `images_load` does at the bottom of the file, and what makes miner's index reach 61.
PLATFORM_BG = [
    "platform_backgrounds/alien_bg.png",
    "platform_backgrounds/another_world_bg.png",
    "platform_backgrounds/back_cave.png",
    "platform_backgrounds/caverns.png",
    "platform_backgrounds/cyberpunk_bg.png",
    "platform_backgrounds/parallax_forest.png",
    "platform_backgrounds/scifi_bg.png",
    "platform_backgrounds/scifi2_bg.png",
    "platform_backgrounds/living_tissue_bg.png",
    "platform_backgrounds/airadventurelevel1.png",
    "platform_backgrounds/airadventurelevel2.png",
    "platform_backgrounds/airadventurelevel3.png",
    "platform_backgrounds/airadventurelevel4.png",
    "platform_backgrounds/cave_background.png",
    "platform_backgrounds/blue_desert.png",
    "platform_backgrounds/blue_grass.png",
    "platform_backgrounds/blue_land.png",
    "platform_backgrounds/blue_shroom.png",
    "platform_backgrounds/colored_desert.png",
    "platform_backgrounds/colored_grass.png",
    "platform_backgrounds/colored_land.png",
    "platform_backgrounds/colored_shroom.png",
    "platform_backgrounds/landscape1.png",
    "platform_backgrounds/landscape2.png",
    "platform_backgrounds/landscape3.png",
    "platform_backgrounds/landscape4.png",
    "platform_backgrounds/battleback1.png",
    "platform_backgrounds/battleback2.png",
    "platform_backgrounds/battleback3.png",
    "platform_backgrounds/battleback4.png",
    "platform_backgrounds/battleback5.png",
    "platform_backgrounds/battleback6.png",
    "platform_backgrounds/battleback7.png",
    "platform_backgrounds/battleback8.png",
    "platform_backgrounds/battleback9.png",
    "platform_backgrounds/battleback10.png",
    "platform_backgrounds/sunrise.png",
    "platform_backgrounds_2/beach1.png",
    "platform_backgrounds_2/beach2.png",
    "platform_backgrounds_2/beach3.png",
    "platform_backgrounds_2/beach4.png",
    "platform_backgrounds_2/fantasy1.png",
    "platform_backgrounds_2/fantasy2.png",
    "platform_backgrounds_2/fantasy3.png",
    "platform_backgrounds_2/fantasy4.png",
    "platform_backgrounds_2/candy1.png",
    "platform_backgrounds_2/candy2.png",
    "platform_backgrounds_2/candy3.png",
    "platform_backgrounds_2/candy4.png",
    "space_backgrounds/deep_space_01.png",
    "space_backgrounds/spacegen_01.png",
    "space_backgrounds/milky_way_01.png",
    "space_backgrounds/ez_space_lite_01.png",
    "space_backgrounds/meyespace_v1_01.png",
    "space_backgrounds/eye_nebula_01.png",
    "space_backgrounds/deep_sky_01.png",
    "space_backgrounds/space_nebula_01.png",
    "space_backgrounds/Background-1.png",
    "space_backgrounds/Background-2.png",
    "space_backgrounds/Background-3.png",
    "space_backgrounds/Background-4.png",
    "space_backgrounds/parallax-space-backgound.png",
]

_KEY_RY, _RING_RY, _EXIT_RY = _aspect_ry(0.375, 84, 52), _aspect_ry(0.03, 84, 52), _aspect_ry(0.375, 70, 48)
#: `add_entity(1 - r*(2i + 1.25), r*.75, ...)` with `ring_key_r = 0.03f`, evaluated in float32
_RING_XY = [(float(F(F(1) - F(0.03) * F(2 * i + 1.25))), float(F(F(0.03) * F(0.75)))) for i in range(3)]
_RING_ROT = float(PI_F / F(2))

SPECS = {
    "maze": _Spec(
        "maze",
        25,
        ("space", "wall", "goal"),
        ["kenney/Ground/Sand/sandCenter.png", "misc_assets/cheese.png", "kenney/Enemies/mouse_move.png"],
        TOPDOWN_BG,
        [_EntSlot("agent", 2, 0.5, 0.5, z=1, reflect=True)],
        [-1, 0, 1],
    ),
    "heist": _Spec(
        "heist",
        13,
        ("space", "wall"),
        [
            "kenney/Ground/Dirt/dirtCenter.png",
            "misc_assets/gemYellow.png",
            "misc_assets/spaceAstronauts_008.png",
            "misc_assets/keyBlue.png",
            "misc_assets/keyGreen.png",
            "misc_assets/keyRed.png",
            "misc_assets/lock_blue.png",
            "misc_assets/lock_green.png",
            "misc_assets/lock_red.png",
        ],
        TOPDOWN_BG,
        [_EntSlot(f"lock{i}", 6 + i, 0.5, 0.5, z=0) for i in range(3)]
        + [_EntSlot(f"key{i}", 3 + i, 0.375, _KEY_RY, z=0) for i in range(3)]
        + [_EntSlot("exit", 1, 0.375, _EXIT_RY, z=0), _EntSlot("agent", 2, 0.375, 0.375, z=1, rotated=True)]
        + [
            _EntSlot(
                f"ring{i}",
                3 + i,
                0.03,
                _RING_RY,
                z=1,
                rotated=True,
                abs_coords=True,
                fixed_xy=_RING_XY[i],
                rotation=_RING_ROT,
            )
            for i in range(3)
        ],
        [-1, 0],
    ),
    "miner": _Spec(
        "miner",
        20,
        ("space", "boulder", "diamond", "dirt"),
        [
            "misc_assets/robot_greenDrive1.png",
            "misc_assets/elementStone007.png",
            "misc_assets/gemBlue.png",
            "misc_assets/window.png",
            "misc_assets/dirt.png",
        ],
        PLATFORM_BG,
        [_EntSlot("exit", 3, 0.5, 0.5, z=-1), _EntSlot("agent", 0, 0.5, 0.5, z=1, reflect=True)],
        [-1, 1, 2, 4],
    ),
    # Hard-mode Chaser has three slots per entity type; grid orbs use solid fills.
    "chaser": _Spec(
        "chaser",
        13,
        ("space", "wall", "orb"),
        [
            "misc_assets/enemyFloating_1b.png",  # 0  PLAYER
            "misc_assets/yellowCrystal.png",  # 1  LARGE_ORB
            "misc_assets/enemyWalking_1b.png",  # 2  ENEMY_WEAK   (48x48)
            "misc_assets/enemySpikey_1b.png",  # 3  ENEMY_EGG
            "misc_assets/tileStone_slope.png",  # 4  MAZE_WALL
            "misc_assets/enemyFlying_1.png",  # 5  ENEMY        (64x64)
            "misc_assets/enemyFlying_2.png",  # 6  ENEMY2       (64x64)
            "misc_assets/enemyFlying_3.png",
        ],  # 7  ENEMY3       (64x64)
        TOPDOWN_SIMPLE_BG,
        [_EntSlot(f"orb{i}", 1, 0.4, 0.4, z=0) for i in range(3)]
        + [_EntSlot(f"egg{i}", 3, 0.5, 0.5, z=0) for i in range(3)]
        + [_EntSlot(f"enemy{i}", 5, 0.5, 0.5, z=0) for i in range(3)]
        + [_EntSlot("agent", 0, 0.5, 0.5, z=1)],
        [-1, 4, -1],
        fills=[(2, (0, 255, 0), 0.3)],
    ),
}
# Chaser’s weak/time-dependent enemy images map to four shared sprite choices.
_CHASER_ENEMY_SPRITE = (2, 5, 6, 7)  # atlas slots for image ids 3, 6, 7, 8
#: procgen grid object id -> render class. miner folds MOVING_BOULDER/MOVING_DIAMOND into their
#: stationary forms because `image_for_type` gives them the same sprite.
_CLASS_OF = {
    "maze": {P_SPACE: 0, P_WALL: 1, 2: 2},
    "heist": {P_SPACE: 0, P_WALL: 1},
    "miner": {P_SPACE: 0, 1: 1, 3: 1, 2: 2, 4: 2, 9: 3},
    # MARKER (1001) is generation scratch cleared before source frames are observed.
    "chaser": {P_SPACE: 0, 5: 1, 1002: 2},
}
_PROCGEN_ID = {
    "maze": [P_SPACE, P_WALL, 2],
    "heist": [P_SPACE, P_WALL],
    "miner": [P_SPACE, 1, 2, 9],
    "chaser": [P_SPACE, 5, 1002],
}


def _cache_dir() -> str:
    root = os.environ.get("STABLE_ENVS_CACHE") or os.path.expanduser("~/.cache/stable-envs")
    d = os.path.join(root, "procgen")
    os.makedirs(d, exist_ok=True)
    return d


class _Assets:
    """Cache premultiplied sprites and row-resampled background atlases per game/device."""

    _atlas: dict = {}
    _bgatlas: dict = {}

    @classmethod
    def atlas(cls, game: str, device):
        key = (game, str(device))
        if key in cls._atlas:
            return cls._atlas[key]
        path = os.path.join(_cache_dir(), f"sprites_{game}.npz")
        packed = wh = None
        if os.path.exists(path):
            try:
                z = np.load(path)
                packed, wh = z["packed"], z["wh"]
            except Exception:  # noqa: BLE001 partial/corrupt cache
                packed = None
        if packed is None:
            from PIL import Image

            imgs = [
                _premultiply(np.asarray(Image.open(os.path.join(ASSET_ROOT, p)).convert("RGBA"), np.uint8).copy())
                for p in SPECS[game].sprites
            ]
            wh = np.array([[im.shape[1], im.shape[0]] for im in imgs], np.int64)
            H, W = int(wh[:, 1].max()), int(wh[:, 0].max())
            packed = np.zeros((len(imgs), H, W, 4), np.uint8)
            for i, im in enumerate(imgs):
                packed[i, : im.shape[0], : im.shape[1]] = im
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, packed=packed, wh=wh)
            os.replace(tmp, path)
        t = torch.from_numpy(np.ascontiguousarray(packed)).to(device)
        # the flat (N,4) view the kernels index is cached too: rebuilding it per render call is a
        # copy of the whole atlas on the critical path, for no reason
        out = (t, np.asarray(wh), t.reshape(-1, 4).contiguous())
        cls._atlas[key] = out
        return out

    @classmethod
    def bg_atlas(cls, game: str, size: int, g, d: int, device):
        """Pack all backgrounds at the requested row resolution, padded to the widest image.

        This leaves one per-row column map and gather when drawing mixed backgrounds."""
        key = (game, int(size), str(device))
        if key in cls._bgatlas:
            return cls._bgatlas[key]
        path = os.path.join(_cache_dir(), f"bg_{game}_{size}.npz")
        tab = bw = bh = None
        if os.path.exists(path):
            try:
                z = np.load(path)
                tab, bw, bh = z["tab"], z["bw"], z["bh"]
            except Exception:  # noqa: BLE001 partial/corrupt cache
                tab = None
        if tab is None:
            from PIL import Image

            mr = _screen_rect(0, d, d, d, g)
            rows_img, bw, bh = [], [], []
            for p in SPECS[game].backgrounds:
                a = np.asarray(Image.open(os.path.join(ASSET_ROOT, p)).convert("RGB"), np.uint8)
                r, _, _ = _blit_axis(mr[1], mr[3], a.shape[0], size)
                rows_img.append(a[r])
                bw.append(a.shape[1])
                bh.append(a.shape[0])
            bw, bh = np.asarray(bw, np.int64), np.asarray(bh, np.int64)
            tab = np.zeros((len(rows_img), size, int(bw.max()), 3), np.uint8)
            for i, im in enumerate(rows_img):
                tab[i, :, : im.shape[1]] = im
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, tab=tab, bw=bw, bh=bh)
            os.replace(tmp, path)
        out = (
            torch.from_numpy(np.ascontiguousarray(tab).reshape(-1, 3)).to(device),
            int(tab.shape[2]),
            torch.from_numpy(np.asarray(bw)).to(device),
            torch.from_numpy(np.asarray(bh)).to(device),
        )
        cls._bgatlas[key] = out
        return out


def _blit_axis(pos, extent, src_extent, size):
    """Map one device axis to source indices with Qt’s 16.16 walk.

    Clamp unused entries for safe gathers; callers mask pixels outside [lo, hi)."""
    l = int(_qround(pos))
    w = int(_qround(pos + extent)) - l
    out = np.zeros(size, np.int64)
    lo, hi = max(l, 0), min(l + w, size)
    if hi <= lo:
        return out, lo, hi
    sf = float(src_extent) / float(extent)
    step = int(0x00010000 * sf)
    base = (int(np.ceil((lo + 0.5 - pos) * sf * 65536)) - 1) & 0xFFFFFFFF
    idx = np.arange(hi - lo, dtype=np.int64)
    out[lo:hi] = np.clip(((base + idx * step) & 0xFFFFFFFF) >> 16, 0, src_extent - 1)
    return out, lo, hi


def _rot_matrix(rect, deg, sw, sh):
    """Build Qt’s source-to-device transform for a rotated image rectangle."""
    rx, ry, rw, rh = rect
    sina, cosa = (float(np.asarray(v).reshape(-1)[0]) for v in _qt_sincos(np.float64(deg)))
    dx = rx + rw / 2.0 - rw / 2.0 * cosa + rh / 2.0 * sina
    dy = ry + rh / 2.0 - rw / 2.0 * sina - rh / 2.0 * cosa
    M = np.array([[cosa * (rw / sw), sina * (rw / sw)], [-sina * (rh / sh), cosa * (rh / sh)]], np.float64)
    return M, np.array([dx, dy], np.float64)


def _rot_index_np(rect, deg, sw, sh, size):
    """Precompute pixel-centre coverage and texel indices for a fixed rotated blit."""
    M, T = _rot_matrix(rect, deg, sw, sh)
    corners = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float64) @ M + T
    ys, xs = np.mgrid[0:size, 0:size]
    px, py = xs + 0.5, ys + 0.5
    inside = np.ones((size, size), bool)
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        inside &= ((b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])) >= 0
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    inv = np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]]) / det
    iT = -T @ inv
    fx = ((inv[1, 0] * py + inv[0, 0] * px + iT[0]) * 65536).astype(np.int64) >> 16
    fy = ((inv[1, 1] * py + inv[0, 1] * px + iT[1]) * 65536).astype(np.int64) >> 16
    ok = inside & (fx >= 0) & (fx < sw) & (fy >= 0) & (fy < sh)
    return ok, np.clip(fx, 0, sw - 1), np.clip(fy, 0, sh - 1)


class _Geom:
    """Cache geometry maps for one game, size and device.

    Two cell overlaps per axis produce four ordered grid layers. Assert the overlap
    bound while building tables so another game cannot silently lose coverage."""

    _cache: dict = {}
    K = 2

    @classmethod
    def get(cls, game: str, size: int, device):
        key = (game, int(size), str(device))
        if key not in cls._cache:
            cls._cache[key] = _Geom(game, int(size), device)
        return cls._cache[key]

    def __init__(self, game: str, size: int, device):
        spec = SPECS[game]
        self.spec, self.size, self.device = spec, size, torch.device(device)
        d = spec.dim
        self.g = _prepare(d, d, size)
        wh = _Assets.atlas(game, device)[1]
        S, K = len(spec.sprites), self.K

        cx = -np.ones((size, K), np.int64)
        cy = -np.ones((size, K), np.int64)
        sxi = np.zeros((S, size, K), np.int64)
        syi = np.zeros((S, size, K), np.int64)
        ncol = np.zeros(size, np.int64)
        nrow = np.zeros(size, np.int64)
        for i in range(d):
            rx, ry, rw, rh = _screen_rect(i, i + 1, 1, 1, self.g, eps=RENDER_EPS)
            xa = [_blit_axis(rx, rw, int(wh[s, 0]), size) for s in range(S)]
            ya = [_blit_axis(ry, rh, int(wh[s, 1]), size) for s in range(S)]
            for c in range(xa[0][1], xa[0][2]):
                k = ncol[c]
                if k >= K:
                    raise AssertionError(
                        f"{game}@{size}: device column {c} is covered by more than "
                        f"{K} grid cells; the layer count is wrong"
                    )
                cx[c, k] = i
                for s in range(S):
                    sxi[s, c, k] = xa[s][0][c]
                ncol[c] += 1
            for r in range(ya[0][1], ya[0][2]):
                k = nrow[r]
                if k >= K:
                    raise AssertionError(
                        f"{game}@{size}: device row {r} is covered by more than {K} "
                        f"grid cells; the layer count is wrong"
                    )
                cy[r, k] = i
                for s in range(S):
                    syi[s, r, k] = ya[s][0][r]
                nrow[r] += 1
        # Cache flat atlas offsets per layer to avoid repeated broadcast indexing.
        _, Hs, Ws, _ = _Assets.atlas(game, device)[0].shape  # noqa: E501
        self.layers = []
        for kx in range(K):
            for ky in range(K):
                mask = (cx[:, kx] >= 0)[None, :] & (cy[:, ky] >= 0)[:, None]
                cell = (np.clip(cy[:, ky], 0, None)[:, None] * d + np.clip(cx[:, kx], 0, None)[None, :]).astype(
                    np.int32
                )
                idx = (
                    (np.arange(S)[:, None, None] * Hs + syi[:, :, ky][:, :, None]) * Ws + sxi[:, :, kx][:, None, :]
                ).astype(np.int32)
                self.layers.append(
                    (
                        torch.from_numpy(cell.reshape(-1)).to(device),
                        torch.from_numpy(mask.reshape(-1).astype(np.int8)).to(device),
                        torch.from_numpy(idx.reshape(-1)).to(device),
                    )
                )
        # Solid sub-rectangles use one coverage table; assert they never overlap.
        self.fills = []
        for cls_idx, rgb, frac in spec.fills:
            cellof = np.full(size * size, -1, np.int64)
            for i in range(d):
                for j in range(d):
                    rx, ry, rw, rh = _screen_rect(i, j + 1, 1, 1, self.g, eps=RENDER_EPS)
                    l, t, w, h = _to_rect(rx + rw * (1 - frac) / 2, ry + rh * (1 - frac) / 2, rw * frac, rh * frac)
                    x0, x1 = max(int(l), 0), min(int(l) + int(w), size)
                    y0, y1 = max(int(t), 0), min(int(t) + int(h), size)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    sub = cellof.reshape(size, size)[y0:y1, x0:x1]
                    if (sub >= 0).any():
                        raise AssertionError(
                            f"{game}@{size}: two grid fill sub-rects share a device pixel; the "
                            f"single-table fill layer is wrong for this game"
                        )
                    sub[:] = j * d + i
            self.fills.append(
                (
                    int(cls_idx),
                    torch.from_numpy(cellof.astype(np.int32)).to(device),
                    torch.tensor(rgb, dtype=torch.uint8, device=device),
                )
            )
        self.slot_lut = torch.from_numpy(spec.grid_slot.astype(np.int32)).to(device)
        # Keep sprite dimensions on device for per-row sprite choices.
        self.wh_dev = torch.from_numpy(np.ascontiguousarray(wh)).to(device)
        self.cols = torch.arange(size, device=device)
        self.rc = torch.arange(size * size, dtype=torch.int32, device=device)
        self.SS = size * size
        #: a valid float pointer for the kernels' unused `gain` argument when HAS_GAIN is False
        self.dummy_gain = torch.ones(1, 3, dtype=torch.float32, device=device)
        self._bg = None

        # entities whose rect is a constant of the size (heist's rotated, use_abs_coords ring keys)
        self.fixed = {}
        for si, sl in enumerate(spec.ent_slots):
            if sl.fixed_xy is None:
                continue
            m, fx, fy = _rot_index_np(
                self._abs_rect(sl),
                float(F(F(sl.rotation) * F(180)) / PI_F),
                int(wh[sl.sprite, 0]),
                int(wh[sl.sprite, 1]),
                size,
            )
            self.fixed[si] = (
                torch.from_numpy(np.ascontiguousarray(m)).to(device),
                torch.from_numpy(fx.astype(np.int32)).to(device),
                torch.from_numpy(fy.astype(np.int32)).to(device),
            )

    @property
    def bg(self):
        """Lazily return the flat uint8 background atlas, padded width and source dimensions."""
        if self._bg is None:
            self._bg = _Assets.bg_atlas(self.spec.name, self.size, self.g, self.spec.dim, self.device)
        return self._bg

    def _abs_rect(self, sl):
        """Compute a fixed screen-relative entity rectangle in source float32 order."""
        unit, view_dim = self.g[0], self.g[1]
        x, y = sl.fixed_xy
        vals = (
            F(view_dim * F(F(x) - sl.rx)),
            F(view_dim * F(F(y) + sl.ry)),
            F(F(2) * view_dim * sl.rx),
            F(F(2) * view_dim * sl.ry),
        )
        return tuple(float(F(v * unit)) for v in vals)


def _t_qround(v):
    return torch.where(v >= 0, torch.floor(v + 0.5), torch.ceil(v - 0.5))


def _axis_map(pos, extent, src_extent, S: int, cols):
    """Return (coverage, texel index) arrays of shape (B, S) for one blit axis.

    Position and extent are float64; source extent may be scalar or per row."""
    B = pos.shape[0]
    l = _t_qround(pos).long()
    w = _t_qround(pos + extent).long() - l
    lo = l.clamp_min(0)
    hi = torch.minimum(l + w, torch.full_like(l, S))
    se = (
        torch.as_tensor(src_extent, device=pos.device, dtype=torch.float64)
        if not isinstance(src_extent, int)
        else float(src_extent)
    )
    sf = se / extent
    step = torch.trunc(65536.0 * sf).long()
    base = (torch.ceil((lo.double() + 0.5 - pos) * sf * 65536.0).long() - 1) & 0xFFFFFFFF
    idx = ((base.view(B, 1) + (cols.view(1, S) - lo.view(B, 1)) * step.view(B, 1)) & 0xFFFFFFFF) >> 16
    ok = (cols.view(1, S) >= lo.view(B, 1)) & (cols.view(1, S) < hi.view(B, 1))
    if isinstance(src_extent, int):
        return ok, idx.clamp_(0, src_extent - 1)
    return ok, torch.minimum(idx.clamp_min_(0), torch.as_tensor(src_extent, device=pos.device).long().view(B, 1) - 1)


def _rot_maps(rx, ry, rw, rh, rot, sw, sh, S):
    """Compute batched pixel-centre coverage and fixed-point texels for rotated blits."""
    dev = rx.device
    B = rx.shape[0]
    deg = (rot.to(torch.float32) * F(180) / PI_F).double()
    sina_np, cosa_np = _qt_sincos(deg.detach().cpu().numpy())
    sina = torch.as_tensor(sina_np, dtype=torch.float64, device=dev)
    cosa = torch.as_tensor(cosa_np, dtype=torch.float64, device=dev)

    dx = rx + rw / 2 - rw / 2 * cosa + rh / 2 * sina
    dy = ry + rh / 2 - rw / 2 * sina - rh / 2 * cosa
    m11, m12 = cosa * (rw / sw), sina * (rw / sw)
    m21, m22 = -sina * (rh / sh), cosa * (rh / sh)

    cxs = torch.stack([dx, sw * m11 + dx, sw * m11 + sh * m21 + dx, sh * m21 + dx], 1)
    cys = torch.stack([dy, sw * m12 + dy, sw * m12 + sh * m22 + dy, sh * m22 + dy], 1)
    ys = torch.arange(S, device=dev, dtype=torch.float64).view(1, S, 1) + 0.5
    xs = torch.arange(S, device=dev, dtype=torch.float64).view(1, 1, S) + 0.5
    inside = torch.ones(B, S, S, dtype=torch.bool, device=dev)
    for i in range(4):
        j = (i + 1) % 4
        ex = (cxs[:, j] - cxs[:, i]).view(B, 1, 1)
        ey = (cys[:, j] - cys[:, i]).view(B, 1, 1)
        inside &= (ex * (ys - cys[:, i].view(B, 1, 1)) - ey * (xs - cxs[:, i].view(B, 1, 1))) >= 0

    det = m11 * m22 - m12 * m21
    i00, i01, i10, i11 = m22 / det, -m12 / det, -m21 / det, m11 / det
    it0 = -(dx * i00 + dy * i10)
    it1 = -(dx * i01 + dy * i11)
    fx = ((i10.view(B, 1, 1) * ys + i00.view(B, 1, 1) * xs + it0.view(B, 1, 1)) * 65536).to(torch.int64) >> 16
    fy = ((i11.view(B, 1, 1) * ys + i01.view(B, 1, 1) * xs + it1.view(B, 1, 1)) * 65536).to(torch.int64) >> 16
    ok = inside & (fx >= 0) & (fx < sw) & (fy >= 0) & (fy < sh)
    return ok, fx.clamp_(0, sw - 1).int(), fy.clamp_(0, sh - 1).int()


# Fuse each blend into one canvas read/write, preserving integer SourceOver order.

if HAVE_TRITON:

    @triton.jit
    def _blend_kernel(
        canvas, atlas, idx, mask, gain, n, SS: tl.constexpr, HAS_GAIN: tl.constexpr, BLOCK: tl.constexpr
    ):
        """Blend one primitive from per-pixel atlas indices and coverage."""
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = off < n
        m = tl.load(mask + off, mask=ok, other=0).to(tl.int32)
        i = tl.load(idx + off, mask=ok, other=0).to(tl.int32)
        a = tl.load(atlas + i * 4 + 3, mask=ok, other=0).to(tl.int32) * m
        b = off // SS
        for c in tl.static_range(3):
            s = tl.load(atlas + i * 4 + c, mask=ok, other=0).to(tl.int32) * m
            if HAS_GAIN:
                g = tl.load(gain + b * 3 + c, mask=ok, other=0.0)
                s = tl.minimum((s.to(tl.float32) * g).to(tl.int32), 255)
            d = tl.load(canvas + off * 3 + c, mask=ok, other=0).to(tl.int32)
            t = d * (255 - a)
            t = (t + (t >> 8) + 128) >> 8
            tl.store(canvas + off * 3 + c, tl.minimum(s + (t & 255), 255).to(tl.uint8), mask=ok)

    @triton.jit
    def _grid_kernel(
        canvas,
        atlas,
        grid,
        cell_tab,
        mask_tab,
        idx_tab,
        slot_lut,
        gain,
        n,
        NCELL: tl.constexpr,
        SS: tl.constexpr,
        HAS_GAIN: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Gather and blend one of the four ordered grid layers."""
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = off < n
        b = off // SS
        p = off % SS
        cell = tl.load(cell_tab + p, mask=ok, other=0).to(tl.int32)
        m = tl.load(mask_tab + p, mask=ok, other=0).to(tl.int32)
        cls = tl.load(grid + b * NCELL + cell, mask=ok, other=0).to(tl.int32)
        slot = tl.load(slot_lut + cls, mask=ok, other=0).to(tl.int32)
        m = m * (slot >= 0).to(tl.int32)
        i = tl.load(idx_tab + tl.maximum(slot, 0) * SS + p, mask=ok, other=0).to(tl.int32)
        a = tl.load(atlas + i * 4 + 3, mask=ok, other=0).to(tl.int32) * m
        for c in tl.static_range(3):
            s = tl.load(atlas + i * 4 + c, mask=ok, other=0).to(tl.int32) * m
            if HAS_GAIN:
                g = tl.load(gain + b * 3 + c, mask=ok, other=0.0)
                s = tl.minimum((s.to(tl.float32) * g).to(tl.int32), 255)
            d = tl.load(canvas + off * 3 + c, mask=ok, other=0).to(tl.int32)
            t = d * (255 - a)
            t = (t + (t >> 8) + 128) >> 8
            tl.store(canvas + off * 3 + c, tl.minimum(s + (t & 255), 255).to(tl.uint8), mask=ok)


if HAVE_TRITON:

    @triton.jit
    def _fill_kernel(
        canvas, grid, cell_tab, colour, cls, n, NCELL: tl.constexpr, SS: tl.constexpr, BLOCK: tl.constexpr
    ):
        """Write opaque grid fill colours where the selected class covers the pixel."""
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = off < n
        b = off // SS
        p = off % SS
        cell = tl.load(cell_tab + p, mask=ok, other=-1).to(tl.int32)
        hit = cell >= 0
        g = tl.load(grid + b * NCELL + tl.maximum(cell, 0), mask=ok & hit, other=-1).to(tl.int32)
        hit = hit & (g == cls)
        for c in tl.static_range(3):
            v = tl.load(colour + c).to(tl.int32)
            d = tl.load(canvas + off * 3 + c, mask=ok, other=0).to(tl.int32)
            tl.store(canvas + off * 3 + c, tl.where(hit, v, d).to(tl.uint8), mask=ok)


def _blend_torch(canvas, atlas, idx, mask, gain):
    """Torch reference for fused premultiplied blending; mutates the uint8 canvas."""
    B, S = canvas.shape[0], canvas.shape[1]
    src = atlas.index_select(0, idx.reshape(-1)).view(B, S, S, 4).to(torch.int32)
    mm = mask.unsqueeze(-1).to(torch.int32)
    rgb = src[..., :3] * mm
    if gain is not None:
        rgb = (rgb.float() * gain.view(B, 1, 1, 3)).trunc_().clamp_(0, 255).to(torch.int32)
    canvas.copy_(_over(canvas.to(torch.int32), rgb, src[..., 3:4] * mm).to(torch.uint8))


class _ProcgenEnv(Env):
    """Render Maze, Heist, Miner and Chaser in hard mode with a fixed full-world camera.

    Rows encode grid classes and visible entity fields, not resumable game State.
    Miner folds moving/stationary objects that share sprites. Assets come from the
    installed Procgen wheel; local caches hold packed atlases and sampled rows.
    Scrolling cameras, nondefault source assets and other games are unsupported."""

    family = "procgen"

    game: str = ""
    default_size = 64
    parity_interior_min = 1.0
    # Chaser’s single background omits this index; other games store it.
    n_bg_state = 1
    # Warmup must reach dug cells, collected keys and Chaser’s hatched enemies.
    pool_warmup = 48

    def __init__(self, num_envs: int = 1, size=None, supersample=1, device="cuda", backend="auto", seed=0):
        super().__init__(
            num_envs=num_envs, size=size, supersample=supersample, device=device, backend=backend, seed=seed
        )
        # Separate grid and entity render columns; neither contains full simulation state.
        n = int(self.n_cells)
        self.state_layout = (((("grid",), n, 0),) if n else ()) + (
            ((("entities",), int(self.state_dim) - n, n),) if int(self.state_dim) > n else ()
        )
        self._pool = None

    def _close(self) -> None:
        """Release the instance’s sampled rows; shared assets and geometry remain cached."""
        self._pool = None

    @property
    def spec(self) -> _Spec:
        return SPECS[self.game]

    # Disable grid columns only for games whose grid never reaches a pixel.
    has_grid = True

    @property
    def n_cells(self) -> int:
        return self.spec.dim**2 if self.has_grid else 0

    def make_variation_space(self) -> Dict:
        """Expose background visibility and per-channel sprite/background gains.

        Gains multiply premultiplied colours before truncation; one is the identity.
        They do not promise a literal output RGB colour."""
        return Dict(
            {
                "rendering": Dict({"use_backgrounds": Discrete(2, init_value=1)}),
                "sprite_gain": Box(low=0.0, high=2.0, shape=(3,), dtype=np.float32, init_value=[1.0, 1.0, 1.0]),
                "background_gain": Box(low=0.0, high=2.0, shape=(3,), dtype=np.float32, init_value=[1.0, 1.0, 1.0]),
            }
        )

    def _unpack(self, s: torch.Tensor) -> dict:
        d = self.spec.dim
        n = self.n_cells
        bg = (
            s[:, n].round().long().clamp_(0, len(self.spec.backgrounds) - 1)
            if self.n_bg_state
            else torch.zeros(s.shape[0], dtype=torch.long, device=s.device)
        )
        grid = (
            s[:, :n].round().clamp_(0, len(self.spec.classes) - 1).to(torch.int32).view(-1, d, d)
            if self.has_grid
            else None
        )
        return {"grid": grid, "bg_index": bg}

    def _slot_fields(self, f, sl, B):
        """Return (present, x, y, rotation, reflected), with None for unused optional fields."""
        raise NotImplementedError

    def _slot_sprite(self, f, sl, B):
        """Return a per-row sprite index, or None for the slot’s fixed sprite.

        Sprite choices may have different source dimensions, as in Chaser enemies."""
        return None

    @torch.no_grad()
    def _render(self, states=None, size=None, supersample=None, variation=None, style=None, out=None) -> torch.Tensor:
        """Composite nearest-neighbour sprite blits in source draw order into uint8 frames."""
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        B = s.shape[0]
        S = int(size or self.size)
        if supersample not in (None, 1):
            raise ValueError(
                f"{type(self).__name__}.render: supersample={supersample} is meaningless "
                f"— the frame is a nearest-neighbour gather from a bitmap atlas, so "
                f"there is no sub-pixel coverage to estimate."
            )
        gm = _Geom.get(self.game, S, self.device)
        atlas, wh, flat = _Assets.atlas(self.game, self.device)
        sty = self.resolve_style(style, variation, batch=B)
        f = self._unpack(s)

        direct = out is not None and torch.is_tensor(out) and out.dtype == torch.uint8 and out.device == self.device
        # no zero_(): `_draw_background` writes every pixel (the black `fillRect` is `use_bg == 0`)
        canvas = out.view(B, S, S, 3) if direct else torch.empty(B, S, S, 3, dtype=torch.uint8, device=self.device)
        use_bg = sty["rendering.use_backgrounds"].reshape(B) > 0.5
        # Kernels address gain as b*3+c; style column slices have wider row strides
        # and must be copied to contiguous storage first.
        bgain = sty["background_gain"].reshape(B, 3).to(torch.float32).contiguous()
        sgain = sty["sprite_gain"].reshape(B, 3).to(torch.float32).contiguous()
        if bool((sgain == 1).all()):
            sgain = None
        self._draw_background(canvas, f, gm, use_bg, bgain)
        for z in (-1, 0):
            self._draw_entities(canvas, f, gm, flat, atlas.shape, wh, z, sgain)
            if z == -1 and self.has_grid:
                self._draw_grid(canvas, f, gm, flat, atlas.shape, sgain)
        self._draw_entities(canvas, f, gm, flat, atlas.shape, wh, 1, sgain)
        if out is not None and not direct:
            out.view(B, S, S, 3).copy_(canvas)
            canvas = out.view(B, S, S, 3)
        return canvas.view(*lead, S, S, 3) if lead else canvas

    def _draw_background(self, canvas, f, gm, use_bg, gain):
        """Draw the level background through a per-row column map; hidden backgrounds are black."""
        B, S = canvas.shape[0], canvas.shape[1]
        d = self.spec.dim
        mr = _screen_rect(0, d, d, d, gm.g)
        idx = f["bg_index"]
        flat, Wmax, bw, bh = gm.bg
        bwi, bhi = bw[idx], bh[idx]  # (B,) int64
        bg_ar = bwi.float() / bhi.float()
        world_ar = F(d * 1.0 / d)
        pct = f.get("bg_pct_x")
        p32 = torch.zeros(B, device=canvas.device) if pct is None else pct.float()
        off = (p32 * (bg_ar - float(world_ar))).double()
        x = mr[0] + mr[2] * (-off)
        w = mr[2] * (bg_ar / float(world_ar)).double()
        _, cols = _axis_map(x, w, bwi, S, gm.cols)
        lin = ((idx.view(B, 1, 1) * S + gm.cols.view(1, S, 1)) * Wmax + cols.view(B, 1, S)).to(torch.int32)
        px = flat.index_select(0, lin.reshape(-1)).view(B, S, S, 3)
        if not bool((gain == 1).all()):
            px = (px.float() * gain.view(B, 1, 1, 3)).trunc_().clamp_(0, 255).to(torch.uint8)
        canvas.copy_(px * use_bg.view(B, 1, 1, 1))

    def _draw_grid(self, canvas, f, gm, flat, shape, gain):
        B, S = canvas.shape[0], canvas.shape[1]
        grid = f["grid"].reshape(B, -1).contiguous()
        n = B * gm.SS
        for cell, mask, idx in gm.layers:
            if HAVE_TRITON:
                _grid_kernel[(triton.cdiv(n, 512),)](
                    canvas,
                    flat,
                    grid,
                    cell,
                    mask,
                    idx,
                    gm.slot_lut,
                    gain if gain is not None else gm.dummy_gain,
                    n,
                    NCELL=grid.shape[1],
                    SS=gm.SS,
                    HAS_GAIN=gain is not None,
                    BLOCK=512,
                    num_warps=4,
                )
                continue
            cls = grid.index_select(1, cell)
            slot = gm.slot_lut.index_select(0, cls.reshape(-1)).view(B, -1)
            m = (slot >= 0) & mask.reshape(1, -1).bool()  # noqa: E501  int8 table -> bool
            lin = idx.index_select(0, (slot.clamp_min(0) * gm.SS + gm.rc.view(1, -1)).reshape(-1))
            _blend_torch(canvas, flat, lin, m.view(B, S, S), gain)
        for cls_idx, cellof, colour in gm.fills:
            # Fill sub-rectangles stay inside their cells and cannot touch neighbouring sprites.
            if HAVE_TRITON:
                _fill_kernel[(triton.cdiv(n, 512),)](
                    canvas, grid, cellof, colour, cls_idx, n, NCELL=grid.shape[1], SS=gm.SS, BLOCK=512, num_warps=4
                )
                continue
            hit = cellof.view(1, -1) >= 0
            g = grid.gather(1, cellof.clamp_min(0).long().view(1, -1).expand(B, -1))
            hit = hit & (g == cls_idx)
            canvas.view(B, -1, 3).copy_(torch.where(hit.unsqueeze(-1), colour.view(1, 1, 3), canvas.view(B, -1, 3)))

    def _draw_entities(self, canvas, f, gm, flat, shape, wh, z, gain):
        B, S = canvas.shape[0], canvas.shape[1]
        Hs, Ws = int(shape[1]), int(shape[2])
        for si, sl in enumerate(self.spec.ent_slots):
            if sl.z != z:
                continue
            on, x, y, rot, refl = self._slot_fields(f, sl, B)
            spr = self._slot_sprite(f, sl, B)
            if spr is None:
                sw, sh = int(wh[sl.sprite, 0]), int(wh[sl.sprite, 1])
            else:
                sw, sh = gm.wh_dev[spr, 0], gm.wh_dev[spr, 1]
            if sl.fixed_xy is not None:
                m0, fx, fy = gm.fixed[si]
                m = m0.view(1, S, S).expand(B, S, S)
                sxg, syg = fx.view(1, S, S).expand(B, S, S), fy.view(1, S, S).expand(B, S, S)
            else:
                rect = self._ent_rect(sl, x, y, gm, self._slot_radii(f, sl, B))
                m, sxg, syg = self._blit_maps(rect, rot, sw, sh, S, gm)
            if on is not None:
                m = m & on.view(B, 1, 1)
            if refl is not None:
                w1 = (sw - 1) if isinstance(sw, int) else (sw - 1).view(B, 1, 1).int()
                sxg = torch.where(refl.view(B, 1, 1) > 0.5, w1 - sxg, sxg)
            base = sl.sprite if spr is None else spr.view(B, 1, 1).int()
            lin = ((base * Hs + syg) * Ws + sxg).contiguous()
            self._composite(canvas, flat, lin, m.contiguous(), gain, gm)

    def _composite(self, canvas, flat, lin, m, gain, gm):
        """Blend one primitive in place from atlas indices and coverage."""
        n = canvas.shape[0] * canvas.shape[1] * canvas.shape[2]
        if HAVE_TRITON:
            _blend_kernel[(triton.cdiv(n, 512),)](
                canvas,
                flat,
                lin,
                m.to(torch.int8),
                gain if gain is not None else gm.dummy_gain,
                n,
                SS=canvas.shape[1] * canvas.shape[2],
                HAS_GAIN=gain is not None,
                BLOCK=512,
                num_warps=4,
            )
        else:
            _blend_torch(canvas, flat, lin, m, gain)

    def _slot_radii(self, f, sl, B):
        """Optional per-row radii; supported games use constant slot geometry."""
        return None

    def _ent_rect(self, sl, x, y, gm, radii=None):
        """Compute entity rectangles in float32, widening only after world-to-screen arithmetic."""
        unit, view_dim, x_off, y_off = (float(v) for v in gm.g)
        x32, y32 = x.to(torch.float32), y.to(torch.float32)
        if radii is None:
            srx, sry = float(sl.rx), float(sl.ry)
        else:
            srx, sry = (v.to(torch.float32) for v in radii)
        rx = ((x32 - srx) * unit - x_off).double()
        ry = ((view_dim - (y32 + sry)) * unit + y_off).double()
        if radii is None:
            rw = torch.full_like(rx, float(F(F(F(2) * sl.rx) * F(unit))))
            rh = torch.full_like(rx, float(F(F(F(2) * sl.ry) * F(unit))))
        else:
            rw = ((2.0 * srx) * unit).double()
            rh = ((2.0 * sry) * unit).double()
        return rx, ry, rw, rh

    def _blit_maps(self, rect, rot, sw, sh, S, gm):
        """Choose axis-aligned or rotated sampling per row; zero rotation uses the axis path."""
        rx, ry, rw, rh = rect
        B = rx.shape[0]
        okx, sx = _axis_map(rx, rw, sw, S, gm.cols)
        oky, sy = _axis_map(ry, rh, sh, S, gm.cols)
        m = okx.view(B, 1, S) & oky.view(B, S, 1)
        sxg = sx.int().view(B, 1, S).expand(B, S, S)
        syg = sy.int().view(B, S, 1).expand(B, S, S)
        if rot is None:
            return m, sxg, syg
        turned = rot != 0
        if bool(turned.any()):
            mr, fx, fy = _rot_maps(rx, ry, rw, rh, rot, sw, sh, S)
            t = turned.view(B, 1, 1)
            m = torch.where(t, mr, m)
            sxg = torch.where(t, fx, sxg)
            syg = torch.where(t, fy, syg)
        return m, sxg, syg

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample compact render rows from cached source rollouts."""
        rng = rng or self.rng
        if self._pool is None:
            self._pool = self._load_pool()
        n = len(self._pool)
        idx = rng.choice(n, batch, replace=False) if batch <= n else rng.integers(0, n, batch)
        return self._pool[idx].copy()

    def _load_pool(self, n: int = 512) -> np.ndarray:
        path = os.path.join(_cache_dir(), f"pool_{self.game}_{self.state_dim}.npy")
        if os.path.exists(path):
            try:
                return np.load(path)
            except Exception:  # noqa: BLE001 partial write
                pass
        pool = self._build_pool(n)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, pool)
        os.replace(tmp, path)
        return pool

    def _build_pool(self, n: int) -> np.ndarray:
        """Harvest source rollout rows after warmup.

        Warmup exposes dug background cells, collected keys and hatched enemies so
        the sample pool exercises those render inputs."""
        from procgen import ProcgenGym3Env

        num = 16
        env = ProcgenGym3Env(
            num=num,
            env_name=self.game,
            distribution_mode="hard",
            rand_seed=0,
            start_level=0,
            num_levels=0,
            use_backgrounds=True,
        )
        rng = np.random.default_rng(0)
        rows = []
        for step in range(n // num + self.pool_warmup):
            env.act(rng.integers(0, 15, size=num).astype(np.int32))
            if step < self.pool_warmup:
                continue
            for blob in env.get_state():
                rows.append(self.state_from_blob(_parse(blob)))
            if len(rows) >= n:
                break
        return np.stack(rows[:n]).astype(np.float32)

    @classmethod
    def reference_env(cls, size: int = 64):
        if int(size) != 64:
            raise ValueError(f"{cls.__name__}: procgen's observation is 64x64 only; asked for {size}")
        from procgen import ProcgenGym3Env

        env = ProcgenGym3Env(
            num=1,
            env_name=cls.game,
            distribution_mode="hard",
            rand_seed=0,
            start_level=0,
            num_levels=0,
            use_backgrounds=True,
        )
        env.act(np.array([4], np.int32))
        return _Reference(env, _parse(env.get_state()[0]))

    @classmethod
    def reference_render(cls, ref, states) -> np.ndarray:
        """Restore each render row into a source blob and verify it before reading pixels."""
        out = []
        for st in np.asarray(states, np.float32):
            ref.env.set_state([_dump(cls.blob_from_state(ref.template, st))])
            back = cls.state_from_blob(_parse(ref.env.get_state()[0]))
            d = np.abs(back - st)
            if d.max() > 1e-6:
                i = int(np.argmax(d))
                raise AssertionError(
                    f"{cls.__name__}: the reference env did not reach the requested state — "
                    f"'{cls.state_names[i]}' came back {back[i]} for {st[i]} (max |diff| {d.max()})."
                )
            _, obs, _ = ref.env.observe()
            out.append(np.asarray(obs["rgb"][0]).copy())
        return np.stack(out)

    @classmethod
    def state_from_blob(cls, blob: dict) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def blob_from_state(cls, template: dict, st: np.ndarray) -> dict:
        raise NotImplementedError


class _Reference:
    """Own the source environment and a template blob for restoring render inputs."""

    def __init__(self, env, template):
        self.env, self.template = env, template


# Field order follows source Game/BasicAbstractGame/Entity/Grid/RandGen serialization.
# Keep integer and float widths and length-prefixed byte strings unchanged.

_HEAD = [
    ("serialize_version", "i"),
    ("game_name", "s"),
    ("paint_vel_info", "i"),
    ("use_generated_assets", "i"),
    ("use_monochrome_assets", "i"),
    ("restrict_themes", "i"),
    ("use_backgrounds", "i"),
    ("center_agent", "i"),
    ("debug_mode", "i"),
    ("distribution_mode", "i"),
    ("use_sequential_levels", "i"),
    ("use_easy_jump", "i"),
    ("plain_assets", "i"),
    ("physics_mode", "i"),
    ("grid_step", "i"),
    ("level_seed_low", "i"),
    ("level_seed_high", "i"),
    ("game_type", "i"),
    ("game_n", "i"),
    ("lsrg_seeded", "i"),
    ("lsrg", "s"),
    ("rg_seeded", "i"),
    ("rg", "s"),
    ("reward", "f"),
    ("done", "i"),
    ("level_complete", "i"),
    ("action", "i"),
    ("timeout", "i"),
    ("current_level_seed", "i"),
    ("prev_level_seed", "i"),
    ("episodes_remaining", "i"),
    ("episode_done", "i"),
    ("last_reward_timer", "i"),
    ("last_reward", "f"),
    ("default_action", "i"),
    ("fixed_asset_seed", "i"),
    ("cur_time", "i"),
    ("is_waiting_for_step", "i"),
]
_ENTITY = [
    ("x", "f"),
    ("y", "f"),
    ("vx", "f"),
    ("vy", "f"),
    ("rx", "f"),
    ("ry", "f"),
    ("type", "i"),
    ("image_type", "i"),
    ("image_theme", "i"),
    ("render_z", "i"),
    ("will_erase", "i"),
    ("collides_with_entities", "i"),
    ("collision_margin", "f"),
    ("rotation", "f"),
    ("vrot", "f"),
    ("is_reflected", "i"),
    ("fire_time", "i"),
    ("spawn_time", "i"),
    ("life_time", "i"),
    ("expire_time", "i"),
    ("use_abs_coords", "i"),
    ("friction", "f"),
    ("smart_step", "i"),
    ("avoids_collisions", "i"),
    ("auto_erase", "i"),
    ("alpha", "f"),
    ("health", "f"),
    ("theta", "f"),
    ("grow_rate", "f"),
    ("alpha_decay", "f"),
    ("climber_spawn_x", "f"),
]
_MID = [
    ("use_procgen_background", "i"),
    ("background_index", "i"),
    ("bg_tile_ratio", "f"),
    ("bg_pct_x", "f"),
    ("char_dim", "f"),
    ("last_move_action", "i"),
    ("move_action", "i"),
    ("special_action", "i"),
    ("mixrate", "f"),
    ("maxspeed", "f"),
    ("max_jump", "f"),
    ("action_vx", "f"),
    ("action_vy", "f"),
    ("action_vrot", "f"),
    ("center_x", "f"),
    ("center_y", "f"),
    ("random_agent_start", "i"),
    ("has_useful_vel_info", "i"),
    ("step_rand_int", "i"),
    ("arg_seeded", "i"),
    ("arg", "s"),
    ("main_width", "i"),
    ("main_height", "i"),
    ("out_of_bounds_object", "i"),
    ("unit", "f"),
    ("view_dim", "f"),
    ("x_off", "f"),
    ("y_off", "f"),
    ("visibility", "f"),
    ("min_visibility", "f"),
]
_TAIL = {
    "maze": [("maze_dim", "i"), ("world_dim", "i")],
    "heist": [("num_keys", "i"), ("world_dim", "i"), ("has_keys", "vb")],
    "miner": [("diamonds_remaining", "i")],
    # Read pathfinding vectors to reach eat_time at the correct serialized offset.
    "chaser": [
        ("free_cells", "vi"),
        ("is_space_vec", "vb"),
        ("eat_timeout", "i"),
        ("egg_timeout", "i"),
        ("eat_time", "i"),
        ("total_enemies", "i"),
        ("total_orbs", "i"),
        ("orbs_collected", "i"),
        ("maze_dim", "i"),
    ],
    # bigfish is NOT a shipped env (see pre-repository-cleanup:docs/research/pixel_native.md 2.9); its tail stays here
    # because the bigfish reference still drives through this parser.
    "bigfish": [("fish_eaten", "i"), ("r_inc", "f")],
}


def _parse(buf: bytes) -> dict:
    import struct

    o = [0]

    def i():
        v = struct.unpack_from("<i", buf, o[0])[0]
        o[0] += 4
        return v

    def fl():
        v = struct.unpack_from("<f", buf, o[0])[0]
        o[0] += 4
        return v

    def sb():
        n = i()
        v = buf[o[0] : o[0] + n]
        o[0] += n
        return v

    rd = {
        "i": i,
        "f": fl,
        "s": sb,
        "vb": lambda: [i() > 0 for _ in range(i())],
        "vi": lambda: [i() for _ in range(i())],
    }
    st = {k: rd[t]() for k, t in _HEAD}
    st["grid_size"] = i()
    st["entities"] = [{k: rd[t]() for k, t in _ENTITY} for _ in range(i())]
    st.update({k: rd[t]() for k, t in _MID})
    gw, gh = i(), i()
    st["grid_w"], st["grid_h"] = gw, gh
    st["grid"] = np.asarray([i() for _ in range(i())], np.int32).reshape(gh, gw)
    st["_tail"] = _TAIL[st["game_name"].decode()]
    st.update({k: rd[t]() for k, t in st["_tail"]})
    st["_rest"] = buf[o[0] :]  # vecgame.cpp's END_OF_BUFFER sentinel
    return st


def _dump(st: dict) -> bytes:
    import struct

    p = []

    def wi(v):
        p.append(struct.pack("<i", int(v)))

    def wf(v):
        p.append(struct.pack("<f", float(v)))

    def ws(v):
        wi(len(v))
        p.append(bytes(v))

    def wvb(v):
        wi(len(v))
        for x in v:
            wi(1 if x else 0)

    def wvi(v):
        wi(len(v))
        for x in v:
            wi(x)

    w = {"i": wi, "f": wf, "s": ws, "vb": wvb, "vi": wvi}
    for k, t in _HEAD:
        w[t](st[k])
    wi(st["grid_size"])
    wi(len(st["entities"]))
    for e in st["entities"]:
        for k, t in _ENTITY:
            w[t](e[k])
    for k, t in _MID:
        w[t](st[k])
    wi(st["grid_w"])
    wi(st["grid_h"])
    g = np.asarray(st["grid"]).reshape(-1)
    wi(g.size)
    for v in g.tolist():
        wi(v)
    for k, t in st["_tail"]:
        w[t](st[k])
    return b"".join(p) + st["_rest"]


def _make_entity(sl, x, y, theme, type_, image_type, rot=0.0, refl=0, agent=False) -> dict:
    e = {k: (0 if t == "i" else 0.0) for k, t in _ENTITY}
    e.update(
        alpha=1.0,
        alpha_decay=1.0,
        x=float(x),
        y=float(y),
        rx=float(sl.rx),
        ry=float(sl.ry),
        type=int(type_),
        image_type=int(image_type),
        image_theme=int(theme),
        render_z=int(sl.z),
        rotation=float(rot),
        is_reflected=int(refl),
        use_abs_coords=int(sl.abs_coords),
        smart_step=int(agent),
    )
    return e


def _grid_to_classes(game, grid):
    # sized by the largest id in the map, not by a round number: chaser's ORB is 1002, because
    # `draw_grid_obj` values live above `USE_ASSET_THRESHOLD` rather than in the sprite range.
    lut = np.zeros(max(_CLASS_OF[game]) + 1, np.float32)
    for k, v in _CLASS_OF[game].items():
        lut[k] = v
    return lut[np.asarray(grid, np.int64)]


def _classes_to_grid(game, cls, dim):
    ids = np.asarray(_PROCGEN_ID[game], np.int32)
    return ids[np.clip(np.rint(cls).astype(np.int64), 0, len(ids) - 1)].reshape(dim, dim)


@register
class ProcgenMaze(_ProcgenEnv):
    """Render the 25×25 maze with sand walls, cheese and a reflected mouse sprite.

    Square backgrounds make horizontal crop percentage irrelevant to pixels."""

    game = "maze"
    name = "procgen-maze"
    reference_id = "procgen-maze-v0"
    state_dim = 25 * 25 + 4
    state_names = tuple(f"cell_{y}_{x}" for y in range(25) for x in range(25)) + (
        "bg_index",
        "agent_x",
        "agent_y",
        "agent_reflected",
    )

    def _unpack(self, s):
        f = super()._unpack(s)
        n = self.n_cells
        f["agent"] = (s[:, n + 1], s[:, n + 2], s[:, n + 3])
        return f

    def _slot_fields(self, f, sl, B):
        x, y, r = f["agent"]
        return None, x, y, None, r

    @classmethod
    def state_from_blob(cls, b):
        a = [e for e in b["entities"] if e["type"] == 0][0]
        return np.concatenate(
            [
                _grid_to_classes("maze", b["grid"]).reshape(-1),
                [b["background_index"], a["x"], a["y"], a["is_reflected"]],
            ]
        ).astype(np.float32)

    @classmethod
    def blob_from_state(cls, template, st):
        b = dict(template)
        n = 625
        b["grid"] = _classes_to_grid("maze", st[:n], 25)
        b["grid_w"] = b["grid_h"] = 25
        b["grid_size"] = n
        b["background_index"] = int(round(float(st[n])))
        sl = SPECS["maze"].ent_slots[0]
        b["entities"] = [
            _make_entity(sl, st[n + 1], st[n + 2], 0, 0, 0, refl=int(round(float(st[n + 3]))), agent=True)
        ]
        return b


@register
class ProcgenHeist(_ProcgenEnv):
    """Render the 13×13 maze with locks, keys and a screen-relative key ring.

    Agent rotation and fixed quarter-turned ring keys use Qt’s transformed blit path."""

    game = "heist"
    name = "procgen-heist"
    reference_id = "procgen-heist-v0"
    state_dim = 13 * 13 + 1 + 3 + 2 + 9 + 9 + 3
    state_names = (
        tuple(f"cell_{y}_{x}" for y in range(13) for x in range(13))
        + ("bg_index", "agent_x", "agent_y", "agent_rotation", "exit_x", "exit_y")
        + tuple(f"{k}{i}_{c}" for k in ("lock", "key") for i in range(3) for c in ("x", "y", "present"))
        + ("has_key0", "has_key1", "has_key2")
    )
    _BASE = 13 * 13 + 6

    def _unpack(self, s):
        f = super()._unpack(s)
        n = self.n_cells
        b = self._BASE
        f["agent"] = (s[:, n + 1], s[:, n + 2], s[:, n + 3])
        f["exit"] = (s[:, n + 4], s[:, n + 5])
        f["lock"] = [(s[:, b + 3 * i], s[:, b + 3 * i + 1], s[:, b + 3 * i + 2]) for i in range(3)]
        f["key"] = [(s[:, b + 9 + 3 * i], s[:, b + 9 + 3 * i + 1], s[:, b + 9 + 3 * i + 2]) for i in range(3)]
        f["has_key"] = s[:, b + 18 : b + 21]
        return f

    def _slot_fields(self, f, sl, B):
        k = sl.key
        if k.startswith("lock") or k.startswith("key"):
            x, y, on = f["lock" if k.startswith("lock") else "key"][int(k[-1])]
            return on > 0.5, x, y, None, None
        if k.startswith("ring"):
            z = f["has_key"][:, 0] * 0
            return f["has_key"][:, int(k[-1])] > 0.5, z, z, None, None
        if k == "exit":
            return (None, *f["exit"], None, None)
        x, y, rot = f["agent"]
        return None, x, y, rot, None

    @classmethod
    def state_from_blob(cls, b):
        ents = b["entities"]
        a = [e for e in ents if e["type"] == 0][0]
        ex = [e for e in ents if e["type"] == 9]
        locks = {e["image_theme"]: e for e in ents if e["type"] == 1}
        keys = {e["image_theme"]: e for e in ents if e["type"] == 2}
        hk = list(b["has_keys"]) + [False] * 3
        row = [
            b["background_index"],
            a["x"],
            a["y"],
            a["rotation"],
            ex[0]["x"] if ex else 0.0,
            ex[0]["y"] if ex else 0.0,
        ]
        for tbl in (locks, keys):
            for i in range(3):
                e = tbl.get(i)
                row += [e["x"] if e else 0.0, e["y"] if e else 0.0, 1.0 if e else 0.0]
        row += [1.0 if hk[i] else 0.0 for i in range(3)]
        return np.concatenate([_grid_to_classes("heist", b["grid"]).reshape(-1), row]).astype(np.float32)

    @classmethod
    def blob_from_state(cls, template, st):
        b = dict(template)
        n, base = 169, cls._BASE
        b["grid"] = _classes_to_grid("heist", st[:n], 13)
        b["grid_w"] = b["grid_h"] = 13
        b["grid_size"] = n
        b["background_index"] = int(round(float(st[n])))
        S = {s.key: s for s in SPECS["heist"].ent_slots}
        ents = []
        for kind, off, ty in (("lock", 0, 1), ("key", 9, 2)):
            for i in range(3):
                x, y, on = st[base + off + 3 * i : base + off + 3 * i + 3]
                if on > 0.5:
                    ents.append(_make_entity(S[f"{kind}{i}"], x, y, i, ty, ty))
        ents.append(_make_entity(S["exit"], st[n + 4], st[n + 5], 0, 9, 9))
        ents.append(_make_entity(S["agent"], st[n + 1], st[n + 2], 0, 0, 0, rot=float(st[n + 3]), agent=True))
        for i in range(3):
            sl = S[f"ring{i}"]
            ents.append(_make_entity(sl, *sl.fixed_xy, i, 11, 2, rot=sl.rotation))
        b["entities"] = ents
        b["has_keys"] = [bool(st[base + 18 + i] > 0.5) for i in range(3)]
        b["num_keys"] = 3
        return b


@register
class ProcgenChaser(_ProcgenEnv):
    """Render a 13×13 maze with orb fills and three slots each for orbs, eggs and enemies.

    All enemies share one four-way sprite selector. Grid orbs use rounded centred
    sub-rectangles, not sprites; the single background needs no row field.
    Fixed z=0 slot order assumes source layouts where those entities do not overlap."""

    game = "chaser"
    name = "procgen-chaser"
    reference_id = "procgen-chaser-v0"
    n_bg_state = 0
    pool_warmup = 128
    state_dim = 13 * 13 + 2 + 9 + 9 + 9 + 1
    state_names = (
        tuple(f"cell_{y}_{x}" for y in range(13) for x in range(13))
        + ("agent_x", "agent_y")
        + tuple(f"{k}{i}_{c}" for k in ("orb", "egg", "enemy") for i in range(3) for c in ("x", "y", "present"))
        + ("enemy_sprite",)
    )
    _BASE = 13 * 13 + 2

    def _unpack(self, s):
        f = super()._unpack(s)
        b = self._BASE
        f["agent"] = (s[:, b - 2], s[:, b - 1])
        for j, k in enumerate(("orb", "egg", "enemy")):
            o = b + 9 * j
            f[k] = [(s[:, o + 3 * i], s[:, o + 3 * i + 1], s[:, o + 3 * i + 2]) for i in range(3)]
        f["enemy_sprite"] = s[:, b + 27].round().clamp_(0, 3).long()
        return f

    def _slot_fields(self, f, sl, B):
        if sl.key == "agent":
            return (None, *f["agent"], None, None)
        x, y, on = f[sl.key[:-1]][int(sl.key[-1])]
        return on > 0.5, x, y, None, None

    _sprite_lut: dict = {}

    def _slot_sprite(self, f, sl, B):
        if not sl.key.startswith("enemy"):
            return None
        key = str(self.device)
        lut = self._sprite_lut.get(key)
        if lut is None:
            lut = torch.as_tensor(_CHASER_ENEMY_SPRITE, device=self.device)
            self._sprite_lut[key] = lut
        return lut[f["enemy_sprite"]]

    @classmethod
    def state_from_blob(cls, b):
        ents = b["entities"]
        a = [e for e in ents if e["type"] == 0][0]
        row = [a["x"], a["y"]]
        for ty in (2, 4, 6):  # LARGE_ORB, ENEMY_EGG, ENEMY
            got = [e for e in ents if e["type"] == ty][:3]
            for i in range(3):
                e = got[i] if i < len(got) else None
                row += [e["x"] if e else 0.0, e["y"] if e else 0.0, 1.0 if e else 0.0]
        row.append(cls._enemy_sprite(b))
        return np.concatenate([_grid_to_classes("chaser", b["grid"]).reshape(-1), row]).astype(np.float32)

    @staticmethod
    def _enemy_sprite(b) -> int:
        """Map source time/eat-time fields to the four-way enemy sprite index."""
        if b["cur_time"] - b["eat_time"] < b["eat_timeout"]:
            return 0
        rem = (int(b["cur_time"]) // 2) % 4
        return 1 + (1 if rem == 3 else rem)

    @classmethod
    def blob_from_state(cls, template, st):
        b = dict(template)
        n, base = 169, cls._BASE
        b["grid"] = _classes_to_grid("chaser", st[:n], 13)
        b["grid_w"] = b["grid_h"] = 13
        b["grid_size"] = n
        S = {s.key: s for s in SPECS["chaser"].ent_slots}
        ents = []
        for j, (k, ty) in enumerate((("orb", 2), ("egg", 4), ("enemy", 6))):
            for i in range(3):
                x, y, on = st[base + 9 * j + 3 * i : base + 9 * j + 3 * i + 3]
                if on > 0.5:
                    ents.append(_make_entity(S[f"{k}{i}"], x, y, 0, ty, ty))
        ents.append(_make_entity(S["agent"], st[base - 2], st[base - 1], 0, 0, 0, agent=True))
        b["entities"] = ents
        # Drive `image_for_type(ENEMY)` to the requested sprite through the two fields it reads.
        # `eat_timeout` is 75 and comes from the template rather than being written here.
        k = int(round(float(st[base + 27])))
        if k == 0:
            b["cur_time"], b["eat_time"] = 0, 0
        else:
            b["cur_time"] = 2 * (k - 1)
            b["eat_time"] = b["cur_time"] - int(template["eat_timeout"]) - 1
        return b


@register
class ProcgenMiner(_ProcgenEnv):
    """Render the 20×20 digging grid with robot and exit sprites.

    Moving and stationary boulders/diamonds share render classes, so these rows
    cannot resume dynamics. Non-square backgrounds require horizontal crop state."""

    game = "miner"
    name = "procgen-miner"
    reference_id = "procgen-miner-v0"
    state_dim = 20 * 20 + 7
    state_names = tuple(f"cell_{y}_{x}" for y in range(20) for x in range(20)) + (
        "bg_index",
        "bg_pct_x",
        "agent_x",
        "agent_y",
        "agent_reflected",
        "exit_x",
        "exit_y",
    )

    def _unpack(self, s):
        f = super()._unpack(s)
        n = self.n_cells
        f["bg_pct_x"] = s[:, n + 1]
        f["agent"] = (s[:, n + 2], s[:, n + 3], s[:, n + 4])
        f["exit"] = (s[:, n + 5], s[:, n + 6])
        return f

    def _slot_fields(self, f, sl, B):
        if sl.key == "exit":
            return (None, *f["exit"], None, None)
        x, y, r = f["agent"]
        return None, x, y, None, r

    @classmethod
    def state_from_blob(cls, b):
        ents = b["entities"]
        a = [e for e in ents if e["type"] == 0][0]
        x = [e for e in ents if e["type"] == 6]
        return np.concatenate(
            [
                _grid_to_classes("miner", b["grid"]).reshape(-1),
                [
                    b["background_index"],
                    b["bg_pct_x"],
                    a["x"],
                    a["y"],
                    a["is_reflected"],
                    x[0]["x"] if x else 0.0,
                    x[0]["y"] if x else 0.0,
                ],
            ]
        ).astype(np.float32)

    @classmethod
    def blob_from_state(cls, template, st):
        b = dict(template)
        n = 400
        b["grid"] = _classes_to_grid("miner", st[:n], 20)
        b["grid_w"] = b["grid_h"] = 20
        b["grid_size"] = n
        b["background_index"] = int(round(float(st[n])))
        b["bg_pct_x"] = float(st[n + 1])
        S = {s.key: s for s in SPECS["miner"].ent_slots}
        b["entities"] = [
            _make_entity(S["exit"], st[n + 5], st[n + 6], 0, 6, 6),
            _make_entity(S["agent"], st[n + 2], st[n + 3], 0, 0, 0, refl=int(round(float(st[n + 4]))), agent=True),
        ]
        return b
