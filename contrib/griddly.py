"""Griddly wrappers must remain in the module cache until process exit.

Both the no-release subclass and cache ownership are required to avoid native
finalizer crashes. Do not release wrappers when an individual environment closes."""

from __future__ import annotations

import collections
import json
import os

import numpy as np
import torch

from stable_envs.env import Env
from stable_envs.registry import register
from stable_envs.render import _resources as _RES
from stable_envs.render import tilemap as tm
from stable_envs.render._cache import cache_path
from stable_envs.spaces import Box, Dict

#: bit k of the neighbour mask is ``OFFSETS[k]``. The four orthogonal offsets come first so a WALL_16
#: game keys on the low four bits and its table is 16 wide.
OFFSETS = ((0, -1), (-1, 0), (1, 0), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))
EMPTY_CHAR = "."
#: frames harvested from level 0 to (a) find the multi-object cell codes and (b) CHECK the table
BUILD_FRAMES = 200
#: griddly's own default when a game declares no ``TileSize``
DEFAULT_TILE = 24


def _require_griddly():
    """Probe source construction and rendering before registering the family."""
    try:
        import griddly  # noqa: F401
        from griddly import GymWrapper, gd  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            f"griddly is not importable ({type(e).__name__}: {e}). The atlas is composed of tiles "
            f"griddly's own Vulkan renderer draws, so there is no degraded mode. "
            f"`pip install griddly==1.6.7` — the only release, from 2023-03-15; upstream's last commit "
            f"is 2024-04-07 and there is no successor."
        ) from e
    try:
        env = _open("Single-Player/GVGAI/sokoban.yaml", DEFAULT_TILE)
        env.reset(level_id=0)
        fr = np.array(env.game.observe(), copy=True)
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            f"griddly is installed but cannot render: {type(e).__name__}: {e}. It renders headlessly "
            f"through Vulkan, so this is a driver/device problem rather than a packaging one."
        ) from e
    if fr.ndim != 3 or len(np.unique(fr)) < 2:
        raise ImportError(
            f"griddly returned a {fr.shape} frame with {len(np.unique(fr))} distinct "
            f"values; a uniform frame means the renderer is not drawing"
        )


def _games_dir() -> str:
    import griddly

    return os.path.join(os.path.dirname(griddly.__file__), "resources", "games")


def _read_yaml(rel: str):
    import yaml as pyyaml

    with open(os.path.join(_games_dir(), rel)) as f:
        src = f.read()
    return src, pyyaml.load(src, Loader=pyyaml.FullLoader)


def _yaml_at(rel: str, tile_size: int) -> str:
    """Set tile size on the Sprite2D observer before constructing its wrapper."""
    import yaml as pyyaml

    src, g = _read_yaml(rel)
    g["Environment"].setdefault("Observers", {}).setdefault("Sprite2D", {})["TileSize"] = int(tile_size)
    return pyyaml.dump(g, sort_keys=False)


def native_tile_size(rel: str) -> int:
    _src, g = _read_yaml(rel)
    sp = (g["Environment"].get("Observers") or {}).get("Sprite2D") or {}
    t = sp.get("TileSize", DEFAULT_TILE)
    return DEFAULT_TILE if isinstance(t, list) else int(t)


_WRAPPERS: dict[tuple[str, int], object] = {}
_NO_RELEASE: list = []


def _no_release_wrapper():
    """Create the lazy wrapper subclass used by the process-lifetime cache."""
    if not _NO_RELEASE:
        from griddly import GymWrapper

        class _NoReleaseGymWrapper(GymWrapper):
            def __del__(self):  # noqa: D105  — see ``_open``
                pass

        _NO_RELEASE.append(_NoReleaseGymWrapper)
    return _NO_RELEASE[0]


def _open(rel: str, tile_size: int):
    """Return the shared wrapper for a game and tile size.

    Keep both the no-release finalizer and module-cache reference: either alone
    is insufficient to prevent Griddly’s interpreter-exit crash."""
    key = (rel, int(tile_size))
    if key not in _WRAPPERS:
        from griddly import gd

        _WRAPPERS[key] = _no_release_wrapper()(
            yaml_string=_yaml_at(rel, tile_size),
            level=0,
            global_observer_type=gd.ObserverType.SPRITE_2D,
            player_observer_type=gd.ObserverType.SPRITE_2D,
        )
    return _WRAPPERS[key]


_require_griddly()


def _frame(env) -> np.ndarray:
    """Copy source ``(3, W, H)`` pixels into ``(H, W, 3)`` order."""
    return np.array(env.game.observe(), copy=True).swapaxes(0, 2)


def cell_codes(state) -> dict:
    """Map positions to sorted object-name tuples, excluding internal objects.

    Orientation, PlayerId and RenderTileId cannot be restored by level strings.
    Decoder construction rejects games whose frames require those fields."""
    m = collections.defaultdict(list)
    for o in state["Objects"]:
        if o["Name"].startswith("_"):
            continue
        m[tuple(int(v) for v in o["Location"])].append(o["Name"])
    return {k: tuple(sorted(v)) for k, v in m.items()}


def _sample_action(env, rng):
    sp = env.action_space
    if hasattr(sp, "nvec"):
        return [int(rng.integers(k)) for k in sp.nvec]
    if hasattr(sp, "n"):
        return int(rng.integers(sp.n))
    return [([int(rng.integers(k)) for k in s.nvec] if hasattr(s, "nvec") else int(rng.integers(s.n))) for s in sp]


def level_string(grid: dict, W: int, H: int, chars: dict) -> str:
    """Encode cells as space-separated tokens, joining stacked objects with ``/``."""
    return "\n".join(
        " ".join("/".join(chars[n] for n in grid.get((x, y), ())) or EMPTY_CHAR for x in range(W)) for y in range(H)
    )


def _cache_path(name: str, tile: int) -> str:
    return str(cache_path("tilemap", f"griddly_{name}_t{tile}.npz"))


class GriddlyDecoder:
    """Cache a game’s atlas and table indexed by cell code and neighbour classes.

    Live neighbour offsets determine a base-m mask. Out-of-grid neighbours use
    the class selected by exact reconstruction of source rollout frames."""

    _mem: dict[tuple[str, int, str], GriddlyDecoder] = {}

    def __init__(self, name: str, rel: str, tile: int, device="cuda"):
        self.name, self.rel, self.T = name, rel, int(tile)
        self.device = torch.device(device)
        d = self._load_or_build()
        self.alphabet = [tuple(c) for c in json.loads(str(d["alphabet"]))]
        self.chars = json.loads(str(d["chars"]))
        self.grid_w, self.grid_h = int(d["grid_w"]), int(d["grid_h"])
        self.m, self.oob_class = int(d["m"]), int(d["oob_class"])
        self.live = [int(k) for k in np.asarray(d["live"]).reshape(-1)]
        self.code_of = {c: i for i, c in enumerate(self.alphabet)}
        self.n_codes = len(self.alphabet)
        self.official = torch.from_numpy(np.ascontiguousarray(d["atlas"])).to(self.device)
        self.table = torch.from_numpy(np.ascontiguousarray(d["table"])).long().to(self.device)
        self.cls = torch.from_numpy(np.ascontiguousarray(d["cls"])).long().to(self.device)
        self._rgb_cache = _RES.BoundedCache(8, name="griddly.rgb")

    @classmethod
    def get(cls, name: str, rel: str, tile: int, device="cuda") -> GriddlyDecoder:
        key = (name, int(tile), str(device))
        if key not in cls._mem:
            cls._mem[key] = cls(name, rel, tile, device)
        return cls._mem[key]

    def _load_or_build(self) -> dict:
        path = _cache_path(self.name, self.T)
        if os.path.exists(path):
            try:
                return dict(np.load(path, allow_pickle=False))
            except Exception:  # noqa: BLE001  partial/corrupt
                pass
        d = build_decoder(self.name, self.rel, self.T)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **d)
        os.replace(tmp, path)
        return d

    def rgb(self, gain: np.ndarray) -> torch.Tensor:
        """Apply channel gains to atlas pixels before gathering; unity reuses source tiles."""
        g = np.asarray(gain, np.float64).reshape(3)
        if np.array_equal(g, np.ones(3)):
            return self.official
        key = g.tobytes()
        hit = self._rgb_cache.get(key)
        if hit is None:
            t = torch.as_tensor(g, dtype=torch.float32, device=self.device)
            hit = self._rgb_cache.put(key, (self.official.float() * t).clamp_(0, 255).to(torch.uint8).contiguous())
        return hit


class _Canvas:
    """Probe independent neighbourhoods in disjoint 3×3 blocks of one source frame."""

    def __init__(self, env, T: int, chars: dict, avatar, k: int = 10):
        self.env, self.T, self.chars, self.avatar, self.k = env, T, chars, avatar, k
        self.W, self.H = 3 * k + 2, 3 * k
        self.slots = [(3 * bi + 1, 3 * bj + 1) for bj in range(k) for bi in range(k)]
        self.loads = 0

    def render(self, configs) -> list[bytes]:
        out = []
        for i in range(0, len(configs), len(self.slots)):
            chunk = configs[i : i + len(self.slots)]
            grid = {}
            for (cx, cy), (centre, nbs) in zip(self.slots, chunk):
                if centre:
                    grid[(cx, cy)] = centre
                for k, code in nbs.items():
                    dx, dy = OFFSETS[k]
                    if code:
                        grid[(cx + dx, cy + dy)] = code
            # a game with an avatar object needs one on the board; parked two cells clear of every
            # probe block, and only when no configuration in this chunk already places one
            if self.avatar and not any(self.avatar in c for c in grid.values()):
                grid[(self.W - 1, 0)] = (self.avatar,)
            self.env.reset(level_string=level_string(grid, self.W, self.H, self.chars))
            self.loads += 1
            fr = _frame(self.env)
            for (cx, cy), _ in zip(self.slots, chunk):
                out.append(fr[cy * self.T : (cy + 1) * self.T, cx * self.T : (cx + 1) * self.T].tobytes())
        return out


def build_decoder(name: str, rel: str, tile: int, level: int = 0) -> dict:
    """Enumerate tiles and reject a decoder that fails exact source-frame reconstruction."""
    _src, g = _read_yaml(rel)
    chars = {o["Name"]: str(o["MapCharacter"]) for o in g.get("Objects", []) if o.get("MapCharacter") is not None}
    avatar = (g["Environment"].get("Player") or {}).get("AvatarObject")
    env = _open(rel, tile)
    env.seed(0)
    env.reset(level_id=level)
    fr0 = _frame(env)
    GW, GH = fr0.shape[1] // tile, fr0.shape[0] // tile

    # Collect stacked-object codes and independent frames for reconstruction checks.
    recs, ep = [], 0
    while len(recs) < BUILD_FRAMES and ep < 200:
        env.seed(ep)
        env.reset(level_id=level)
        rng = np.random.default_rng(4000 + ep)
        for _ in range(60):
            recs.append((cell_codes(env.get_state()), _frame(env)))
            try:
                _obs, _r, done, _info = env.step(_sample_action(env, rng))
            except Exception:  # noqa: BLE001
                break
            if np.any(done) or len(recs) >= BUILD_FRAMES:
                break
        ep += 1
    harvested = {c for cm, _ in recs for c in cm.values() if c}
    alphabet = [()] + sorted({(n,) for n in chars} | harvested, key=lambda c: (len(c), c))
    no_char = sorted({n for c in alphabet for n in c} - set(chars))
    if no_char:
        raise RuntimeError(
            f"griddly game {name}: the objects {no_char} occur on the board and declare no "
            f"MapCharacter, so no level string can place them and this game cannot be written back "
            f"into griddly at all"
        )
    cid = {c: i for i, c in enumerate(alphabet)}
    n = len(alphabet)
    canvas = _Canvas(env, tile, chars, avatar)

    # The source must restore every code before we use it as a renderer reference.
    grid = {(3 * (i % canvas.k) + 1, 3 * (i // canvas.k) + 1): c for i, c in enumerate(alphabet) if c}
    env.reset(level_string=level_string(grid, canvas.W, canvas.H, chars))
    back = cell_codes(env.get_state())
    wrong = {k: (v, back.get(k)) for k, v in grid.items() if back.get(k) != v}
    if wrong:
        k, (want, got) = next(iter(wrong.items()))
        raise RuntimeError(
            f"griddly game {name}: {len(wrong)} cell codes do not round-trip through a level string "
            f"(cell {k} was written as {want} and reads back as {got}). The state could not be put "
            f"back into griddly, so no parity number could be measured"
        )

    # Group indistinguishable neighbours and retain offsets that affect a tile.
    base = dict(zip(alphabet, canvas.render([(c, {}) for c in alphabet])))
    sig_tiles = canvas.render([(c, {k: d for k in range(8)}) for d in alphabet for c in alphabet])
    sig = collections.defaultdict(list)
    for (d, _c), t in zip([(d, c) for d in alphabet for c in alphabet], sig_tiles):
        sig[d].append(t)
    groups = collections.defaultdict(list)
    for d in alphabet:
        groups[tuple(sig[d])].append(d)
    # class 0 is the one holding the empty cell, so an all-empty neighbourhood keys to mask 0
    classes = sorted(groups.values(), key=lambda gg: (0 if () in gg else 1, str(gg[0])))
    cls_of = {d: i for i, gg in enumerate(classes) for d in gg}
    m = len(classes)
    rep = [gg[0] for gg in classes]
    pos_key = [(k, c) for k in range(8) for _r in rep[1:] for c in alphabet]
    pos_tiles = canvas.render([(c, {k: r}) for k in range(8) for r in rep[1:] for c in alphabet])
    live = sorted({k for (k, c), t in zip(pos_key, pos_tiles) if t != base[c]})
    K = len(live)

    # Enumerate all code/mask combinations, including unseen rollout neighbourhoods.
    cfgs, keys = [], []
    for c in alphabet:
        for mask in range(m**K):
            nbs, v = {}, mask
            for k in live:
                nbs[k] = rep[v % m]
                v //= m
            cfgs.append((c, nbs))
            keys.append((cid[c], mask))
    tiles = canvas.render(cfgs)
    uniq = sorted(set(tiles))
    tidx = {t: i for i, t in enumerate(uniq)}
    table = np.zeros((n, m**K), np.int32)
    for (ci, mask), t in zip(keys, tiles):
        table[ci, mask] = tidx[t]
    atlas = np.stack([np.frombuffer(t, np.uint8).reshape(tile, tile, 3) for t in uniq])
    cls = np.array([cls_of[c] for c in alphabet], np.int32)

    # Select an out-of-grid class only if every captured frame reconstructs exactly.
    grids = np.full((len(recs), GH, GW), -1, np.int32)
    for i, (cm, _f) in enumerate(recs):
        for y in range(GH):
            for x in range(GW):
                grids[i, y, x] = cid.get(cm.get((x, y), ()), -1)
    if (grids < 0).any():
        raise RuntimeError(f"griddly game {name}: a harvested cell code is outside the alphabet")
    ref = np.stack([r[1] for r in recs])
    oob_class, worst = None, None
    for oob in range(m):
        idx = _mask_index_np(grids, cls, live, m, oob)
        got = atlas[table[grids, idx]].transpose(0, 1, 3, 2, 4, 5).reshape(len(recs), GH * tile, GW * tile, 3)
        d = int(np.abs(got.astype(np.int32) - ref.astype(np.int32)).max())
        worst = d if worst is None else min(worst, d)
        if d == 0:
            oob_class = oob
            break
    if oob_class is None:
        raise RuntimeError(
            f"griddly game {name}: the (cell code, neighbour class) table does not reproduce griddly's "
            f"own frames (best max abs {worst}/255 over {len(recs)} frames). The picture depends on "
            f"something the cell code does not carry — an Orientation, a RenderTileId or a PlayerId, "
            f"none of which a level string can set — so this game is not renderable from this state"
        )
    env.reset(level_id=level)
    return {
        "alphabet": json.dumps([list(c) for c in alphabet]),
        "chars": json.dumps(chars),
        "atlas": atlas,
        "table": table,
        "cls": cls,
        "live": np.array(live, np.int32),
        "m": m,
        "oob_class": oob_class,
        "grid_w": GW,
        "grid_h": GH,
        "canvas_loads": canvas.loads,
    }


def _mask_index_np(grids: np.ndarray, cls: np.ndarray, live, m: int, oob: int) -> np.ndarray:
    B, H, W = grids.shape
    pad = np.full((B, H + 2, W + 2), oob, np.int64)
    pad[:, 1:-1, 1:-1] = cls[grids]
    out = np.zeros((B, H, W), np.int64)
    for bit, k in enumerate(live):
        dx, dy = OFFSETS[k]
        out += pad[:, 1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W] * (m**bit)
    return out


class _GriddlyGame(Env):
    """Render one fixed Griddly level from cell codes and their neighbours."""

    family = "griddly"

    #: path under griddly's ``resources/games``
    game_yaml: str = ""
    #: the level this class pins. A class fixes ``state_dim``, and griddly's levels differ in grid size.
    level: int = 0
    grid_w: int = 0
    grid_h: int = 0

    @property
    def frame_shape(self) -> tuple[int, int]:
        """Return grid dimensions multiplied by the tile edge derived from ``size``."""
        t = self._tile(int(self.size))
        return self.grid_h * t, self.grid_w * t

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.dec = GriddlyDecoder.get(
            type(self).__name__, self.game_yaml, self._tile(self.size), device=str(self.device)
        )
        self._check_grid(self.dec)
        self._pool: np.ndarray | None = None

    def _close(self) -> None:
        """Drop the instance’s rollout pool; shared wrappers remain alive until exit."""
        self._pool = None

    def _check_grid(self, dec) -> None:
        if (dec.grid_w, dec.grid_h) != (self.grid_w, self.grid_h):
            raise AssertionError(
                f"{type(self).__name__} declares a {self.grid_w}x{self.grid_h} grid but griddly builds "
                f"{dec.grid_w}x{dec.grid_h} for level {self.level} of {self.game_yaml}; every rendered "
                f"frame would be the wrong shape"
            )

    def _tile(self, size: int) -> int:
        t, r = divmod(int(size), self.grid_h)
        if r or t < 4:
            raise ValueError(
                f"{type(self).__name__}: size {size} is not a whole tile size for a {self.grid_w}x"
                f"{self.grid_h} grid; use a multiple of {self.grid_h} that is at least {4 * self.grid_h}"
            )
        return t

    def make_variation_space(self) -> Dict:
        """Expose a global RGB gain applied to the source sprite colours."""
        return Dict({"channel_gain": Box(low=0.0, high=2.0, shape=(3,), dtype=np.float32, init_value=[1.0, 1.0, 1.0])})

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
        """Gather cell rows into ``(..., grid_h*T, grid_w*T, 3)`` uint8 frames."""
        if supersample not in (None, 1):
            raise ValueError(
                f"{type(self).__name__}: supersample={supersample} is not available. The frame is a "
                f"gather from tiles griddly composed at TileSize, so there is no sub-pixel coverage to "
                f"estimate — ask for a larger `size`, which rebuilds the atlas at that TileSize."
            )
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        tile = self._tile(int(size or self.size))
        dec = (
            self.dec
            if tile == self.dec.T
            else GriddlyDecoder.get(type(self).__name__, self.game_yaml, tile, device=str(self.device))
        )
        sty = self.resolve_style(style, variation, batch=int(s.shape[0]))
        idx = self._tile_indices(s, dec)
        flat = None
        if out is not None:
            flat = out.reshape(-1, self.grid_h * tile, self.grid_w * tile, 3)
        gain = sty["channel_gain"]
        if sty.shared:
            img = tm.blit(dec.rgb(gain[0].detach().cpu().numpy()), idx, tile, out=flat)
        else:
            # Per-pixel scaling commutes with gathering, including uint8 truncation.
            img = tm.blit(dec.official, idx, tile)
            img = (img.float() * gain.view(-1, 1, 1, 3)).clamp_(0, 255).to(torch.uint8)
            if flat is not None:
                flat.copy_(img)
                img = flat
        return img.reshape(*lead, *img.shape[1:]) if lead else img

    def _tile_indices(self, s: torch.Tensor, dec) -> torch.Tensor:
        """Round and wrap cell codes, then resolve their neighbour-dependent atlas slots."""
        B, H, W = s.shape[0], self.grid_h, self.grid_w
        codes = torch.round(s).long().remainder(dec.n_codes).reshape(B, H, W)
        if not dec.live:
            return dec.table[codes, 0].to(torch.int32)
        pad = torch.full((B, H + 2, W + 2), dec.oob_class, dtype=torch.long, device=s.device)
        pad[:, 1:-1, 1:-1] = dec.cls[codes]
        mask = torch.zeros((B, H, W), dtype=torch.long, device=s.device)
        for bit, k in enumerate(dec.live):
            dx, dy = OFFSETS[k]
            mask += pad[:, 1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W] * (dec.m**bit)
        return dec.table[codes, mask].to(torch.int32)

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample source rollouts at the registered level, preserving valid map structure."""
        rng = rng or self.rng
        if self._pool is None:
            self._pool = self._build_pool(max(256, batch))
        if batch <= len(self._pool):
            return self._pool[rng.choice(len(self._pool), batch, replace=False)].copy()
        return self._pool[rng.integers(0, len(self._pool), batch)].copy()

    def _build_pool(self, n: int) -> np.ndarray:
        env = _open(self.game_yaml, self.dec.T)
        out, ep = [], 0
        while len(out) < n and ep < 400:
            env.seed(ep)
            env.reset(level_id=self.level)
            r = np.random.default_rng(9000 + ep)
            for _ in range(60):
                out.append(self.encode(cell_codes(env.get_state())))
                if len(out) >= n:
                    break
                try:
                    _o, _rw, done, _i = env.step(_sample_action(env, r))
                except Exception:  # noqa: BLE001
                    break
                if np.any(done):
                    break
            ep += 1
        env.reset(level_id=self.level)
        return np.stack(out[:n]).astype(np.float32)

    def encode(self, cm: dict) -> np.ndarray:
        dec = self.dec
        out = np.zeros(self.state_dim, np.float32)
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                c = cm.get((x, y), ())
                if c not in dec.code_of:
                    raise ValueError(
                        f"{type(self).__name__}: griddly produced the cell {c}, which is "
                        f"not in the alphabet the decoder was built for"
                    )
                out[y * self.grid_w + x] = dec.code_of[c]
        return out

    @classmethod
    def reference_env(cls, size: int | None = None):
        tile = int(size or cls.default_size) // cls.grid_h
        env = _open(cls.game_yaml, tile)
        env.reset(level_id=cls.level)
        env._se_decoder = GriddlyDecoder.get(cls.__name__, cls.game_yaml, tile, device="cpu")
        return env

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Restore cells through source level strings and verify the cell-code round trip.

        The check covers object names and positions; decoder construction separately
        checks that this representation reproduces source pixels."""
        dec = env._se_decoder
        out = []
        for st in np.asarray(states):
            codes = np.round(np.asarray(st, np.float64)).astype(np.int64) % dec.n_codes
            grid = {}
            for k, c in enumerate(codes):
                code = dec.alphabet[int(c)]
                if code:
                    grid[(k % cls.grid_w, k // cls.grid_w)] = code
            env.reset(level_string=level_string(grid, cls.grid_w, cls.grid_h, dec.chars))
            back = cell_codes(env.get_state())
            wrong = [k for k in set(back) | set(grid) if back.get(k, ()) != grid.get(k, ())]
            if wrong:
                raise AssertionError(
                    f"{cls.__name__}: the reference did not reach the requested state "
                    f"({len(wrong)} cells differ, e.g. {wrong[0]} asked for "
                    f"{grid.get(wrong[0], ())} and holds {back.get(wrong[0], ())}); parity measured "
                    f"against this frame would compare two scenes"
                )
            out.append(_frame(env))
        return np.stack(out)


def _cell_names(w: int, h: int) -> tuple[str, ...]:
    return tuple(f"cell[{r},{c}]" for r in range(h) for c in range(w))


def game(class_name: str, rel: str, w: int, h: int, doc: str = "", level: int = 0) -> type:
    """Register a fixed game/level with declared grid dimensions checked at construction."""
    tile = native_tile_size(rel)
    cls = type(
        class_name,
        (_GriddlyGame,),
        {
            "__doc__": doc or f"griddly {rel}, level {level}: a {w}x{h} grid.",
            "__module__": __name__,
            "name": "griddly-" + os.path.basename(rel)[:-5].replace("_", "-"),
            "reference_id": f"griddly/{rel}#level{level}",
            "game_yaml": rel,
            "level": level,
            "grid_w": w,
            "grid_h": h,
            "state_dim": w * h,
            "state_names": _cell_names(w, h),
            "default_size": h * tile,
        },
    )
    return register(cls)


GriddlySokoban = game(
    "GriddlySokoban",
    "Single-Player/GVGAI/sokoban.yaml",
    13,
    9,
    doc="GVGAI Sokoban: push every box into a hole. 6 cell codes, WALL_16 walls, and the one game "
    "whose avatar shares a cell with another object (it walks over holes).",
)
GriddlySokoban2 = game(
    "GriddlySokoban2",
    "Single-Player/GVGAI/sokoban2.yaml",
    8,
    7,
    doc="GVGAI Sokoban2: boxes that stay on the board once placed, so the alphabet carries two "
    "stacked cell codes rather than one.",
)
GriddlyLabyrinth = game(
    "GriddlyLabyrinth",
    "Single-Player/GVGAI/labyrinth.yaml",
    16,
    14,
    doc="GVGAI Labyrinth: reach the exit without stepping in a trap. The densest wall map in the "
    "family, so the WALL_16 neighbour table is fully exercised.",
)
GriddlyButterflies = game(
    "GriddlyButterflies",
    "Single-Player/GVGAI/butterflies.yaml",
    28,
    11,
    doc="GVGAI Butterflies: catch the butterflies before they hatch every cocoon. The widest grid "
    "here, 264x672 at the native tile size.",
)
GriddlyBait = game(
    "GriddlyBait",
    "Single-Player/GVGAI/bait.yaml",
    5,
    6,
    doc="GVGAI Bait: keys, boxes and holes on the smallest shipped grid — the case where a whole "
    "frame is 30 cells and the state is 30 numbers.",
)
GriddlyClusters = game(
    "GriddlyClusters",
    "Single-Player/GVGAI/clusters.yaml",
    13,
    10,
    doc="GVGAI Clusters: push coloured blocks together. Its MapCharacters are the DIGITS 1/2/3, "
    "which is the case a naive level-string writer gets wrong (a digit after a character is a "
    "player id) and the round-trip assertion catches.",
)
GriddlyCookMePasta = game(
    "GriddlyCookMePasta",
    "Single-Player/GVGAI/cookmepasta.yaml",
    14,
    11,
    doc="GVGAI CookMePasta: fetch and combine ingredients. The one registered game that declares no "
    "TilingMode at all, so its table has a single mask column and the frame is a plain per-cell "
    "gather — the control that shows the neighbour term is measured, not assumed.",
)
