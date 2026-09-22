"""Griddly — the family with the largest headroom in this repo, because its own renderer is the slow half.

Reference: ``griddly`` 1.6.7 (PyPI, 2023-03-15; upstream's last commit is 2024-04-07 and there is no
successor and no deprecation notice, so the version is PINNED rather than floated). Frames come from
the ``SPRITE_2D`` global observer, which is a grid of ``TileSize`` blocks with no sub-tile offsets.

Every other family here is compared against a reference renderer that is already faster than reading
the dataset it would replace. Griddly's is **slower**: measured through these classes on one H200, the
official Vulkan renderer does 499-1,352 f/s on a pinned core against 873-5,636 f/s/core of JPEG decode
of the *same* frames — i.e. **0.24-0.65 cores** — and the render is **98.1-99.5 %** of an env step. So
the crossover is not at some large batch, it is at **batch 1** (3,186-5,736 f/s against 499-1,352), and
the headline is not a ratio but a core count: **(x3,687-x8,015 at batch 8,192)** and **1,628-3,450 CPU
cores of decode replaced**, at 2.29-9.18 M frames/s. See ``docs/research/griddly.md``.

The decoder, and why it is a table rather than a harvest
-------------------------------------------------------

**A griddly tile is NOT a function of what stands in its cell.** 23 of the 32 shipped games declare
``TilingMode: WALL_16`` on their walls, which picks the wall image from neighbour occupancy and leaves
``RenderTileId`` at 0; over 609,103 cells across 8 games a per-cell key collides 676-59,773 times while
a cell + 8-neighbour key collides 0 times. That is a genuinely different shape from MiniGrid, MiniHack,
Craftax and Crafter, all of which are per-cell gathers — and it is still a gather, one extra table
lookup before the blit.

A 3x3 stencil of cell codes cannot be *stored* as a table (``n_codes ** 9`` entries), and one harvested
from rollouts would not be COMPLETE — the conformance suite proves a state dimension is live by
substituting a donor code into a real state, which manufactures neighbourhoods no rollout produces, and
a harvested table would fall off its own end there. So the stencil is FACTORED, by measurement:

    tile = table[cell code, sum over the live neighbour offsets of class(neighbour code) * m**bit]

``class`` and the live offsets are not read off the yaml, they are asked of griddly. Every (centre,
neighbour) pair is rendered on a lattice of 3x3 probe blocks; codes that no centre can tell apart share
a class, and an offset that changes no tile is dropped. Measured on the seven registered games: **m = 2
(wall / not-wall) and the 4 orthogonal offsets** for the six WALL_16 games, and **m = 1 with no live
offset at all** for ``cookmepasta``, which declares no tiling and really is a per-cell gather. The whole
table is then 5-10 codes x 16 masks, the atlas is 9-25 composed tiles (15.5-43.2 kB), and building it
costs **3-5 renders and about a second**.

The build is not trusted, it is CHECKED: 200 frames of real rollouts are reconstructed from the table
and the build raises unless **max abs pixel difference is 0 on every one**. That check is also what
decides which games are registered, rather than a hand-maintained opinion — see "what is refused".

Nothing is placed through ``load_state``
----------------------------------------

Two griddly facts shaped this, both measured:

  * ``GymWrapper.load_state`` **returns a new wrapper** and leaves the receiver untouched, and each
    wrapper owns a Vulkan device. Holding them dies at ~260 with
    ``vkGetDeviceQueue: Invalid device``; dropping them is safe (600 created and dropped, zero pixel
    difference) but then every reference frame costs a device create/destroy.
  * ``load_state`` also **loses information**: restoring an avatar with ``Orientation: RIGHT`` reads
    back ``LEFT`` and vice versa (UP/DOWN unchanged), which is griddly issue #301 — approved fix PR
    #300, unmerged since 2024-03-27. The frame agrees with what the env reports, so this is a state
    bug and not a renderer bug, but it makes ``load_state`` unusable as a way to put a chosen scene on
    the board.

``reset(level_string=...)`` has neither problem: it mutates the wrapper in place, accepts any grid
size, and its grammar has a ``/`` separator for a cell holding more than one object (``A/h`` reads back
as ``('avatar', 'hole')``). So one env builds the atlas AND serves as the parity reference, and this
module creates exactly one griddly wrapper per (game, tile size).

What is refused, and why that is a measurement
---------------------------------------------

A level string cannot set an object's ``Orientation``, its ``RenderTileId`` or its ``PlayerId``, so the
cell code cannot carry them either. Games whose picture depends on them are REFUSED by the
reconstruction check rather than by a list here. Measured over the 15 candidate games:

    registered   sokoban, sokoban2, labyrinth, butterflies, bait, clusters, cookmepasta
    refused      zelda, spider-nest, minigrid-drunkdwarf/doggo/eyeball/spiders — an oriented avatar
                 or monster, so the frame depends on a field the cell code does not carry
                 foragers — four harvesters that differ only by PlayerId, which collapse to one code
                 zenpuzzle — an object in the alphabet (``walked``) has no MapCharacter at all

The interpreter-exit segfault
-----------------------------

Griddly segfaults at interpreter exit on **6/6** runs, after all output is flushed. That is the one
crash shape crash isolation cannot resume past: pytest has already written its report, so no test is in
flight and ``stable_envs.testing.supervisor`` would have to report a green run as a session-level
failure.

Containment takes TWO things and it is measured which, because the obvious single fix does not work
(the frozen measurement record, four modes x 6 runs, exit codes only):

    stock       GymWrapper bound to a script-level name                  -11 -11 -11 -11 -11 -11
    noreload    __del__ overridden to do nothing, script-level name      -11 -11 -11 -11 -11 -11
    stockcache  stock GymWrapper parked in an imported module's dict     -11 -11 -11 -11 -11 -11
    cached      no-release subclass, reached ONLY through _open()'s cache  0   0   0   0   0   0

So it is not enough to stop ``GymWrapper.__del__`` calling ``game.release()`` — the pybind11 object's
own destructor is fatal too — and it is not enough to keep the wrapper alive in a module: a
script-level name is finalised early in teardown and an imported module's dict is not, but the stock
finaliser still runs. Both together are clean, 6/6. This module therefore builds every wrapper as a
``_NoReleaseGymWrapper`` AND hands it out only from ``_WRAPPERS``, so nothing user-facing ever holds
the only reference.

That is the same create-and-never-free rule ``testing/conformance.py`` states for EGL contexts, for the
same reason — the release is what corrupts — and it costs a few megabytes held for the life of the
process. It is not a substitute for the supervisor, which is still how the suite is run; it is what
stops the supervisor being asked to tell a real crash apart from an exit-time one. End to end:
``supervisor -- tests/ -k Griddly`` returns **238 passed, 35 skipped, 1 segment, rc 0**.
"""

from __future__ import annotations

import collections
import json
import os

import numpy as np
import torch

from ..env import Env
from ..registry import register
from ..render import _resources as _RES
from ..render import tilemap as tm
from ..spaces import Box, Dict

#: bit k of the neighbour mask is ``OFFSETS[k]``. The four orthogonal offsets come first so a WALL_16
#: game keys on the low four bits and its table is 16 wide.
OFFSETS = ((0, -1), (-1, 0), (1, 0), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))
EMPTY_CHAR = "."
#: frames harvested from level 0 to (a) find the multi-object cell codes and (b) CHECK the table
BUILD_FRAMES = 200
#: griddly's own default when a game declares no ``TileSize``
DEFAULT_TILE = 24


def _require_griddly():
    """Probe the PATH THAT WOULD FAIL, not just ``import griddly``.

    Importing the package proves nothing useful: it succeeds on a machine with no Vulkan device, where
    every render then aborts, and it says nothing about the shipped game resources — which ARE this
    env, since the atlas is composed of tiles griddly draws and nothing here re-derives them. So a
    wrapper is built and a frame is taken, and ``ImportError`` is raised if any of it fails, which is
    what makes the family cleanly ABSENT rather than present and failing inside a fixture.
    """
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
    """The game's yaml with ``Observers.Sprite2D.TileSize`` set to ``tile_size``.

    Tile size is a property of the OBSERVER, not of a render call, so it is patched into the yaml the
    wrapper is built from. Frame shape was verified to be exactly ``grid_w*T x grid_h*T`` at
    T in {4, 8, 12, 16, 24, 32, 48, 64}.
    """
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
    """``GymWrapper`` with its finaliser removed, built once.

    Griddly's ``__del__`` calls ``self.game.release()``; see ``_open``. Defined lazily because the
    import guard below is what decides whether griddly can be imported at all.
    """
    if not _NO_RELEASE:
        from griddly import GymWrapper

        class _NoReleaseGymWrapper(GymWrapper):
            def __del__(self):  # noqa: D105  — see ``_open``
                pass

        _NO_RELEASE.append(_NoReleaseGymWrapper)
    return _NO_RELEASE[0]


def _open(rel: str, tile_size: int):
    """One griddly wrapper per (game, tile size), built once and NEVER released.

    ``GymWrapper.__del__`` runs ``game.release()``, which segfaults at interpreter exit on 6/6 runs —
    after pytest has written its report, so no crash-isolating runner can attribute it to a test. The
    subclass below removes the finaliser and the cache keeps the wrapper alive, which is the same
    create-and-never-free rule ``testing/conformance.py`` states for EGL contexts.
    """
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
    """``(H, W, 3)`` uint8. griddly hands back ``(3, W, H)``; the swap is griddly's own convention."""
    return np.array(env.game.observe(), copy=True).swapaxes(0, 2)


def cell_codes(state) -> dict:
    """``(x, y) -> the sorted tuple of object names standing there``, ``_``-prefixed ones dropped.

    ``_empty`` and ``_boundary`` are griddly's own bookkeeping objects; they appear in ``get_state()``
    and paint nothing. Orientation, PlayerId and RenderTileId are deliberately NOT in the key: a level
    string cannot set them, so a state that carried them could not be written back into griddly, and a
    game whose picture depends on them is refused by the reconstruction check rather than approximated.
    """
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
    """``{(x, y): code}`` -> a griddly level string; space separated, ``/`` inside a cell."""
    return "\n".join(
        " ".join("/".join(chars[n] for n in grid.get((x, y), ())) or EMPTY_CHAR for x in range(W)) for y in range(H)
    )


def _cache_path(name: str, tile: int) -> str:
    root = os.environ.get("STABLE_ENVS_CACHE") or os.path.expanduser("~/.cache/stable-envs")
    return os.path.join(root, "tilemap", f"griddly_{name}_t{tile}.npz")


class GriddlyDecoder:
    """The complete tile table for one (game, tile size), enumerated by asking griddly.

    Held per process and cached on disk. Everything a frame needs is in here: the cell alphabet, the
    neighbour class of each code, which offsets are live, the ``(n_codes, m**live)`` table, the atlas
    of composed tiles, and which class an out-of-grid neighbour counts as.
    """

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
        """The atlas under a channel gain. ``gain == 1`` returns the official tiles untouched.

        Recolouring the 9-25 tile ATLAS rather than the frame is exactly the same arithmetic — a
        per-pixel scale commutes with a gather — and it is two orders of magnitude less work.
        """
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
    """Renders many (centre, neighbourhood) configurations per griddly frame, through one wrapper.

    Configurations sit on a lattice of 3x3 blocks so each centre's whole 8-neighbourhood is inside its
    own block and no two configurations can see one another.
    """

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
    """Ask griddly for the whole decoder, then refuse it unless it reproduces real frames exactly."""
    _src, g = _read_yaml(rel)
    chars = {o["Name"]: str(o["MapCharacter"]) for o in g.get("Objects", []) if o.get("MapCharacter") is not None}
    avatar = (g["Environment"].get("Player") or {}).get("AvatarObject")
    env = _open(rel, tile)
    env.seed(0)
    env.reset(level_id=level)
    fr0 = _frame(env)
    GW, GH = fr0.shape[1] // tile, fr0.shape[0] // tile

    # --- harvest level 0: the multi-object cell codes, and the frames the table is CHECKED against
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

    # --- every code must survive a level-string round trip, or the reference bridge is a fiction
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

    # --- which codes are distinguishable AS a neighbour, and which offsets matter at all
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

    # --- enumerate (code, mask) -> tile
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

    # --- THE CHECK. Rebuild every harvested frame from the table; anything but 0 is a refusal.
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
    """One griddly game at one level, rendered as a gather from an enumerated tile table."""

    family = "griddly"

    #: path under griddly's ``resources/games``
    game_yaml: str = ""
    #: the level this class pins. A class fixes ``state_dim``, and griddly's levels differ in grid size.
    level: int = 0
    grid_w: int = 0
    grid_h: int = 0

    @property
    def frame_shape(self) -> tuple[int, int]:
        """``(H, W)`` of a frame: ``(grid_h * T, grid_w * T)``, which is what ``render`` reshapes to.

        ``Env.frame_spec`` assumes a SQUARE frame unless the env says otherwise, and griddly's
        levels are not: the module header records the frames as ``grid_w*T x grid_h*T`` and the games
        this family pins are wider than they are tall, so an unqualified spec described a
        ``(size, size)`` buffer. ``size`` divides the grid HEIGHT into tiles (``_tile``); read off the
        same two class attributes ``render`` uses, so the spec cannot drift from the frame.
        """
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
        """Drop the sampled state pool. ``Env.close`` calls this once and is idempotent.

        The griddly wrapper and the decoder are NOT released, and that is this module's standing
        policy rather than an oversight: ``GymWrapper.__del__`` runs ``game.release()``, which
        segfaults at interpreter exit (see ``_open``), so the wrapper cache exists precisely to keep
        every one of them alive. What an instance owns is the pool of states sampled off the
        reference, which is rebuildable.
        """
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

    # variation

    def make_variation_space(self) -> Dict:
        """A global channel gain, and deliberately nothing else.

        There is no per-object colour knob and there cannot be an honest one: griddly composites 207
        sprite PNGs, so a "wall colour" would have to invent a recolouring rule the reference does not
        have — the same reason ``craftax.py`` offers only ``channel_gain``. The gain is exactly the
        identity at 1.0, so the default reproduces the official frame by construction rather than by
        agreement, and it is applied to the 9-25 tile ATLAS rather than to the frame, which is the same
        arithmetic for a fraction of the work.
        """
        return Dict({"channel_gain": Box(low=0.0, high=2.0, shape=(3,), dtype=np.float32, init_value=[1.0, 1.0, 1.0])})

    # rendering

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
        """``(...,grid_w*grid_h)`` -> ``(...,grid_h*T,grid_w*T,3)`` uint8 on device."""
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
            # one appearance per environment: gather the official tiles once and scale the frame.
            # trunc(atlas*g) then gather and gather then trunc(*g) are the same numbers — a per-pixel
            # scale commutes with a gather — so the two paths cannot disagree.
            img = tm.blit(dec.official, idx, tile)
            img = (img.float() * gain.view(-1, 1, 1, 3)).clamp_(0, 255).to(torch.uint8)
            if flat is not None:
                flat.copy_(img)
                img = flat
        return img.reshape(*lead, *img.shape[1:]) if lead else img

    def _tile_indices(self, s: torch.Tensor, dec) -> torch.Tensor:
        """``(B, n_cells)`` float codes -> ``(B, grid_h, grid_w)`` int32 atlas slots.

        Wrapping rather than clamping, for the reason MiniGrid wraps: the conformance suite proves a
        dimension reaches the render by moving it, and a clamp would pin an already-large code in place
        and report a live dimension as dead.
        """
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

    # states

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """States from real rollouts of the official env at the pinned level.

        Griddly's levels are hand-authored maps with strong invariants (a wall ring, one avatar), so
        uniform noise over the code alphabet would exercise a region of state space no episode visits
        and prove nothing about the frames anyone would store.
        """
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

    # reference bridge

    @classmethod
    def reference_env(cls, size: int | None = None):
        tile = int(size or cls.default_size) // cls.grid_h
        env = _open(cls.game_yaml, tile)
        env.reset(level_id=cls.level)
        env._se_decoder = GriddlyDecoder.get(cls.__name__, cls.game_yaml, tile, device="cpu")
        return env

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Official frames, by writing each state back as a griddly level string.

        The round trip is asserted rather than assumed: if the reference did not end up at the
        requested scene, every parity number computed from it compares two different pictures, and the
        failure presents as a bug in this renderer. That assertion is also what refuses a game whose
        picture depends on an Orientation or a PlayerId — the level string cannot carry them, so the
        read-back differs and the comparison stops instead of quietly measuring something else.
        """
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
    """Build and register a class for one griddly game at one level.

    A class per game rather than one parameterised env: the registry is the conformance suite's
    parameter list, so a game reachable only through a constructor argument is a game nothing tests.
    The declared grid size is re-checked against griddly every time an env is built (``_check_grid``),
    because a hardcoded one can drift when the package changes a level and a wrong one is silent.
    """
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
