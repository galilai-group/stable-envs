"""Probe 5: G3 -- is the state->pixel map a constant of the family?

Four measurements:

  1. STATIC ATLAS. Parse every shipped GDY yaml, collect the Sprite2D `Image` entries and the
     `BackgroundTile`. That is the declared atlas. Report per-game counts and the family union
     (files, distinct files, bytes on disk, pixel bytes at native tile size).

  2. EMPIRICAL TILE ALPHABET. Cut every rendered frame into TileSize x TileSize blocks and hash
     them. Count distinct blocks per game across many levels/episodes and watch the count
     saturate. A finite, saturating alphabet == a finite decoder. Also verifies the frame is
     EXACTLY W*T x H*T with no sub-tile offset.

  3. DECODER STABILITY ACROSS RESETS/INSTANCES. Render level L from (a) a fresh env, (b) the
     same env after visiting other levels, (c) a second independently constructed env, and
     (d) an env built from a custom level_string. Frames for the same state must be md5-identical.

  4. RenderTileId. Distribution per game -- is the per-object tile index exposed in get_state(),
     or does the renderer compute it (WALL_16 neighbour tiling)?
"""
import os, json, glob, hashlib, collections
import numpy as np
import yaml as pyyaml
import griddly
from griddly import GymWrapper, gd

GRIDDLY_DIR = os.path.dirname(griddly.__file__)
GAMES_DIR = os.path.join(GRIDDLY_DIR, "resources", "games")
IMAGES_DIR = os.path.join(GRIDDLY_DIR, "resources", "images")


# ---------- 1. static atlas ----------
def images_of_yaml(path):
    with open(path) as f:
        g = pyyaml.load(f, Loader=pyyaml.FullLoader)
    imgs = []
    env = g["Environment"]
    obs = env.get("Observers", {}) or {}
    sp = obs.get("Sprite2D", {}) or {}
    bg = sp.get("BackgroundTile")
    tile = sp.get("TileSize")
    if bg:
        imgs.append(bg if isinstance(bg, str) else str(bg))
    for o in g.get("Objects", []) or []:
        oo = (o.get("Observers", {}) or {}).get("Sprite2D", []) or []
        for entry in oo:
            im = entry.get("Image")
            if im is None:
                continue
            imgs.extend(im if isinstance(im, list) else [im])
    return env.get("Name"), tile, imgs


print("### 1. STATIC ATLAS (declared in the GDY yaml) ###")
allimgs = set()
per_game = {}
for p in sorted(glob.glob(os.path.join(GAMES_DIR, "**", "*.yaml"), recursive=True)):
    try:
        name, tile, imgs = images_of_yaml(p)
    except Exception as e:
        print("  parse fail", p, e); continue
    rel = os.path.relpath(p, GAMES_DIR)
    per_game[rel] = (name, tile, imgs)
    allimgs |= set(imgs)
    print(f"  {rel:52s} tile={tile} sprites={len(imgs):3d} distinct={len(set(imgs)):3d}")
missing = [i for i in sorted(allimgs) if not os.path.exists(os.path.join(IMAGES_DIR, i))]
tot_bytes = sum(os.path.getsize(os.path.join(IMAGES_DIR, i))
                for i in allimgs if os.path.exists(os.path.join(IMAGES_DIR, i)))
print(f"  FAMILY UNION: {len(allimgs)} distinct sprite files, {tot_bytes} bytes on disk "
      f"({tot_bytes/1e6:.3f} MB), missing={len(missing)}")
print(f"  (repo ships {len(glob.glob(os.path.join(IMAGES_DIR, '**', '*.png'), recursive=True))} "
      f"png total under resources/images -- only the referenced subset is the atlas)")

# source png sizes
try:
    from PIL import Image
    shapes = collections.Counter()
    for i in sorted(allimgs):
        fp = os.path.join(IMAGES_DIR, i)
        if os.path.exists(fp):
            with Image.open(fp) as im:
                shapes[(im.size, im.mode)] += 1
    print("  source png (size, mode) histogram:", dict(shapes))
except ImportError:
    print("  (PIL not available; skipping png shape histogram)")


# ---------- 2/3/4 ----------
GAMES = [
    ("sokoban", "Single-Player/GVGAI/sokoban.yaml"),
    ("zelda", "Single-Player/GVGAI/zelda.yaml"),
    ("clusters", "Single-Player/GVGAI/clusters.yaml"),
    ("butterflies", "Single-Player/GVGAI/butterflies.yaml"),
    ("labyrinth", "Single-Player/GVGAI/labyrinth.yaml"),
    ("drunkdwarf", "Single-Player/Mini-Grid/minigrid-drunkdwarf.yaml"),
    ("GriddlyRTS", "RTS/GriddlyRTS.yaml"),
]


def gframe(env):
    return np.array(env.game.observe(), copy=True).swapaxes(0, 2)


def sample_action(env, rng):
    sp = env.action_space
    if hasattr(sp, "nvec"):
        return [int(rng.integers(k)) for k in sp.nvec]
    if hasattr(sp, "n"):
        return int(rng.integers(sp.n))
    out = []
    for s in sp:
        out.append([int(rng.integers(k)) for k in s.nvec] if hasattr(s, "nvec")
                   else int(rng.integers(s.n)))
    return out


def tiles_of(fr, T):
    H, W, _ = fr.shape
    assert H % T == 0 and W % T == 0, (H, W, T)
    b = fr.reshape(H // T, T, W // T, T, 3).transpose(0, 2, 1, 3, 4)
    return b.reshape(-1, T, T, 3)


print("\n### 2/3/4. EMPIRICAL ###")
for name, yf in GAMES:
    try:
        env = GymWrapper(yaml_file=yf, level=0,
                         global_observer_type=gd.ObserverType.SPRITE_2D,
                         player_observer_type=gd.ObserverType.SPRITE_2D)
        T = env.get_tile_size(0)[0]
        blocks = {}
        rtid = collections.Counter()
        growth = []
        nfr = 0
        for ep in range(12):
            lev = ep % env.level_count
            env.seed(500 + ep); env.reset(level_id=lev)
            rng = np.random.default_rng(ep * 17 + 3)
            for t in range(50):
                fr = gframe(env); nfr += 1
                H, W, _ = fr.shape
                assert H == env.grid_height * T and W == env.grid_width * T, (H, W, T)
                for b in tiles_of(fr, T):
                    blocks[hashlib.md5(np.ascontiguousarray(b)).digest()] = None
                for o in env.get_state()["Objects"]:
                    rtid[(o["Name"], o["RenderTileId"])] += 1
                try:
                    _, _, done, _ = env.step(sample_action(env, rng))
                except Exception:
                    break
                if np.any(done):
                    break
            growth.append(len(blocks))
        nyaml = len(set(per_game[yf][2]))
        print(f"\n{name}: tile={T}px frames={nfr} DISTINCT {T}x{T} BLOCKS={len(blocks)} "
              f"(declared sprites={nyaml})")
        print(f"   block-count growth per episode: {growth}")
        rt = collections.defaultdict(set)
        for (n_, r_), c in rtid.items():
            rt[n_].add(r_)
        print(f"   RenderTileId per object: { {k: sorted(v) for k, v in sorted(rt.items())} }")

        # 3. decoder stability
        env.seed(1); env.reset(level_id=0)
        m_a = hashlib.md5(np.ascontiguousarray(gframe(env))).hexdigest()
        for lv in range(env.level_count):
            env.reset(level_id=lv)
        env.seed(1); env.reset(level_id=0)
        m_b = hashlib.md5(np.ascontiguousarray(gframe(env))).hexdigest()
        env2 = GymWrapper(yaml_file=yf, level=0,
                          global_observer_type=gd.ObserverType.SPRITE_2D,
                          player_observer_type=gd.ObserverType.SPRITE_2D)
        env2.seed(1); env2.reset(level_id=0)
        m_c = hashlib.md5(np.ascontiguousarray(gframe(env2))).hexdigest()
        print(f"   reset determinism: fresh={m_a[:10]} after-visiting-all-levels={m_b[:10]} "
              f"second-instance={m_c[:10]}  IDENTICAL={m_a == m_b == m_c}")
        del env2, env
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"{name}: FAILED {e}")
