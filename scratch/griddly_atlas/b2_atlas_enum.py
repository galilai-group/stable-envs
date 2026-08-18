"""B2: ENUMERATE the atlas. What does baking orientation and view rotation actually cost?

docs/research/griddly.md predicts "**1.79 MB** with four orientations baked", derived as
207 referenced PNGs -> 448,000 B decoded RGBA, times four. That is an upper bound on a SPRITE atlas.
The thing a ``tilemap.blit`` path actually stores is the set of composed T x T output tiles, which is
a different and much smaller object: a wall sprite drawn over a background at three WALL_16 neighbour
codes is three tiles, and the same sprite at four orientations may well be one tile if it is
rotationally symmetric. Nobody has counted them, so this counts them.

Per game (all shipped GDYs that construct, Sprite2D observer, native TileSize read from the yaml):

  * distinct T x T blocks in the GLOBAL frame, with a saturation curve over frames;
  * distinct T x T blocks in the PLAYER frame for the 15 games with ``RotateWithAvatar``, which is
    where every view rotation actually appears;
  * the union, per game and across the family, deduplicated by pixel bytes so a sprite shared between
    two games is stored once;
  * total bytes at uint8 RGB.

Every frame is checked non-uniform before its blocks are counted.
"""
import os, sys, json, glob, collections
import numpy as np
import yaml as pyyaml
import griddly
from griddly import GymWrapper, gd

OUT = os.path.dirname(os.path.abspath(__file__))
GRIDDLY_DIR = os.path.dirname(griddly.__file__)
GAMES_DIR = os.path.join(GRIDDLY_DIR, "resources", "games")
N_FRAMES = 260


def sample_action(env, rng):
    sp = env.action_space
    if hasattr(sp, "nvec"):
        return [int(rng.integers(k)) for k in sp.nvec]
    if hasattr(sp, "n"):
        return int(rng.integers(sp.n))
    return [([int(rng.integers(k)) for k in s.nvec] if hasattr(s, "nvec") else int(rng.integers(s.n)))
            for s in sp]


def tiles_of(fr, T):
    h, w = fr.shape[0] // T, fr.shape[1] // T
    if h == 0 or w == 0:
        return []
    return [fr[j * T:(j + 1) * T, i * T:(i + 1) * T].tobytes() for j in range(h) for i in range(w)]


def run(rel):
    path = os.path.join(GAMES_DIR, rel)
    with open(path) as f:
        g = pyyaml.load(f, Loader=pyyaml.FullLoader)
    sp = ((g["Environment"].get("Observers") or {}).get("Sprite2D") or {})
    T = int(sp.get("TileSize", 24)) if not isinstance(sp.get("TileSize"), list) else 24
    plobs = ((g["Environment"].get("Player") or {}).get("Observer") or {})
    rotates = bool(plobs.get("RotateWithAvatar", False))
    env = GymWrapper(yaml_file=rel, level=0,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     player_observer_type=gd.ObserverType.SPRITE_2D)
    gl, pov = set(), set()
    curve, uniform, n = [], 0, 0
    ep = 0
    while n < N_FRAMES and ep < 200:
        env.seed(ep); env.reset(level_id=ep % env.level_count)
        rng = np.random.default_rng(7000 + ep)
        for t in range(60):
            fr = np.array(env.game.observe(), copy=True).swapaxes(0, 2)
            if len(np.unique(fr)) < 2:
                uniform += 1
            gl.update(tiles_of(fr, T))
            if rotates:
                pf = np.array(env._players[0].observe(), copy=True).swapaxes(0, 2)
                if len(np.unique(pf)) < 2:
                    uniform += 1
                pov.update(tiles_of(pf, T))
            n += 1
            if n in (25, 50, 100, 200, N_FRAMES):
                curve.append([n, len(gl), len(pov)])
            try:
                _, _, done, _ = env.step(sample_action(env, rng))
            except Exception:
                break
            if np.any(done) or n >= N_FRAMES:
                break
        ep += 1
    out = dict(game=rel, tile_size=T, rotates=rotates, frames=n, uniform_frames=uniform,
               n_levels=int(env.level_count), players=int(env.player_count),
               tiles_global=len(gl), tiles_pov=len(pov), tiles_union=len(gl | pov),
               bytes_union=len(gl | pov) * T * T * 3,
               pov_multiplier=(len(gl | pov) / len(gl)) if gl else None,
               saturation=curve)
    del env
    return out, (gl | pov)


if __name__ == "__main__":
    rels = sorted(os.path.relpath(p, GAMES_DIR)
                  for p in glob.glob(os.path.join(GAMES_DIR, "**", "*.yaml"), recursive=True))
    per, family = {}, set()
    fam_bytes = 0
    for rel in rels:
        try:
            r, tiles = run(rel)
            before = len(family)
            family |= tiles
            r["new_tiles_added_to_family"] = len(family) - before
            per[rel] = r
            fam_bytes = sum(len(t) for t in family)
            print(f"{rel:56s} T={r['tile_size']:2d} rot={int(r['rotates'])} "
                  f"global={r['tiles_global']:4d} pov={r['tiles_pov']:4d} "
                  f"union={r['tiles_union']:4d} mult={r['pov_multiplier']} "
                  f"family={len(family)} ({fam_bytes/1e6:.3f} MB)", flush=True)
        except Exception as e:                                       # noqa: BLE001
            per[rel] = dict(game=rel, error=f"{type(e).__name__}: {e}")
            print(f"{rel:56s} FAILED {type(e).__name__}: {e}", flush=True)
    summary = dict(
        n_games_ok=sum(1 for v in per.values() if "error" not in v),
        n_games_failed=sum(1 for v in per.values() if "error" in v),
        family_distinct_tiles=len(family),
        family_bytes=fam_bytes,
        family_MB=fam_bytes / 1e6,
        sum_per_game_tiles=sum(v.get("tiles_union", 0) for v in per.values() if "error" not in v),
        sum_per_game_bytes=sum(v.get("bytes_union", 0) for v in per.values() if "error" not in v),
        note_prediction_MB=1.79,
    )
    print(json.dumps(summary, indent=2), flush=True)
    with open(os.path.join(OUT, "b2_atlas_enum.json"), "w") as f:
        json.dump(dict(summary=summary, per_game=per), f, indent=2, default=str)
    print("WROTE", os.path.join(OUT, "b2_atlas_enum.json"), flush=True)
    sys.stdout.flush()
