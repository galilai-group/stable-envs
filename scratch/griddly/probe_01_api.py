"""Probe 1: can we build a griddly env headlessly and get a non-degenerate rgb frame?

Also dumps the python API surface (gd module, ObserverType members, get_state payload).
"""
import os, sys, time, json, hashlib
import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

t0 = time.time()
import griddly
from griddly import GymWrapper, gd
print(f"[import] griddly {griddly.__file__} in {time.time()-t0:.2f}s")

print("[gd] dir:", [a for a in dir(gd) if not a.startswith("_")])
print("[ObserverType]", [a for a in dir(gd.ObserverType) if not a.startswith("_")])

GAMES = [
    "Single-Player/GVGAI/sokoban.yaml",
    "Single-Player/GVGAI/clusters.yaml",
    "Single-Player/GVGAI/zelda.yaml",
    "Single-Player/GVGAI/butterflies.yaml",
    "RTS/GriddlyRTS.yaml",
]


def frame_stats(a):
    a = np.asarray(a)
    return dict(shape=list(a.shape), dtype=str(a.dtype), min=int(a.min()), max=int(a.max()),
                n_unique=int(len(np.unique(a))),
                uniform=bool(a.min() == a.max()),
                md5=hashlib.md5(np.ascontiguousarray(a)).hexdigest()[:12])


for g in GAMES:
    print("=" * 70)
    print("GAME", g)
    try:
        t = time.time()
        env = GymWrapper(yaml_file=g, level=0,
                         global_observer_type=gd.ObserverType.SPRITE_2D,
                         player_observer_type=gd.ObserverType.SPRITE_2D)
        t_ctor = time.time() - t
        t = time.time()
        obs = env.reset()
        t_reset = time.time() - t
        print(f"  ctor {t_ctor*1000:.1f} ms  reset {t_reset*1000:.1f} ms")
        print("  grid", env.grid_width, "x", env.grid_height,
              "players", env.player_count, "levels", env.level_count,
              "tile_size", env.get_tile_size(0))
        print("  player obs", frame_stats(obs))
        gf = env.render(mode="rgb_array", observer="global")
        print("  global render", frame_stats(gf))
        st = env.get_state()
        print("  state keys:", list(st.keys()) if isinstance(st, dict) else type(st))
        if isinstance(st, dict):
            for k, v in st.items():
                if k == "Objects":
                    print(f"    Objects: n={len(v)}  first={v[0] if v else None}")
                else:
                    print(f"    {k}: {v}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("  FAILED:", e)
print("done")
