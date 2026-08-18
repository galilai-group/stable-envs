"""B0: smoke -- griddly renders headlessly, POV geometry, frame non-uniformity.

Prints, for each partially-observed game we will use: player observer config from the yaml,
POV frame shape, global frame shape, number of distinct byte values in each (must not be 1),
and whether get_state() exposes Orientation for the avatar.
"""
import os, json, collections
import numpy as np
import yaml as pyyaml
import griddly
from griddly import GymWrapper, gd

GAMES = {
    "sokoban2_po": "Single-Player/GVGAI/sokoban2_partially_observable.yaml",
    "zelda_po": "Single-Player/GVGAI/zelda_partially_observable.yaml",
    "labyrinth_po": "Single-Player/GVGAI/labyrinth_partially_observable.yaml",
    "clusters_po": "Single-Player/GVGAI/clusters_partially_observable.yaml",
    "cookmepasta_po": "Single-Player/GVGAI/cookmepasta_partially_observable.yaml",
    "drunkdwarf": "Single-Player/Mini-Grid/minigrid-drunkdwarf.yaml",
}
GAMES_DIR = os.path.join(os.path.dirname(griddly.__file__), "resources", "games")
OUT = os.path.dirname(os.path.abspath(__file__))

res = {}
for name, yf in GAMES.items():
    path = os.path.join(GAMES_DIR, yf)
    with open(path) as f:
        g = pyyaml.load(f, Loader=pyyaml.FullLoader)
    env_cfg = g["Environment"]
    obs = (env_cfg.get("Observers") or {}).get("Sprite2D", {}) or {}
    pl = (env_cfg.get("Player") or {}).get("Observer", {}) or {}
    env = GymWrapper(yaml_file=yf, level=0,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     player_observer_type=gd.ObserverType.SPRITE_2D)
    env.seed(0)
    env.reset(level_id=0)
    gfr = np.array(env.game.observe(), copy=True).swapaxes(0, 2)
    pfr = np.array(env._players[0].observe(), copy=True).swapaxes(0, 2)
    st = env.get_state()
    avatar = [o for o in st["Objects"] if o["Name"] == (env.gdy.get_avatar_object() or "")]
    res[name] = dict(
        yaml=yf, tile_size=obs.get("TileSize"), background=obs.get("BackgroundTile"),
        player_observer=pl, n_levels=int(env.level_count),
        global_shape=list(gfr.shape), pov_shape=list(pfr.shape),
        global_distinct_bytes=int(len(np.unique(gfr))),
        pov_distinct_bytes=int(len(np.unique(pfr))),
        avatar_object=env.gdy.get_avatar_object(),
        avatar_state=avatar[:1],
        state_keys=sorted(st.keys()),
        object_keys=sorted(st["Objects"][0].keys()) if st["Objects"] else [],
        action_names=list(env.action_names),
        action_space=str(env.action_space),
    )
    print(name, json.dumps(res[name], default=str)[:600])
    del env

with open(os.path.join(OUT, "b0_smoke.json"), "w") as f:
    json.dump(res, f, indent=2, default=str)
print("WROTE", os.path.join(OUT, "b0_smoke.json"))
