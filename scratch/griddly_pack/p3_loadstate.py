"""P3: what exactly does griddly's ``load_state`` accept? Printed, not guessed."""
import copy
import json
import os
import sys

import numpy as np
import griddly
from griddly import GymWrapper, gd

GAMES_DIR = os.path.join(os.path.dirname(griddly.__file__), "resources", "games")
KEEP = []


def frame(env):
    return np.array(env.game.observe(), copy=True).swapaxes(0, 2)


env = GymWrapper(yaml_file="Single-Player/GVGAI/sokoban.yaml", level=0,
                 global_observer_type=gd.ObserverType.SPRITE_2D,
                 player_observer_type=gd.ObserverType.SPRITE_2D)
env.seed(0)
env.reset(level_id=0)
st = env.get_state()
print("STATE KEYS:", list(st))
print("OBJECT[0]:", json.dumps(st["Objects"][0], default=str))
print("N OBJECTS:", len(st["Objects"]))
print("REST:", json.dumps({k: v for k, v in st.items() if k != "Objects"}, default=str)[:600])

# 1. identity round trip
e1 = env.load_state(copy.deepcopy(st))
KEEP.append(e1)
d = np.abs(frame(e1).astype(int) - frame(env).astype(int)).max()
print("IDENTITY load_state max_abs:", int(d))

# 2. drop one object
st2 = copy.deepcopy(st)
box = [i for i, o in enumerate(st2["Objects"]) if o["Name"] == "box"]
st2["Objects"] = [o for i, o in enumerate(st2["Objects"]) if i != box[0]]
try:
    e2 = env.load_state(st2)
    KEEP.append(e2)
    print("DROP-ONE ok, changed pixels:",
          int((np.abs(frame(e2).astype(int) - frame(env).astype(int)) > 0).sum()))
except Exception as e:                                              # noqa: BLE001
    print("DROP-ONE FAILED:", type(e).__name__, e)

# 3. move one object
st3 = copy.deepcopy(st)
st3["Objects"][box[0]]["Location"] = [6, 6]
try:
    e3 = env.load_state(st3)
    KEEP.append(e3)
    print("MOVE-ONE ok, changed pixels:",
          int((np.abs(frame(e3).astype(int) - frame(env).astype(int)) > 0).sum()))
except Exception as e:                                              # noqa: BLE001
    print("MOVE-ONE FAILED:", type(e).__name__, e)

# 4. keep ONLY the underscore-free objects (what a rebuilt state would look like)
for drop_underscore in (True, False):
    st4 = copy.deepcopy(st)
    st4["Objects"] = [o for o in st["Objects"]
                      if not (drop_underscore and o["Name"].startswith("_"))]
    try:
        e4 = env.load_state(copy.deepcopy(st4))
        KEEP.append(e4)
        print(f"DROP_UNDERSCORE={drop_underscore} ok, max_abs:",
              int(np.abs(frame(e4).astype(int) - frame(env).astype(int)).max()),
              "n_objects", len(st4["Objects"]))
    except Exception as e:                                          # noqa: BLE001
        print(f"DROP_UNDERSCORE={drop_underscore} FAILED:", type(e).__name__, e)

# 5. rebuild the object dicts from scratch, one key set at a time
proto = {o["Name"]: o for o in st["Objects"]}
keysets = {
    "full": lambda o: dict(o),
    "no_variables": lambda o: {k: v for k, v in o.items() if k != "Variables"},
    "minimal": lambda o: {"Name": o["Name"], "Location": list(o["Location"]),
                          "Orientation": o["Orientation"], "PlayerId": o["PlayerId"],
                          "RenderTileId": o["RenderTileId"]},
}
for tag, fn in keysets.items():
    st5 = copy.deepcopy(st)
    st5["Objects"] = [fn(o) for o in st["Objects"]]
    try:
        e5 = env.load_state(st5)
        KEEP.append(e5)
        print(f"REBUILD[{tag}] ok, max_abs:",
              int(np.abs(frame(e5).astype(int) - frame(env).astype(int)).max()))
    except Exception as e:                                          # noqa: BLE001
        print(f"REBUILD[{tag}] FAILED:", type(e).__name__, e)

# 6. a state with only a handful of objects at chosen cells
for keep_underscore in (True, False):
    st6 = copy.deepcopy(st)
    objs = []
    if keep_underscore:
        objs += [copy.deepcopy(o) for o in st["Objects"] if o["Name"].startswith("_")]
    for name, (x, y) in (("wall", (4, 4)), ("box", (7, 4)), ("avatar", (10, 4))):
        o = copy.deepcopy(proto[name])
        o["Location"] = [x, y]
        objs.append(o)
    st6["Objects"] = objs
    try:
        e6 = env.load_state(st6)
        KEEP.append(e6)
        got = {tuple(o["Location"]): o["Name"] for o in e6.get_state()["Objects"]
               if not o["Name"].startswith("_")}
        print(f"SPARSE[keep_underscore={keep_underscore}] ok, cells:", got)
    except Exception as e:                                          # noqa: BLE001
        print(f"SPARSE[keep_underscore={keep_underscore}] FAILED:", type(e).__name__, e)

# 7. orientation and RenderTileId honoured?
zel = GymWrapper(yaml_file="Single-Player/GVGAI/zelda.yaml", level=0,
                 global_observer_type=gd.ObserverType.SPRITE_2D,
                 player_observer_type=gd.ObserverType.SPRITE_2D)
KEEP.append(zel)
zel.seed(0)
zel.reset(level_id=0)
zst = zel.get_state()
zproto = {o["Name"]: o for o in zst["Objects"]}
print("ZELDA object names:", sorted(zproto))
for ori in ("UP", "RIGHT", "DOWN", "LEFT"):
    for rt in (0, 1):
        s = copy.deepcopy(zst)
        objs = [copy.deepcopy(o) for o in zst["Objects"] if o["Name"].startswith("_")]
        a = copy.deepcopy(zproto["avatar"])
        a["Location"] = [4, 4]
        a["Orientation"] = ori
        a["RenderTileId"] = rt
        objs.append(a)
        s["Objects"] = objs
        try:
            e = zel.load_state(s)
            KEEP.append(e)
            back = [o for o in e.get_state()["Objects"] if o["Name"] == "avatar"]
            fr = frame(e)
            print(f"ZELDA ori={ori} rt={rt} -> back={back[0]['Orientation'] if back else None}"
                  f"/{back[0]['RenderTileId'] if back else None} "
                  f"tile_sha={hash(fr[4*24:5*24, 4*24:5*24].tobytes()) & 0xffffff:06x}")
        except Exception as e:                                      # noqa: BLE001
            print(f"ZELDA ori={ori} rt={rt} FAILED:", type(e).__name__, e)

sys.stdout.flush()
os._exit(0)
