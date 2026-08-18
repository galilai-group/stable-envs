"""Enumerate stable_worldmodel's registered gymnasium ids LIVE, and probe each for constructibility.

The registry is built dynamically at import time (stable_worldmodel/envs/__init__.py calls
gymnasium.envs.registration.register), and an optional-dependency block can silently skip a whole
family (the ale import is inside a try/except ImportError), so the only trustworthy list is the one
this process observes.
"""
import json
import sys
import traceback

import gymnasium
import stable_worldmodel as swm
from gymnasium.envs import registration

out = {}
ids = sorted(i for i in registration.registry if i.startswith("swm/"))
out["ids"] = ids
out["WORLDS"] = sorted(swm.envs.WORLDS)
out["DISCRETE_WORLDS"] = sorted(swm.envs.DISCRETE_WORLDS)
out["entry_points"] = {i: str(registration.registry[i].entry_point) for i in ids}
out["kwargs"] = {i: {k: str(v) for k, v in (registration.registry[i].kwargs or {}).items()} for i in ids}

probe = {}
for i in ids:
    rec = {}
    try:
        e = gymnasium.make(i, render_mode="rgb_array")
        rec["make"] = "ok"
        u = e.unwrapped
        rec["class"] = type(u).__name__
        rec["obs_space"] = str(e.observation_space)[:300]
        rec["act_space"] = str(e.action_space)[:300]
        try:
            vs = getattr(u, "variation_space", None)
            rec["variation_names"] = vs.names() if vs is not None else None
        except Exception as ex:
            rec["variation_names"] = f"ERR {ex}"
        try:
            o, info = e.reset(seed=0)
            rec["reset"] = "ok"
            rec["info_keys"] = sorted(info.keys()) if isinstance(info, dict) else str(type(info))
            f = e.render()
            rec["frame_shape"] = None if f is None else list(getattr(f, "shape", []))
        except Exception as ex:
            rec["reset"] = f"ERR {type(ex).__name__}: {ex}"[:400]
        e.close()
    except Exception as ex:
        rec["make"] = f"ERR {type(ex).__name__}: {ex}"[:400]
    probe[i] = rec
    print(i, rec.get("make"), rec.get("reset", ""), flush=True)

out["probe"] = probe
with open(sys.argv[1], "w") as fh:
    json.dump(out, fh, indent=1)
print("WROTE", sys.argv[1])
