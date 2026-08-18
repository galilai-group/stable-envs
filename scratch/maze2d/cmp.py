import sys, json, numpy as np, hashlib
sys.path.insert(0, ".")
import stable_envs as se
from stable_envs.testing.conformance import frames_digest
d = np.load(sys.argv[1]); rep = json.load(open(sys.argv[2]))["envs"]
for name in ("PointMazeUMaze", "PointMazeOpen", "PointMazeMedium", "PointMazeLarge"):
    st = d[f"{name}::states"]; sty = d[f"{name}::style"]
    env = se.make(name, batch=8, device="cuda")
    a = frames_digest(env.render(st)); b = frames_digest(env.render(st, style=sty))
    print(name, "frames", a == rep[name]["frames_sha"], "styled", b == rep[name]["frames_styled_sha"])
