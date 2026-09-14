import sys, os, json, numpy as np, mujoco, importlib.metadata as md
from dm_control import suite
out = sys.argv[1]
res = {"mujoco": mujoco.__version__, "dm_control": md.version("dm_control"), "gl": os.environ.get("MUJOCO_GL")}
frames = {}
for dom, task in (("reacher","hard"),("cheetah","run"),("hopper","hop"),("walker","walk"),("humanoid","stand")):
    env = suite.load(dom, task, task_kwargs={"random": 0}); env.reset()
    p = env.physics
    f = p.render(height=84, width=84, camera_id=0)
    frames[f"{dom}-{task}"] = f
    fi = f.astype(np.int32); h, w, _ = fi.shape
    q = [fi[:h//2,:w//2].mean(), fi[:h//2,w//2:].mean(), fi[h//2:,:w//2].mean(), fi[h//2:,w//2:].mean()]
    res[f"{dom}-{task}"] = {"quadrant_spread": round(max(q)-min(q),2), "mean": round(float(fi.mean()),3), "distinct": int(len(np.unique(f.reshape(-1,3),axis=0)))}
np.savez(out + ".npz", **frames)
json.dump(res, open(out + ".json", "w"), indent=1)
print(json.dumps(res))
