"""Does MuJoCo's own renderer stop drawing a geom/site when we zero (a) its own rgba alpha,
(b) its material's alpha? Measured, not reasoned about."""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np, mujoco
from dm_control import suite

def probe(domain, task, kind, name):
    env = suite.load(domain, task, task_kwargs={"random": 0}); env.reset()
    p = env.physics; m = p.model.ptr if hasattr(p.model, "ptr") else p.model
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 84)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), 84)
    r = mujoco.Renderer(m, height=84, width=84)
    obj = mujoco.mjtObj.mjOBJ_GEOM if kind == "geom" else mujoco.mjtObj.mjOBJ_SITE
    i = mujoco.mj_name2id(m, obj, name)
    rgba = getattr(m, f"{kind}_rgba")
    mid = int(getattr(m, f"{kind}_matid")[i])
    def shot():
        r.update_scene(p.data.ptr if hasattr(p.data,"ptr") else p.data, camera=0, scene_option=mujoco.MjvOption())
        return r.render().copy()
    base = shot()
    keep = np.array(rgba[i], copy=True)
    rgba[i, 3] = 0.0
    own = shot()
    rgba[i] = keep
    keepm = np.array(m.mat_rgba[mid], copy=True)
    m.mat_rgba[mid, 3] = 0.0
    mat = shot()
    m.mat_rgba[mid] = keepm
    again = shot()
    d = lambda a, b: (int(np.abs(a.astype(int)-b.astype(int)).max()), int((np.abs(a.astype(int)-b.astype(int)).max(-1) > 0).sum()))
    print(f"{domain}-{task} {kind} {name!r} mat={mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mid)}")
    print(f"   own-alpha=0  vs base: max={d(own,base)[0]} changed_px={d(own,base)[1]}")
    print(f"   mat-alpha=0  vs base: max={d(mat,base)[0]} changed_px={d(mat,base)[1]}")
    print(f"   restored     vs base: max={d(again,base)[0]}  (must be 0)")

probe("reacher", "hard", "geom", "target")
probe("finger", "turn_hard", "site", "target")
