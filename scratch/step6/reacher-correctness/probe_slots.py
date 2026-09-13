import os
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np, mujoco
import stable_envs as se
from stable_envs.envs import dmcontrol as D

for cname in list(D.DOMAINS) + list(D.TRACKCOM_DOMAINS):
    cls = se.REGISTRY[cname]
    m = cls._model()
    slots = cls._slots()
    els = []
    for arr, k in slots:
        kind, eid = D._slot_element(m, arr, k)
        obj = {"geom": mujoco.mjtObj.mjOBJ_GEOM, "site": mujoco.mjtObj.mjOBJ_SITE,
               "body": mujoco.mjtObj.mjOBJ_BODY, "light": mujoco.mjtObj.mjOBJ_LIGHT}[kind]
        els.append((kind, mujoco.mj_id2name(m, obj, eid), arr))
    print(f"{cname:22} slots={[D._slot_name(m,a,i) for a,i in slots]}  elements={sorted(set(els))}")
