"""Can a MATERIAL alpha knob express visibility on every DM Control cell we ship?

For each parity-tier cell: list used materials, their wearers, and whether each wearer's
effective alpha is governed by the MATERIAL (geom_rgba is the compiler default) or by the
element's own rgba (in which case a material-alpha knob would be inert for it).
"""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np, mujoco
import stable_envs as se
from stable_envs.envs import dmcontrol as D

DEFAULT = np.array([0.5, 0.5, 0.5, 1.0])

for cname in D.DOMAINS:
    cls = se.REGISTRY[cname]
    m = cls._model(); pr = cls._pristine()
    used = cls._used_materials()
    print(f"\n=== {cname}  materials={used}")
    for kind in ("geom", "site"):
        n = m.ngeom if kind == "geom" else m.nsite
        rgba = np.asarray(getattr(m, f"{kind}_rgba"), float)
        matid = np.asarray(getattr(m, f"{kind}_matid"), int)
        grp = pr[f"{kind}_group"]
        alpha = D._effective_alpha(m, kind)
        for i in range(n):
            if int(grp[i]) not in D.VISIBLE_GROUPS or alpha[i] <= 0:
                continue
            name = mujoco.mj_id2name(m, getattr(mujoco.mjtObj, {"geom":"mjOBJ_GEOM","site":"mjOBJ_SITE"}[kind]), i)
            mid = int(matid[i])
            mn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mid) if mid >= 0 else None
            gov = "material" if (np.allclose(rgba[i], DEFAULT) and mid >= 0) else "own_rgba"
            print(f"   {kind} {name!r:16} mat={mn!r:14} rgba={np.round(rgba[i],2)} governed_by={gov}")
