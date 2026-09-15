"""O1: does ``stable_worldmodel``'s mjcf ROUND-TRIP compile to the same render model as ``suite.load``?

Gate A S4.2 requires the certified configuration to be the registered SWM default, and this cell is
certified against ``suite.load`` with the SWM default reached THROUGH LEAVES (coordinator ruling).
That argument is only sound if the two compiled models are the same scene up to the leaves, and SWM
does NOT call ``suite.load``: it calls ``manipulator.make_model``, rewrites the asset paths,
round-trips the XML through ``dm_control.mjcf`` (``from_xml_string`` -> ``export_with_assets`` ->
``Physics.from_xml_path``) and builds ``Bring`` itself. An mjcf round-trip may rename or reorder
elements and re-emit defaults, and ``_episode_slots`` / ``_movers`` / ``_prims`` all key on model
ORDER and on ``mj_id2name``.

This compares the RENDER-MODEL SIGNATURE of the two, after a reset, the way Step 1 defined it:
geoms (name, type, group, matid, size, bodyid, rgba), sites, materials (rgba, reflectance, specular,
shininess, texid), textures (type, size, and a hash of the bytes), cameras (name, mode, fovy, pos,
quat), lights (type, mode, castshadow, pos, dir, ambient/diffuse/specular, cutoff, exponent), the
visual/quality block, and the per-episode draw discovered by diffing resets.

RUN ON OSCAR, in a venv that has ``stable_worldmodel`` (``$SCR/venvs/se-step5``), because
``stable_worldmodel`` is not installed on the Mac. It needs no GPU and no renderer -- it compiles two
models and diffs arrays -- so it is a login-node-safe script by the letter of the SLURM rules, but it
is run under ``--partition=batch --qos=normal`` by ``oscar_o1_signature.sbatch`` anyway.

    python scratch/step7/dmc-manipulator/probe_swm_signature.py --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

EPISODE_ARRAYS = ("geom_pos", "geom_quat", "geom_size", "site_pos", "site_quat", "site_size",
                  "body_pos", "body_quat", "light_pos", "light_dir")
N_DRAWS = 6


def nm(m, obj, i):
    import mujoco
    return mujoco.mj_id2name(m, obj, i) or f"<unnamed {obj} {i}>"


def signature(m) -> dict:
    import mujoco
    O = mujoco.mjtObj
    f = lambda a: [[float(x) for x in np.atleast_1d(r)] for r in np.asarray(a)]   # noqa: E731
    tex = []
    attr = "tex_rgb" if hasattr(m, "tex_rgb") else "tex_data"
    for t in range(int(m.ntex)):
        a0, h, w = int(m.tex_adr[t]), int(m.tex_height[t]), int(m.tex_width[t])
        raw = np.asarray(getattr(m, attr)[a0:a0 + 3 * h * w], np.uint8).tobytes()
        tex.append(dict(id=t, name=nm(m, O.mjOBJ_TEXTURE, t), type=int(m.tex_type[t]),
                        h=h, w=w, sha=hashlib.sha256(raw).hexdigest()[:16]))
    return dict(
        counts=dict(nq=int(m.nq), nv=int(m.nv), nbody=int(m.nbody), ngeom=int(m.ngeom),
                    nsite=int(m.nsite), nmat=int(m.nmat), ntex=int(m.ntex), nlight=int(m.nlight),
                    ncam=int(m.ncam), ntendon=int(m.ntendon), nmesh=int(m.nmesh),
                    nhfield=int(m.nhfield)),
        geoms=[dict(id=i, name=nm(m, O.mjOBJ_GEOM, i), type=int(m.geom_type[i]),
                    group=int(m.geom_group[i]), matid=int(m.geom_matid[i]),
                    bodyid=int(m.geom_bodyid[i]), body=nm(m, O.mjOBJ_BODY, int(m.geom_bodyid[i])),
                    size=[float(x) for x in m.geom_size[i]],
                    rgba=[float(x) for x in m.geom_rgba[i]]) for i in range(int(m.ngeom))],
        sites=[dict(id=i, name=nm(m, O.mjOBJ_SITE, i), type=int(m.site_type[i]),
                    group=int(m.site_group[i]), matid=int(m.site_matid[i]),
                    bodyid=int(m.site_bodyid[i]), size=[float(x) for x in m.site_size[i]],
                    rgba=[float(x) for x in m.site_rgba[i]]) for i in range(int(m.nsite))],
        materials=[dict(id=i, name=nm(m, O.mjOBJ_MATERIAL, i),
                        rgba=[float(x) for x in m.mat_rgba[i]],
                        reflectance=float(m.mat_reflectance[i]),
                        specular=float(m.mat_specular[i]), shininess=float(m.mat_shininess[i]),
                        texid=[int(x) for x in np.atleast_1d(m.mat_texid[i])])
                   for i in range(int(m.nmat))],
        textures=tex,
        cameras=[dict(id=i, name=nm(m, O.mjOBJ_CAMERA, i), mode=int(m.cam_mode[i]),
                      fovy=float(m.cam_fovy[i]), pos=[float(x) for x in m.cam_pos[i]],
                      quat=[float(x) for x in m.cam_quat[i]]) for i in range(int(m.ncam))],
        lights=[dict(id=i, name=nm(m, O.mjOBJ_LIGHT, i),
                     type=(int(m.light_type[i]) if hasattr(m, "light_type") else None),
                     directional=(int(m.light_directional[i])
                                  if hasattr(m, "light_directional") else None),
                     mode=int(m.light_mode[i]), castshadow=int(m.light_castshadow[i]),
                     pos=[float(x) for x in m.light_pos[i]],
                     dir=[float(x) for x in m.light_dir[i]],
                     ambient=[float(x) for x in m.light_ambient[i]],
                     diffuse=[float(x) for x in m.light_diffuse[i]],
                     specular=[float(x) for x in m.light_specular[i]],
                     cutoff=float(m.light_cutoff[i]), exponent=float(m.light_exponent[i]))
                for i in range(int(m.nlight))],
        visual=dict(shadowsize=int(m.vis.quality.shadowsize),
                    offsamples=int(m.vis.quality.offsamples),
                    numslices=int(m.vis.quality.numslices),
                    shadowclip=float(m.vis.map.shadowclip),
                    headlight_active=int(m.vis.headlight.active),
                    headlight_ambient=[float(x) for x in m.vis.headlight.ambient],
                    headlight_diffuse=[float(x) for x in m.vis.headlight.diffuse],
                    headlight_specular=[float(x) for x in m.vis.headlight.specular]),
        _f=f,
    )


def episode_draw(reset, m) -> list:
    have = [k for k in EPISODE_ARRAYS if hasattr(m, k)]
    draws = []
    for _ in range(N_DRAWS):
        reset()
        draws.append({k: np.array(getattr(m, k), copy=True).reshape(-1) for k in have})
    out = []
    for k in have:
        st = np.stack([d[k] for d in draws]).astype(np.float64)
        for i in np.flatnonzero(st.max(0) - st.min(0) > 0):
            out.append(f"{k}[{int(i)}]")
    return out


def prune(sig):
    sig = dict(sig)
    sig.pop("_f", None)
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="examples/results/step7/dmc-manipulator/oscar")
    a = ap.parse_args()
    import mujoco
    from importlib.metadata import version as _v

    out = {"mujoco": mujoco.__version__, "dm_control": _v("dm_control")}

    # --- A: suite.load, what this repo certifies against -------------------
    from dm_control import suite
    envA = suite.load("manipulator", "bring_ball", task_kwargs={"random": 0})
    pA = envA.physics
    mA = pA.model.ptr if hasattr(pA.model, "ptr") else pA.model
    out["suite_load"] = prune(signature(mA))
    out["suite_load"]["episode_draw"] = episode_draw(envA.reset, mA)

    # --- B: stable_worldmodel's registered wrapper -------------------------
    try:
        from stable_worldmodel.envs.dmcontrol.manipulator import ManipulatorDMControlWrapper
        w = ManipulatorDMControlWrapper(seed=0)
        out["swm"] = {"camera_id": int(w.camera_id), "action_repeat": int(w.action_repeat),
                      "variation_leaves": sorted(w.variation_space.names())
                      if hasattr(w.variation_space, "names") else None}
        # BEFORE any reset: the compiled model as the wrapper constructs it
        pB = w.env.physics
        mB = pB.model.ptr if hasattr(pB.model, "ptr") else pB.model
        out["swm_as_constructed"] = prune(signature(mB))
        # AFTER a reset: modify_mjcf_model has run and may have recompiled
        w.reset(seed=0)
        pB = w.env.physics
        mB = pB.model.ptr if hasattr(pB.model, "ptr") else pB.model
        out["swm_after_reset"] = prune(signature(mB))
        out["swm_after_reset"]["episode_draw"] = episode_draw(lambda: w.reset(seed=None), mB)
        out["swm_frame_shape"] = list(np.asarray(w.render()).shape)
    except Exception as e:                                           # noqa: BLE001
        out["swm_error"] = f"{type(e).__name__}: {e}"

    # --- the diff, field by field -----------------------------------------
    if "swm_after_reset" in out:
        A, B = out["suite_load"], out["swm_after_reset"]
        d = {}
        for key in ("counts", "visual"):
            d[key] = {k: [A[key].get(k), B[key].get(k)]
                      for k in set(A[key]) | set(B[key]) if A[key].get(k) != B[key].get(k)}
        for key in ("geoms", "sites", "materials", "textures", "cameras", "lights"):
            if len(A[key]) != len(B[key]):
                d[key] = f"length {len(A[key])} vs {len(B[key])}"
                continue
            rows = []
            for x, y in zip(A[key], B[key]):
                fields = [f for f in set(x) | set(y) if x.get(f) != y.get(f)]
                if fields:
                    rows.append({"id": x.get("id"), "name_A": x.get("name"),
                                 "name_B": y.get("name"),
                                 "fields": {f: [x.get(f), y.get(f)] for f in fields}})
            if rows:
                d[key] = rows
        d["episode_draw"] = [A.get("episode_draw"), B.get("episode_draw")]
        d["order_and_names_identical"] = all(
            [x.get("name") for x in A[k]] == [y.get("name") for y in B[k]]
            for k in ("geoms", "sites", "materials", "cameras", "lights")
            if len(A[k]) == len(B[k]))
        out["diff"] = d
        print(json.dumps({k: v for k, v in d.items() if v}, indent=1, default=str)[:6000])

    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "o1_swm_signature.json")
    json.dump(out, open(p, "w"), indent=1, default=str)
    print("wrote", p)


if __name__ == "__main__":
    main()
