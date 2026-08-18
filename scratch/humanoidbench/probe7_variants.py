"""Three renderers, one reference: give the port its best shot before deciding a tier.

probe5 says two things are undrawable on this family, and they are the same two Adroit died on:

  * **alpha** -- ``generated_xml_hurdles.xml`` gives every barrier ``rgba="0.1 0.1 0.1 0.1"``, i.e.
    BLACK AT ALPHA 0.1. MuJoCo blends them to a faint tint over the floor; MJWarp reads only
    ``color[0..2]`` and paints them opaque black, which is the black wedge over 62.8 % of a hurdle
    frame.
  * **the floor's texture** -- a 2 cm ``mark="edge"`` grid line every 2 m, minified. MuJoCo filters
    it, MJWarp point-samples it, and the floor comes out dithered.

Both have a cheap mitigation with a precedent in this repo, and both cost something. Measured here
against the SAME official frame, so the three columns are comparable:

  V0  as-is
  V1  translucent geoms hidden on OUR side only (``fetch.py`` does this for its 4 mocap markers and
      declares the cost in ``parity_note``)
  V2  V1 + the floor texture replaced by its DOMINANT texel on our side only
      (``mjwarp.bake_primitive_textures`` is the same idea, and is explicitly a visual-tier tool)
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/humanoidbench")

from hb_common import (LOGS, VISIBLE_GROUPS, MjwRenderer, agreement, jsave, make_env, mj_render,
                       resolved_model, sample_qpos)
from stable_envs.render.mjwarp import HIDDEN_GROUP, effective_alpha
from stable_envs.testing.conformance import ssim

SIZE = 256
NSTATE = 16


def score(got, ref):
    a = agreement(got, ref)
    a["ssim"] = round(ssim(got, ref), 6)
    return a


def main():
    task, robot = sys.argv[1], sys.argv[2]
    import mujoco
    res = {"task": task, "robot": robot}
    try:
        env, u = make_env(task, robot, SIZE)
        cam_name, path = u.camera_name, u.fullpath
        Q = sample_qpos(task, robot, NSTATE, seed=0, steps=60)
        env.close()

        m, _ = resolved_model(path, SIZE)
        d = mujoco.MjData(m)
        cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        ref = mj_render(m, d, cam, SIZE, Q, tag="ref")

        alpha = effective_alpha(m, "geom")
        trans = [g for g in range(m.ngeom)
                 if 0.0 < alpha[g] < 1.0 and int(m.geom_group[g]) in VISIBLE_GROUPS]
        res["n_translucent"] = len(trans)

        R = MjwRenderer(m, cam, SIZE, NSTATE)
        res["V0"] = score(R.render(Q), ref)
        del R

        keep = np.array(m.geom_group, copy=True)
        if trans:
            m.geom_group[trans] = HIDDEN_GROUP
        R = MjwRenderer(m, cam, SIZE, NSTATE)
        res["V1_hide_translucent"] = score(R.render(Q), ref)
        del R

        # the floor's texture -> its dominant texel, on our side only
        texids = sorted({int(np.max(m.mat_texid[i])) if np.ndim(m.mat_texid[i]) else int(m.mat_texid[i])
                         for i in range(m.nmat)} - {-1})
        tex2d = [t for t in texids if int(m.tex_type[t]) == int(mujoco.mjtTexture.mjTEXTURE_2D)]
        saved = {}
        for t in tex2d:
            adr = int(m.tex_adr[t])
            w, h, c = int(m.tex_width[t]), int(m.tex_height[t]), int(m.tex_nchannel[t])
            n = w * h * c
            old = np.array(m.tex_data[adr:adr + n], copy=True)
            saved[t] = old
            px = old.reshape(-1, c)
            vals, cnt = np.unique(px, axis=0, return_counts=True)
            m.tex_data[adr:adr + n] = np.tile(vals[int(cnt.argmax())], (w * h, 1)).reshape(-1)
        R = MjwRenderer(m, cam, SIZE, NSTATE)
        res["V2_hide_and_flat_floor"] = score(R.render(Q), ref)
        del R
        for t, old in saved.items():
            adr = int(m.tex_adr[t])
            m.tex_data[adr:adr + old.size] = old
        m.geom_group[:] = keep
    except Exception as e:                                              # pragma: no cover
        res["error"] = f"{type(e).__name__}: {e}"
    jsave(f"{LOGS}/probe7_{robot}_{task}.json", res)
    print(res.get("error") or {k: (v["interior_exact"], v["mean_abs"], v["ssim"])
                               for k, v in res.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
