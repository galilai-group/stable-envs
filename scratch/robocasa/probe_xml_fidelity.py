"""Does a dumped ``get_xml()`` scene still *look* like the scene it came from?

``probe_batch.py`` checked that the dumped MJCF reloads standalone and has matching nq/ngeom, and
called that good enough to hand to another renderer. Shape agreement is not appearance agreement:
the MJWarp reference frame rendered from that same file came out garish and flat, exactly like the
textures-ablated image from ``probe_render_features.py``. That points at the material->texture
bindings surviving as *numbers* but not as *references*.

So the same model is rendered two ways at an identical qpos -- from the live robosuite env, and from
the dumped file -- and both the pixels and the binding tables (``mat_texid``, ``geom_matid``) are
compared. If the dump loses textures, every downstream "just export the scene and render it
elsewhere" plan is broken at the first step, and that is worth knowing precisely.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out"
XML = os.path.join(OUT, "pinned_scene.xml")
RES = 128


def bindings(m):
    texid = m.mat_texid
    n_mat_tex = int(np.sum(texid.max(axis=1) >= 0)) if texid.ndim > 1 else int(np.sum(texid >= 0))
    return {"ntex": int(m.ntex), "nmat": int(m.nmat),
            "texdata_MB": float(m.tex_data.size / 1e6),
            "n_mat_with_texture": n_mat_tex,
            "n_geom_with_material": int(np.sum(m.geom_matid >= 0)),
            "n_tex_named": int(sum(1 for i in range(m.ntex)
                                   if int(m.name_texadr[i]) >= 0
                                   and m.names[int(m.name_texadr[i]):].split(b"\0")[0] != b"")),
            "n_mesh_with_texcoord": int(np.sum(m.mesh_texcoordadr >= 0)),
            }


def main():
    import mujoco
    import imageio.v2 as imageio
    from robocasa.utils.env_utils import create_env

    rec = {}
    env = create_env(env_name="PickPlaceCounterToSink", robots="PandaOmron",
                     camera_names=["robot0_agentview_left"],
                     camera_widths=RES, camera_heights=RES, seed=0, render_onscreen=False)
    env.reset()
    live = env.sim.model._model
    d_live = env.sim.data._data
    qpos = np.load(os.path.join(OUT, "qpos_bank.npy"))[0]

    # the dumped file must describe *this* kitchen for the comparison to mean anything
    dumped = mujoco.MjModel.from_xml_path(XML)
    rec["live"] = bindings(live)
    rec["dumped"] = bindings(dumped)
    rec["same_shape"] = bool(dumped.nq == live.nq and dumped.ngeom == live.ngeom)
    print("live  :", json.dumps(rec["live"]))
    print("dumped:", json.dumps(rec["dumped"]))
    print("same shape:", rec["same_shape"])

    if not rec["same_shape"]:
        print("NOTE: dumped file is a different episode than the live env; "
              "comparing bindings only, not pixels.")
    cam = mujoco.mj_name2id(live, mujoco.mjtObj.mjOBJ_CAMERA, "robot0_agentview_left")

    def shot(m, qp):
        d = mujoco.MjData(m)
        d.qpos[:] = qp[:m.nq]
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, height=RES, width=RES)
        c = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "robot0_agentview_left")
        r.update_scene(d, camera=c)
        px = r.render().copy()
        r.close()
        return px

    a = shot(live, qpos)
    b = shot(dumped, qpos)
    imageio.imwrite(os.path.join(OUT, "fid_live.png"), a)
    imageio.imwrite(os.path.join(OUT, "fid_dumped.png"), b)
    if a.shape == b.shape:
        e = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
        rec["pixels"] = {"mean": float(e.mean()), "max": int(e.max()),
                         "exact": float((e == 0).mean()), "within8": float((e <= 8).mean())}
        print("live vs dumped pixels:", json.dumps(rec["pixels"]))
        imageio.imwrite(os.path.join(OUT, "fid_sheet.png"),
                        np.concatenate([a, np.zeros((RES, 3, 3), np.uint8), b], axis=1))

    with open(os.path.join(OUT, "xml_fidelity.json"), "w") as f:
        json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main()
