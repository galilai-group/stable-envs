"""What does it actually take to reproduce one RoboCasa frame from a state vector?

Measures, on a real compiled kitchen, the three things the stable-envs thesis needs to be true:
  (a) the *pose* part of the state -- how many numbers move per step (nq/nv, mocap, object free joints);
  (b) the *scene* part -- how much of the picture is baked into the MjModel rather than mjData, and
      whether that part is constant across a reset (which is what would let it be a small "scene id");
  (c) the asset bill -- geoms, meshes, triangles, texture bytes -- because a GPU rasterizer has to
      carry all of it.

Nothing here is assumed from the RoboCasa paper: every number is read off the compiled mjModel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np


GEOM_TYPES = {
    0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid",
    5: "cylinder", 6: "box", 7: "mesh", 8: "sdf",
}
JNT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}


def model_stats(m):
    """Everything about the compiled model that a renderer would have to reproduce."""
    gt = Counter(GEOM_TYPES.get(int(t), str(t)) for t in m.geom_type)
    jt = Counter(JNT_TYPES.get(int(t), str(t)) for t in m.jnt_type)

    # visual geoms are the ones a camera sees: group 0/1 with nonzero rgba alpha in MuJoCo's
    # default visual flags. Collision-only geoms (group 3 in robosuite) never reach the pixels.
    grp = Counter(int(g) for g in m.geom_group)

    tex_bytes = int(m.tex_data.size) if hasattr(m, "tex_data") else int(m.ntexdata)
    tex_res = [(int(m.tex_width[i]), int(m.tex_height[i])) for i in range(m.ntex)]

    return {
        "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu),
        "nbody": int(m.nbody), "njnt": int(m.njnt), "ngeom": int(m.ngeom),
        "nsite": int(m.nsite), "nmocap": int(m.nmocap),
        "ncam": int(m.ncam), "nlight": int(m.nlight),
        "nmesh": int(m.nmesh), "nmeshvert": int(m.nmeshvert), "nmeshface": int(m.nmeshface),
        "ntex": int(m.ntex), "ntexdata_bytes": tex_bytes,
        "nmat": int(m.nmat), "nhfield": int(m.nhfield), "nskin": getattr(m, "nskin", 0),
        "geom_type_counts": dict(gt),
        "geom_group_counts": dict(grp),
        "jnt_type_counts": dict(jt),
        "n_free_joints": int(jt.get("free", 0)),
        "tex_resolutions_top10": sorted(set(tex_res), reverse=True)[:10],
        "tex_total_megapixels": sum(w * h for w, h in tex_res) / 1e6,
        "mesh_faces_top10": sorted((int(f) for f in m.mesh_facenum), reverse=True)[:10],
    }


def visual_triangles(m):
    """Triangles the camera actually rasterizes: mesh geoms in visual groups, plus primitives."""
    vis = [i for i in range(m.ngeom) if int(m.geom_group[i]) in (0, 1) and m.geom_rgba[i, 3] > 0]
    tris = 0
    n_mesh_vis = 0
    for i in vis:
        if int(m.geom_type[i]) == 7:  # mesh
            tris += int(m.mesh_facenum[int(m.geom_dataid[i])])
            n_mesh_vis += 1
    return {"n_visual_geoms": len(vis), "n_visual_mesh_geoms": n_mesh_vis,
            "visual_mesh_triangles": tris}


def xml_digest(env):
    x = env.sim.model.get_xml()
    return {"xml_chars": len(x), "xml_sha1": hashlib.sha1(x.encode()).hexdigest()[:16]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", default="PickPlaceCounterToSink")
    ap.add_argument("--layout", type=int, default=None)
    ap.add_argument("--style", type=int, default=None)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import mujoco
    import robosuite
    import robocasa
    from robocasa.utils.env_utils import create_env
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    rec = {"versions": {
        "mujoco": mujoco.__version__,
        "robosuite": robosuite.__version__,
        "robocasa": getattr(robocasa, "__version__", "editable"),
        "numpy": np.__version__,
    }}
    rec["layouts"] = {t.name: int(t.value) for t in LayoutType}
    rec["styles"] = {t.name: int(t.value) for t in StyleType}
    rec["n_layouts_real"] = sum(1 for t in LayoutType if t.value >= 0)
    rec["n_styles_real"] = sum(1 for t in StyleType if t.value >= 0)
    print(json.dumps({k: rec[k] for k in ("versions", "n_layouts_real", "n_styles_real")}, indent=2))

    kw = {}
    if args.layout is not None:
        kw["layout_ids"] = args.layout
    if args.style is not None:
        kw["style_ids"] = args.style

    t0 = time.time()
    env = create_env(
        env_name=args.env_name,
        robots="PandaOmron",
        camera_names=["robot0_agentview_left"],
        camera_widths=args.res, camera_heights=args.res,
        seed=0, render_onscreen=False, **kw,
    )
    rec["t_create_env_s"] = time.time() - t0

    t0 = time.time()
    env.reset()
    rec["t_first_reset_s"] = time.time() - t0

    m = env.sim.model._model
    rec["model"] = model_stats(m)
    rec["visual"] = visual_triangles(m)
    rec["xml"] = xml_digest(env)
    rec["layout_id"] = int(getattr(env, "layout_id", -999))
    rec["style_id"] = int(getattr(env, "style_id", -999))

    # --- what is actually in the state vs the model -------------------------------------------
    d = env.sim.data._data
    rec["state"] = {
        "qpos": int(m.nq), "qvel": int(m.nv),
        "mocap_pos": int(m.nmocap) * 3, "mocap_quat": int(m.nmocap) * 4,
        "act": int(m.na),
        "mjstate_full_floats": int(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)),
        "mjstate_integration_floats": int(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION)),
    }
    # objects vs fixtures: free joints are the movable objects
    free_jnt_bodies = [int(m.jnt_bodyid[j]) for j in range(m.njnt) if int(m.jnt_type[j]) == 0]
    names = []
    for b in free_jnt_bodies:
        adr = int(m.name_bodyadr[b])
        nm = m.names[adr:m.names.index(b"\0", adr)].decode()
        names.append(nm)
    rec["free_body_names"] = names

    # --- cameras and lights -------------------------------------------------------------------
    rec["cameras"] = []
    for i in range(m.ncam):
        adr = int(m.name_camadr[i])
        nm = m.names[adr:m.names.index(b"\0", adr)].decode()
        rec["cameras"].append({"name": nm, "pos": m.cam_pos[i].tolist(),
                               "fovy": float(m.cam_fovy[i]),
                               "mode": int(m.cam_mode[i]),
                               "bodyid": int(m.cam_bodyid[i])})
    rec["lights"] = [{"pos": m.light_pos[i].tolist(), "dir": m.light_dir[i].tolist(),
                      "directional": int(m.light_directional[i]) if hasattr(m, "light_directional") else -1,
                      "castshadow": int(m.light_castshadow[i]),
                      "diffuse": m.light_diffuse[i].tolist()} for i in range(m.nlight)]

    # --- is the frame a pure function of (qpos, qvel) given this model? ------------------------
    cam = "robot0_agentview_left"

    def render():
        return env.sim.render(width=args.res, height=args.res, camera_name=cam)[::-1].copy()

    s0 = env.sim.get_state().flatten().copy()
    f0 = render()
    for _ in range(20):
        env.step(np.zeros(env.action_dim))
    f_mid = render()
    # restore
    from robosuite.utils.binding_utils import MjSimState
    env.sim.set_state_from_flattened(s0)
    env.sim.forward()
    f1 = render()
    rec["purity"] = {
        "restore_same_model_maxdiff": int(np.abs(f0.astype(np.int16) - f1.astype(np.int16)).max()),
        "restore_same_model_meandiff": float(np.abs(f0.astype(np.int16) - f1.astype(np.int16)).mean()),
        "moved_frames_maxdiff": int(np.abs(f0.astype(np.int16) - f_mid.astype(np.int16)).max()),
        "flat_state_len": int(s0.size),
    }
    print("\npurity (restore qpos/qvel -> same pixels?):", json.dumps(rec["purity"], indent=2))

    # --- does reset() keep the same model? ----------------------------------------------------
    resets = []
    for k in range(4):
        t0 = time.time()
        env.reset()
        dt = time.time() - t0
        mm = env.sim.model._model
        resets.append({
            "t_reset_s": dt,
            "xml_sha1": xml_digest(env)["xml_sha1"],
            "xml_chars": xml_digest(env)["xml_chars"],
            "nq": int(mm.nq), "ngeom": int(mm.ngeom), "nmesh": int(mm.nmesh),
            "ntex": int(mm.ntex), "nmeshface": int(mm.nmeshface),
            "layout_id": int(getattr(env, "layout_id", -999)),
            "style_id": int(getattr(env, "style_id", -999)),
        })
        print(f"reset {k}: {resets[-1]}")
    rec["resets"] = resets
    rec["model_changes_across_reset"] = len({r["xml_sha1"] for r in resets}) > 1

    # --- the decisive test for "scene id + variation" -----------------------------------------
    # RoboCasa already has a candidate compact scene descriptor: ``_ep_meta``, the JSON blob it
    # serializes per episode (layout_id, style_id, object_cfgs, fixture_refs, gen_textures). If
    # replaying it rebuilds a byte-identical model, then the scene really does factor as
    # (small id) x (per-step qpos), and the only question is how small the id is. Measure both.
    env.reset()
    ep = env.get_ep_meta()
    ep_json = json.dumps(ep)
    ref_sha = xml_digest(env)["xml_sha1"]
    ref_qpos = env.sim.data.qpos.copy()
    f_ref = render()

    env.set_ep_meta(ep)
    env.reset()
    rep_sha = xml_digest(env)["xml_sha1"]
    same_nq = env.sim.model._model.nq == m.nq
    if same_nq:
        env.sim.data.qpos[:] = ref_qpos
        env.sim.forward()
        f_rep = render()
        px = int(np.abs(f_ref.astype(np.int16) - f_rep.astype(np.int16)).max())
        pxm = float(np.abs(f_ref.astype(np.int16) - f_rep.astype(np.int16)).mean())
    else:
        px, pxm = -1, -1.0

    rec["ep_meta"] = {
        "json_bytes": len(ep_json),
        "json_bytes_gzip": len(__import__("gzip").compress(ep_json.encode())),
        "top_keys": sorted(ep.keys()),
        "n_object_cfgs": len(ep.get("object_cfgs", []) or []),
        "layout_id": ep.get("layout_id"), "style_id": ep.get("style_id"),
        "replay_xml_matches": rep_sha == ref_sha,
        "replay_ref_sha": ref_sha, "replay_new_sha": rep_sha,
        "replay_same_nq": bool(same_nq),
        "replay_pixel_maxdiff": px, "replay_pixel_meandiff": pxm,
    }
    print("\nep_meta replay:", json.dumps(rec["ep_meta"], indent=2, default=str))
    with open(os.path.join(args.out, "ep_meta_example.json"), "w") as f:
        json.dump(ep, f, indent=2, default=str)

    with open(os.path.join(args.out, f"scene_{args.env_name}.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps({k: rec[k] for k in ("t_create_env_s", "t_first_reset_s", "model", "visual",
                                          "xml", "state", "model_changes_across_reset")}, indent=2))


if __name__ == "__main__":
    main()
