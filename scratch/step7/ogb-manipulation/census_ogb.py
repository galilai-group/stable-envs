"""Step 7 / W5 phase 0 -- render census for swm/OGBCube-v0 and swm/OGBScene-v0.

Constructs the two swm environments exactly as `stable_worldmodel.envs` registers them
(no arguments: the registration passes none), then reports every fact a renderer has to
reproduce: geoms by mjtGeom, mesh triangle counts and convexity, textures, materials,
lights, cameras, the qpos/qvel/mocap contract, and the per-episode edits.

Nothing here writes into the package tree; the output is one JSON per env.

    PYTHONPATH=/Users/samib/stable-worldmodel MUJOCO_GL=glfw \
      python scratch/step7/ogb-manipulation/census_ogb.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
from pathlib import Path

import numpy as np

GEOM_NAMES = {
    0: "PLANE", 1: "HFIELD", 2: "SPHERE", 3: "CAPSULE", 4: "ELLIPSOID",
    5: "CYLINDER", 6: "BOX", 7: "MESH", 8: "SDF",
}
LIGHT_TYPE = {0: "SPOT", 1: "DIRECTIONAL", 2: "POINT", 3: "IMAGE"}
CAM_MODE = {0: "FIXED", 1: "TRACK", 2: "TRACKCOM", 3: "TARGETBODY", 4: "TARGETBODYCOM"}
TEX_TYPE = {0: "2D", 1: "CUBE", 2: "SKYBOX"}


def make(env_id):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  (registers swm/*)
    env = gym.make(env_id)
    return env


def mesh_stats(m):
    """Per-mesh vertex/face counts plus a convexity test against the mesh's own hull."""
    import mujoco
    rows = []
    for i in range(m.nmesh):
        nv, nf = int(m.mesh_vertnum[i]), int(m.mesh_facenum[i])
        va, fa = int(m.mesh_vertadr[i]), int(m.mesh_faceadr[i])
        verts = m.mesh_vert[va:va + nv].reshape(-1, 3)
        try:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, i) or f"mesh_{i}"
        except Exception:
            name = f"mesh_{i}"
        row = {"id": i, "name": name, "nvert": nv, "nface": nf,
               "bbox": (verts.max(0) - verts.min(0)).tolist()}
        # is this mesh its own convex hull?  compare face count to the hull of its verts
        try:
            from scipy.spatial import ConvexHull
            h = ConvexHull(verts)
            row["hull_nvert"] = int(len(h.vertices))
            row["hull_nface"] = int(len(h.simplices))
            row["is_convex_hull"] = bool(len(h.vertices) == nv)
        except Exception as e:
            row["hull_error"] = f"{type(e).__name__}"
        rows.append(row)
    return rows


def census(env_id, out_dir):
    import mujoco
    env = make(env_id)
    u = env.unwrapped
    ob, info = env.reset(seed=0)
    m, d = u.model, u.data

    rec = {
        "env_id": env_id,
        "class": f"{type(u).__module__}.{type(u).__name__}",
        "mro": [f"{c.__module__}.{c.__name__}" for c in type(u).__mro__[:6]],
        "host": socket.gethostname(), "platform": platform.platform(),
        "mujoco": mujoco.__version__, "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "numpy": np.__version__, "python": sys.version.split()[0],
        "ob_type": getattr(u, "_ob_type", None),
        "env_type": getattr(u, "_env_type", None),
        "num_objects": getattr(u, "_num_cubes", None),
        "mode": getattr(u, "_mode", None),
        "visualize_info": getattr(u, "_visualize_info", None),
        "pixel_transparent_arm": getattr(u, "_pixel_transparent_arm", None),
        "render_h": getattr(u, "_render_height", None),
        "render_w": getattr(u, "_render_width", None),
        "offwidth": int(m.vis.global_.offwidth), "offheight": int(m.vis.global_.offheight),
        "render_default_camera": "front" if getattr(u, "_ob_type", None) == "states" else "front_pixels",
        "cameras_declared": {k: {kk: list(vv) for kk, vv in v.items()} for k, v in getattr(u, "cameras", {}).items()},
        "observation_space": str(env.observation_space),
        "observation_shape": list(np.asarray(ob).shape),
        "observation_dtype": str(np.asarray(ob).dtype),
        "reset_info_keys": sorted(info.keys()),
        "goal_in_info": "goal" in info,
        "goal_rendered_in_info": info.get("goal_rendered") is not None,
        "render_goal_flag": getattr(u, "_render_goal", None),
        "variation_space": sorted(_flat_variation(getattr(u, "variation_space", None))),
    }

    # ---- model-level census -------------------------------------------------
    rec["counts"] = {k: int(getattr(m, k)) for k in
                     ("nq", "nv", "nu", "na", "nbody", "ngeom", "nmesh", "nmat", "ntex",
                      "nlight", "ncam", "nsite", "njnt", "nmocap", "nhfield", "nskin",
                      "ntexdata", "nmeshvert", "nmeshface")}

    gt = {}
    for i in range(m.ngeom):
        t = GEOM_NAMES.get(int(m.geom_type[i]), str(int(m.geom_type[i])))
        gt[t] = gt.get(t, 0) + 1
    rec["geom_type_counts"] = gt

    # visible geoms only (rgba alpha > 0 and group rendered)
    vis = {}
    for i in range(m.ngeom):
        if m.geom_rgba[i][3] <= 0.0:
            continue
        t = GEOM_NAMES.get(int(m.geom_type[i]), str(int(m.geom_type[i])))
        vis[t] = vis.get(t, 0) + 1
    rec["geom_type_counts_alpha_gt0"] = vis

    rec["geom_group_counts"] = {}
    for i in range(m.ngeom):
        g = int(m.geom_group[i])
        rec["geom_group_counts"][str(g)] = rec["geom_group_counts"].get(str(g), 0) + 1

    # meshes
    meshes = mesh_stats(m)
    rec["meshes"] = meshes
    rec["mesh_total_tri"] = int(sum(r["nface"] for r in meshes))
    rec["mesh_total_vert"] = int(sum(r["nvert"] for r in meshes))
    rec["mesh_all_convex_hulls"] = bool(all(r.get("is_convex_hull") for r in meshes)) if meshes else None

    # triangles actually instantiated (a mesh used by k geoms costs k x nface)
    inst = 0
    per_geom_mesh = []
    for i in range(m.ngeom):
        if int(m.geom_type[i]) == 7:
            mi = int(m.geom_dataid[i])
            nf = int(m.mesh_facenum[mi])
            alpha = float(m.geom_rgba[i][3])
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
            bid = int(m.geom_bodyid[i])
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}"
            per_geom_mesh.append({"geom": gname, "body": bname, "mesh": meshes[mi]["name"],
                                  "nface": nf, "alpha": alpha, "group": int(m.geom_group[i]),
                                  "matid": int(m.geom_matid[i])})
            inst += nf
    rec["mesh_instanced_tri"] = int(inst)
    rec["mesh_instanced_tri_alpha_gt0"] = int(sum(r["nface"] for r in per_geom_mesh if r["alpha"] > 0))
    rec["mesh_geoms"] = per_geom_mesh

    # primitives with their sizes, per body
    prims = []
    for i in range(m.ngeom):
        t = int(m.geom_type[i])
        if t == 7:
            continue
        gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
        bid = int(m.geom_bodyid[i])
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}"
        prims.append({"geom": gname, "body": bname, "type": GEOM_NAMES.get(t, str(t)),
                      "size": m.geom_size[i].tolist(), "rgba": m.geom_rgba[i].tolist(),
                      "matid": int(m.geom_matid[i]), "group": int(m.geom_group[i]),
                      "bodyid": bid, "static": bool(m.body_dofnum[bid] == 0 and m.body_jntnum[bid] == 0)})
    rec["primitive_geoms"] = prims

    # textures / materials / lights / cameras
    rec["textures"] = [
        {"id": i,
         "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, i) or f"tex_{i}",
         "type": TEX_TYPE.get(int(m.tex_type[i]), str(int(m.tex_type[i]))),
         "h": int(m.tex_height[i]), "w": int(m.tex_width[i]),
         "nchannel": int(m.tex_nchannel[i]) if hasattr(m, "tex_nchannel") else None}
        for i in range(m.ntex)]
    rec["materials"] = [
        {"id": i, "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i) or f"mat_{i}",
         "texid": [int(x) for x in np.atleast_1d(m.mat_texid[i])],
         "rgba": m.mat_rgba[i].tolist(), "specular": float(m.mat_specular[i]),
         "shininess": float(m.mat_shininess[i]), "reflectance": float(m.mat_reflectance[i]),
         "emission": float(m.mat_emission[i]),
         "texrepeat": m.mat_texrepeat[i].tolist(), "texuniform": int(m.mat_texuniform[i])}
        for i in range(m.nmat)]
    rec["lights"] = [
        {"id": i, "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_LIGHT, i) or f"light_{i}",
         "type": LIGHT_TYPE.get(int(m.light_type[i]), str(int(m.light_type[i]))) if hasattr(m, "light_type")
                 else ("DIRECTIONAL" if int(m.light_directional[i]) else "SPOT"),
         "castshadow": int(m.light_castshadow[i]), "active": int(m.light_active[i]),
         "pos": m.light_pos[i].tolist(), "dir": m.light_dir[i].tolist(),
         "diffuse": m.light_diffuse[i].tolist(), "specular": m.light_specular[i].tolist(),
         "ambient": m.light_ambient[i].tolist(),
         "cutoff": float(m.light_cutoff[i]), "exponent": float(m.light_exponent[i]),
         "attenuation": m.light_attenuation[i].tolist(),
         "bodyid": int(m.light_bodyid[i])}
        for i in range(m.nlight)]
    rec["model_cameras"] = [
        {"id": i, "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"cam_{i}",
         "mode": CAM_MODE.get(int(m.cam_mode[i]), str(int(m.cam_mode[i]))),
         "fovy": float(m.cam_fovy[i]), "pos": m.cam_pos[i].tolist(),
         "quat": m.cam_quat[i].tolist(), "bodyid": int(m.cam_bodyid[i]),
         "targetbodyid": int(m.cam_targetbodyid[i]),
         "orthographic": int(m.cam_orthographic[i]) if hasattr(m, "cam_orthographic") else None}
        for i in range(m.ncam)]
    rec["visual"] = {
        "haze": float(m.vis.map.haze), "shadowsize": int(m.vis.quality.shadowsize),
        "offsamples": int(m.vis.quality.offsamples),
        "znear": float(m.vis.map.znear), "zfar": float(m.vis.map.zfar),
        "ambient": m.vis.headlight.ambient.tolist(),
        "headlight_diffuse": m.vis.headlight.diffuse.tolist(),
        "headlight_specular": m.vis.headlight.specular.tolist(),
        "headlight_active": int(m.vis.headlight.active),
        "shadowclip": float(m.vis.map.shadowclip), "shadowscale": float(m.vis.map.shadowscale),
    }
    rec["scene_option_flags"] = _scene_option(u)
    rec["skybox"] = any(t["type"] == "SKYBOX" for t in rec["textures"])

    # ---- state contract -----------------------------------------------------
    rec["state"] = {
        "nq": int(m.nq), "nv": int(m.nv), "nmocap": int(m.nmocap),
        "qpos_bytes": int(m.nq * 8), "qvel_bytes": int(m.nv * 8),
        "mocap_bytes": int(m.nmocap * 7 * 8),
        "joint_names": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"jnt_{j}" for j in range(m.njnt)],
        "mocap_bodies": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body_{b}"
                         for b in range(m.nbody) if int(m.body_mocapid[b]) >= 0],
    }
    return env, u, rec


def _scene_option(u):
    so = getattr(u, "_scene_option", None)
    if so is None:
        return None
    import mujoco
    out = {}
    try:
        out["flags"] = [int(x) for x in so.flags]
        out["geomgroup"] = [int(x) for x in so.geomgroup]
        out["sitegroup"] = [int(x) for x in so.sitegroup]
        out["frame"] = int(so.frame)
        out["label"] = int(so.label)
    except Exception as e:
        out["err"] = str(e)
    return out


def _flat_variation(vs, prefix=""):
    if vs is None:
        return []
    out = []
    try:
        items = vs.items()
    except Exception:
        return [prefix.rstrip(".")]
    for k, v in items:
        p = f"{prefix}{k}"
        if hasattr(v, "items"):
            out.extend(_flat_variation(v, p + "."))
        else:
            shape = getattr(v, "shape", None)
            out.append(f"{p}{list(shape) if shape is not None else ''}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", nargs="+", default=["swm/OGBCube-v0", "swm/OGBScene-v0"])
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for eid in a.envs:
        env, u, rec = census(eid, out)
        tag = eid.split("/")[-1].replace("-v0", "")
        (out / f"census_{tag}.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
        print(f"{eid}: ngeom={rec['counts']['ngeom']} nmesh={rec['counts']['nmesh']} "
              f"tri={rec['mesh_total_tri']} inst_tri={rec['mesh_instanced_tri']} "
              f"nq={rec['counts']['nq']} nv={rec['counts']['nv']} nmocap={rec['counts']['nmocap']} "
              f"nlight={rec['counts']['nlight']} ntex={rec['counts']['ntex']} "
              f"obs={rec['observation_shape']}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
