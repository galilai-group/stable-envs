"""Dump the ACTUAL MjModel / camera / light / material parameters of every OGBench maze scene.

Nothing here is assumed: every number is read out of the compiled ``MjModel`` that ``ogbench`` builds,
after the ``ob_type='pixels'`` XML surgery (wall recolor, ambient removal, torso-light removal) and after
rjepa's ``decolor_floor`` / ``mark_agent_red``. Writes one JSON per scene plus a combined index.

Run under SLURM with MUJOCO_GL=egl.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out"

LOCO = ["point", "ant", "humanoid"]
SIZES = ["medium", "large", "giant", "teleport"]


def jsonable(x):
    """json ``default`` hook. Must never return ``x`` unchanged: json calls ``default`` again on whatever
    it gets back, so returning the same object is an infinite C-level recursion that ends in a stack
    overflow (SIGSEGV) with no Python traceback, mid-write, leaving a truncated file."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return repr(x)


def make(loco, size, img=224, decolor=True, marker=None):
    import gymnasium
    import ogbench  # noqa: F401
    env_id = f"{loco}maze-{size}-v0"
    env = gymnasium.make(env_id, render_mode="rgb_array", ob_type="pixels", width=img, height=img)
    env.reset(options={"task_id": 1})
    u = env.unwrapped
    if decolor:
        sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/counterfactual-WM")
        from rjepa.envs.ogbench_maze import decolor_floor, mark_agent_red
        decolor_floor(u)
        if marker is None:
            marker = 4.0 if loco == "humanoid" else 0.0
        if marker > 0:
            mark_agent_red(u, scale=marker)
    return env, u


def probe(loco, size, img=224):
    import mujoco
    print(f"probe {loco}-{size} ...", flush=True)
    env, u = make(loco, size, img)
    print("  env built", flush=True)
    m, d = u.model, u.data
    out = {"env_id": f"{loco}maze-{size}-v0", "loco": loco, "size": size, "img": img}

    # ---- sizes ----------------------------------------------------------------
    out["counts"] = {k: int(getattr(m, k)) for k in
                     ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nsite", "nmesh", "nhfield",
                      "ntex", "nmat", "nlight", "ncam", "nsensor", "nskin", "nflex" if hasattr(m, "nflex") else "nq")}
    out["maze_map"] = u.maze_map.tolist()
    out["maze_unit"] = float(u._maze_unit)
    out["maze_height"] = float(u._maze_height)
    out["offset"] = [float(u._offset_x), float(u._offset_y)]
    out["qpos0"] = m.qpos0.tolist()
    out["init_qpos"] = np.asarray(u.init_qpos).tolist()
    out["jnt"] = [{"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j),
                   "type": int(m.jnt_type[j]), "bodyid": int(m.jnt_bodyid[j]),
                   "qposadr": int(m.jnt_qposadr[j]), "dofadr": int(m.jnt_dofadr[j]),
                   "axis": m.jnt_axis[j].tolist(), "pos": m.jnt_pos[j].tolist()}
                  for j in range(m.njnt)]

    # ---- geoms ----------------------------------------------------------------
    gtype_name = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid",
                  5: "cylinder", 6: "box", 7: "mesh", 8: "sdf"}
    geoms = []
    for g in range(m.ngeom):
        geoms.append(dict(
            id=g, name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g),
            type=gtype_name.get(int(m.geom_type[g]), int(m.geom_type[g])),
            bodyid=int(m.geom_bodyid[g]),
            body=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])),
            size=m.geom_size[g].tolist(), pos=m.geom_pos[g].tolist(), quat=m.geom_quat[g].tolist(),
            rgba=m.geom_rgba[g].tolist(), matid=int(m.geom_matid[g]),
            group=int(m.geom_group[g]), dataid=int(m.geom_dataid[g]),
            contype=int(m.geom_contype[g]), conaffinity=int(m.geom_conaffinity[g]),
        ))
    out["geoms"] = geoms
    # compact: how many geoms of each type
    from collections import Counter
    out["geom_type_hist"] = dict(Counter(g["type"] for g in geoms))
    # non-wall / non-floor geoms = the moving agent
    out["agent_geoms"] = [g for g in geoms if not (g["name"] or "").startswith("block_")
                          and (g["name"] or "") != "floor"]

    # ---- materials / textures --------------------------------------------------
    mats = []
    for i in range(m.nmat):
        mats.append(dict(
            id=i, name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i),
            rgba=m.mat_rgba[i].tolist(),
            emission=float(m.mat_emission[i]), specular=float(m.mat_specular[i]),
            shininess=float(m.mat_shininess[i]), reflectance=float(m.mat_reflectance[i]),
            texrepeat=m.mat_texrepeat[i].tolist(), texuniform=int(m.mat_texuniform[i]),
            texid=(m.mat_texid[i].tolist() if np.ndim(m.mat_texid[i]) else int(m.mat_texid[i])),
        ))
    out["materials"] = mats
    texs = []
    for i in range(m.ntex):
        texs.append(dict(id=i, name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, i),
                         type=int(m.tex_type[i]), h=int(m.tex_height[i]), w=int(m.tex_width[i]),
                         adr=int(m.tex_adr[i])))
    out["textures"] = texs
    attr = "tex_rgb" if hasattr(m, "tex_rgb") else "tex_data"
    buf = getattr(m, attr)
    for t in texs:
        a0 = t["adr"]
        reg = buf[a0:a0 + 3 * t["h"] * t["w"]].reshape(t["h"], t["w"], 3)
        t["unique_colors"] = int(len(np.unique(reg.reshape(-1, 3), axis=0)))
        t["corner_rgb"] = [reg[0, 0].tolist(), reg[0, -1].tolist(), reg[-1, 0].tolist(), reg[-1, -1].tolist()]

    # ---- lights ----------------------------------------------------------------
    LF = ["light_type", "light_directional", "light_castshadow", "light_active", "light_pos",
          "light_dir", "light_ambient", "light_diffuse", "light_specular", "light_attenuation",
          "light_cutoff", "light_exponent", "light_bodyid", "light_mode", "light_intensity",
          "light_range", "light_bulbradius"]
    lights = []
    for i in range(m.nlight):
        e = dict(id=i, name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_LIGHT, i))
        for a in LF:
            if not hasattr(m, a):
                continue
            try:
                e[a] = jsonable(getattr(m, a)[i])
            except Exception as ex:  # noqa: BLE001
                e[a] = f"ERR {ex}"
        lights.append(e)
    out["lights"] = lights
    print("  lights ok", flush=True)

    # ---- global visual settings ------------------------------------------------
    v = m.vis
    out["vis"] = dict(
        global_=dict(fovy=float(v.global_.fovy),
                     orthographic=int(getattr(v.global_, "orthographic", 0)),
                     azimuth=float(getattr(v.global_, "azimuth", np.nan)),
                     elevation=float(getattr(v.global_, "elevation", np.nan)),
                     linewidth=float(v.global_.linewidth),
                     offwidth=int(v.global_.offwidth), offheight=int(v.global_.offheight),
                     realtime=float(v.global_.realtime)),
        quality=dict(shadowsize=int(v.quality.shadowsize), offsamples=int(v.quality.offsamples),
                     numslices=int(v.quality.numslices), numstacks=int(v.quality.numstacks),
                     numquads=int(v.quality.numquads)),
        headlight=dict(ambient=v.headlight.ambient.tolist(), diffuse=v.headlight.diffuse.tolist(),
                       specular=v.headlight.specular.tolist(), active=int(v.headlight.active)),
        map={k: float(getattr(v.map, k)) for k in dir(v.map)
             if not k.startswith("_") and isinstance(getattr(v.map, k), (float, int))},
        rgba=dict(fog=v.rgba.fog.tolist(), haze=v.rgba.haze.tolist()),
        scale=dict(forcewidth=float(v.scale.forcewidth)),
    )
    out["stat"] = dict(extent=float(m.stat.extent), meansize=float(m.stat.meansize),
                       meanmass=float(m.stat.meanmass), center=m.stat.center.tolist())

    # ---- camera ---------------------------------------------------------------
    cam = u.custom_camera
    out["camera"] = dict(kind=type(cam).__name__)
    if hasattr(cam, "lookat"):
        out["camera"].update(lookat=cam.lookat.tolist(), distance=float(cam.distance),
                             azimuth=float(cam.azimuth), elevation=float(cam.elevation),
                             type=int(cam.type), fixedcamid=int(cam.fixedcamid),
                             trackbodyid=int(cam.trackbodyid),
                             orthographic=int(getattr(cam, "orthographic", 0)))
    out["model_cameras"] = [dict(id=c, name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, c),
                                 mode=int(m.cam_mode[c]), fovy=float(m.cam_fovy[c]),
                                 pos=m.cam_pos[c].tolist(), bodyid=int(m.cam_bodyid[c]),
                                 orthographic=int(getattr(m, "cam_orthographic", np.zeros(m.ncam))[c]))
                            for c in range(m.ncam)]

    # ---- what the scene actually contains at render time ------------------------
    scn = u.custom_renderer.scene
    u.custom_renderer.update_scene(d, camera=u.custom_camera)
    out["scene"] = dict(ngeom=int(scn.ngeom), maxgeom=int(scn.maxgeom), nlight=int(scn.nlight),
                        enabletransform=int(scn.enabletransform),
                        flags=[int(x) for x in scn.flags])
    def dump_obj(o, fields):
        e = {}
        for a in fields:
            if not hasattr(o, a):
                continue
            try:
                e[a] = jsonable(getattr(o, a))
            except Exception as ex:  # noqa: BLE001
                e[a] = f"ERR {ex}"
        return e

    LIGHT_F = ["type", "directional", "castshadow", "pos", "dir", "ambient", "diffuse", "specular",
               "attenuation", "cutoff", "exponent", "headlight", "bulbradius", "intensity", "range"]
    CAM_F = ["pos", "forward", "up", "frustum_center", "frustum_bottom", "frustum_top",
             "frustum_near", "frustum_far", "frustum_width", "orthographic"]
    out["scene_lights"] = [dump_obj(scn.lights[i], LIGHT_F) for i in range(int(scn.nlight))]
    print("  scene_lights ok", flush=True)
    # the free-camera pose mjv actually computed
    out["scene_camera"] = [dump_obj(scn.camera[i], CAM_F) for i in range(2)]
    print("  scene_camera ok", flush=True)

    # visible scene geoms actually emitted (id + type)
    GEOM_F = ["type", "objid", "objtype", "category", "matid", "rgba", "emission",
              "specular", "shininess", "reflectance", "size", "pos"]
    sg = [dump_obj(scn.geoms[i], GEOM_F) for i in range(int(scn.ngeom))]
    print("  scene_geoms ok", flush=True)
    out["scene_geoms_n"] = len(sg)
    out["scene_geoms_head"] = sg[:6]
    from collections import Counter as C2
    out["scene_geom_type_hist"] = dict(C2(x["type"] for x in sg))

    env.close()
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    index = {}
    for loco in LOCO:
        for size in SIZES:
            if loco == "humanoid" and size == "teleport":
                continue  # not among the 25 release cells
            try:
                o = probe(loco, size)
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {loco}-{size}: {type(e).__name__}: {e}", flush=True)
                continue
            p = os.path.join(OUT, f"scene_{loco}_{size}.json")
            with open(p, "w") as f:
                json.dump(o, f, indent=1, default=jsonable)
            index[f"{loco}-{size}"] = dict(
                counts=o["counts"], geom_type_hist=o["geom_type_hist"],
                camera=o["camera"], scene_camera=o["scene_camera"][0],
                nlight_scene=o["scene"]["nlight"], scene_geom_type_hist=o["scene_geom_type_hist"],
                scene_ngeom=o["scene"]["ngeom"],
            )
            print(f"OK {loco}-{size}: nq={o['counts']['nq']} nv={o['counts']['nv']} "
                  f"ngeom={o['counts']['ngeom']} nmesh={o['counts']['nmesh']} "
                  f"scene_ngeom={o['scene']['ngeom']} nlight={o['scene']['nlight']}", flush=True)
    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump(index, f, indent=1, default=jsonable)
    print(json.dumps(index, indent=1, default=jsonable))


if __name__ == "__main__":
    main()
