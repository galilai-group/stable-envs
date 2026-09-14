"""Step 6G cpu-3d, Phase 1: model census for the four MJWarp-only DM Control cells.

Reads the compiled model (never the XML), the frozen Step-4 corpus, and MuJoCo's own forward
kinematics, and writes one JSON per domain under examples/results/step6/cpu-3d/census/. Nothing
here renders; it needs no GL and no GPU, so it runs on the Mac (mujoco 3.12 -- the compiled
geometry, materials and textures are byte-identical to 3.11, see
examples/results/step6/texture-fidelity/findings.md section 2).

    python scratch/step6/cpu-3d/census_dmc.py --out examples/results/step6/cpu-3d/census
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DOMAINS = {"DMCCheetahRun": ("cheetah", "run"), "DMCHopperHop": ("hopper", "hop"),
           "DMCWalkerWalk": ("walker", "walk"), "DMCHumanoidStand": ("humanoid", "stand")}
GEOM = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder",
        6: "box", 7: "mesh", 8: "sdf"}
CAMMODE = {0: "fixed", 1: "track", 2: "trackcom", 3: "targetbody", 4: "targetbodycom"}


def census(cname: str, domain: str, task: str, states_dir: Path) -> dict:
    import mujoco
    from dm_control import suite
    env = suite.load(domain, task, task_kwargs={"random": 0})
    p = env.physics
    m = p.model.ptr
    out: dict = {"config": cname, "domain": domain, "task": task,
                 "mujoco": mujoco.__version__, "nq": int(m.nq), "nbody": int(m.nbody)}

    # ---- geoms: type, body, static/mover, group, size, material
    g = []
    for i in range(m.ngeom):
        mid = int(m.geom_matid[i])
        g.append({"id": i, "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i),
                 "type": GEOM[int(m.geom_type[i])], "body": int(m.geom_bodyid[i]),
                 "static": bool(m.geom_bodyid[i] == 0), "group": int(m.geom_group[i]),
                 "size": [float(x) for x in m.geom_size[i]],
                 "rgba": [round(float(x), 4) for x in m.geom_rgba[i]], "matid": mid,
                 "material": (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mid) if mid >= 0
                              else None)})
    out["geoms"] = g
    vis = [x for x in g if x["group"] in (0, 1, 2) and x["rgba"][3] > 0]
    by_type: dict = {}
    for x in vis:
        k = ("static" if x["static"] else "mover") + ":" + x["type"]
        by_type[k] = by_type.get(k, 0) + 1
    out["visible_geoms_by_type"] = by_type
    out["n_movers"] = sum(not x["static"] for x in vis)
    out["n_static"] = sum(x["static"] for x in vis)
    out["nmesh"] = int(m.nmesh)
    out["nsite"] = int(m.nsite)
    out["sites"] = [{"id": i, "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i),
                     "group": int(m.site_group[i]), "body": int(m.site_bodyid[i]),
                     "rgba": [round(float(x), 4) for x in m.site_rgba[i]]}
                    for i in range(m.nsite)]

    # ---- cameras and lights
    out["cameras"] = [{"id": i, "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i),
                       "mode": CAMMODE[int(m.cam_mode[i])], "body": int(m.cam_bodyid[i]),
                       "targetbody": int(m.cam_targetbodyid[i]), "fovy": float(m.cam_fovy[i]),
                       "pos": [float(x) for x in m.cam_pos[i]],
                       "quat": [float(x) for x in m.cam_quat[i]]} for i in range(m.ncam)]
    out["lights"] = [{"id": i, "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_LIGHT, i),
                      "mode": CAMMODE[int(m.light_mode[i])], "body": int(m.light_bodyid[i]),
                      "type": int(m.light_type[i]) if hasattr(m, "light_type") else None,
                      "directional": (bool(m.light_directional[i]) if hasattr(m, "light_directional")
                                      else None),
                      "castshadow": bool(m.light_castshadow[i]),
                      "pos": [float(x) for x in m.light_pos[i]],
                      "dir": [float(x) for x in m.light_dir[i]],
                      "attenuation": [float(x) for x in m.light_attenuation[i]],
                      "cutoff": float(m.light_cutoff[i]), "exponent": float(m.light_exponent[i]),
                      "ambient": [float(x) for x in m.light_ambient[i]],
                      "diffuse": [float(x) for x in m.light_diffuse[i]],
                      "specular": [float(x) for x in m.light_specular[i]]}
                     for i in range(m.nlight)]
    hl = m.vis.headlight
    out["headlight"] = {"active": int(hl.active), "ambient": [float(x) for x in hl.ambient],
                        "diffuse": [float(x) for x in hl.diffuse],
                        "specular": [float(x) for x in hl.specular]}
    out["vis"] = {"znear": float(m.vis.map.znear), "zfar": float(m.vis.map.zfar),
                  "extent": float(m.stat.extent), "shadowsize": int(m.vis.quality.shadowsize),
                  "offsamples": int(m.vis.quality.offsamples),
                  "numslices": int(m.vis.quality.numslices),
                  "numstacks": int(m.vis.quality.numstacks),
                  "offwidth": int(m.vis.global_.offwidth), "offheight": int(m.vis.global_.offheight)}

    # ---- materials and textures
    mats = []
    for i in range(m.nmat):
        t = m.mat_texid[i]
        tex = [int(x) for x in np.atleast_1d(t)]
        mats.append({"id": i, "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i),
                     "rgba": [round(float(x), 4) for x in m.mat_rgba[i]], "texid": tex,
                     "texrepeat": [float(x) for x in m.mat_texrepeat[i]],
                     "texuniform": int(m.mat_texuniform[i]),
                     "reflectance": float(m.mat_reflectance[i]),
                     "specular": float(m.mat_specular[i]), "shininess": float(m.mat_shininess[i]),
                     "emission": float(m.mat_emission[i])})
    out["materials"] = mats
    texs = []
    for i in range(m.ntex):
        w, h = int(m.tex_width[i]), int(m.tex_height[i])
        nc = int(m.tex_nchannel[i]) if hasattr(m, "tex_nchannel") else 3
        a0 = int(m.tex_adr[i])
        a = np.asarray(m.tex_data[a0:a0 + w * h * nc], np.uint8).reshape(h, w, nc)
        u = np.unique(a.reshape(-1, nc), axis=0)
        texs.append({"id": i, "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, i),
                     "type": int(m.tex_type[i]), "w": w, "h": h, "nchannel": nc,
                     "distinct_colours": int(len(u)), "bytes": int(w * h * nc),
                     "sha256": hashlib.sha256(a.tobytes()).hexdigest()[:16],
                     "mean_rgb": [round(float(x), 2) for x in a.reshape(-1, nc).mean(0)[:3]]})
    out["textures"] = texs

    # ---- the frozen corpus: camera / light travel, mover extents, FK cost
    npz = states_dir / f"{cname}.npz"
    z = np.load(npz)
    st = np.asarray(z["states"], np.float64)
    out["corpus"] = {"path": str(npz.relative_to(ROOT)), "n": int(st.shape[0]),
                     "state_dim": int(st.shape[1]),
                     "sha256": hashlib.sha256(np.asarray(z["states"]).tobytes()).hexdigest()}
    d = mujoco.MjData(m)
    cam_p, cam_R, lt_p, lt_d, ext = [], [], [], [], []
    movers = [x["id"] for x in vis if not x["static"]]
    for row in st:
        d.qpos[:] = row[:m.nq]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        cam_p.append(np.array(d.cam_xpos[0]))
        cam_R.append(np.array(d.cam_xmat[0]))
        lt_p.append(np.array(d.light_xpos))
        lt_d.append(np.array(d.light_xdir))
        xp = np.asarray(d.geom_xpos)[movers]
        r = np.array([max(np.asarray(m.geom_size[i])[:2].sum(), 1e-3) for i in movers])
        ext.append(np.concatenate([xp.min(0) - r.max(), xp.max(0) + r.max()]))
    cam_p, cam_R, lt_p, lt_d, ext = map(np.asarray, (cam_p, cam_R, lt_p, lt_d, ext))
    out["frozen_corpus_motion"] = {
        "cam_xpos_min": cam_p.min(0).round(4).tolist(), "cam_xpos_max": cam_p.max(0).round(4).tolist(),
        "cam_xpos_spread": (cam_p.max(0) - cam_p.min(0)).round(4).tolist(),
        "cam_xmat_constant": bool(np.abs(cam_R - cam_R[0]).max() < 1e-9),
        "cam_xmat": cam_R[0].round(6).tolist(),
        "light_xpos_spread": (lt_p.max(0) - lt_p.min(0)).round(4).tolist(),
        "light_xdir_constant": bool(np.abs(lt_d - lt_d[0]).max() < 1e-9),
        "light_rel_cam_spread": ((lt_p - cam_p[:, None]).max(0) - (lt_p - cam_p[:, None]).min(0)
                                 ).round(4).tolist(),
        "mover_world_aabb_min": ext[:, :3].min(0).round(3).tolist(),
        "mover_world_aabb_max": ext[:, 3:].max(0).round(3).tolist(),
        "mover_aabb_extent_median": np.median(ext[:, 3:] - ext[:, :3], 0).round(3).tolist(),
    }
    # FK cost on this host, one core, for the state -> pose step every CPU design pays
    q = st[0, :m.nq]
    for fn, name in ((mujoco.mj_kinematics, "mj_kinematics"), (mujoco.mj_forward, "mj_forward")):
        d.qpos[:] = q
        fn(m, d)
        t0 = time.perf_counter()
        for _ in range(2000):
            fn(m, d)
        out[f"{name}_us"] = round((time.perf_counter() - t0) / 2000 * 1e6, 2)
    # com_pos-equivalent: mj_comPos exists as a separate stage
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    t0 = time.perf_counter()
    for _ in range(2000):
        mujoco.mj_comPos(m, d)
    out["mj_comPos_us"] = round((time.perf_counter() - t0) / 2000 * 1e6, 2)
    t0 = time.perf_counter()
    for _ in range(2000):
        mujoco.mj_camlight(m, d)
    out["mj_camlight_us"] = round((time.perf_counter() - t0) / 2000 * 1e6, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--states-dir", type=Path, default=ROOT / "examples/results/step4/states")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    for cname, (dom, task) in DOMAINS.items():
        r = census(cname, dom, task, a.states_dir)
        r["host"] = host
        r["cpu"] = platform.processor() or platform.machine()
        (a.out / f"{cname}_{host}.json").write_text(json.dumps(r, indent=1))
        print(f"{cname:18s} movers={r['n_movers']:2d} static={r['n_static']} "
              f"types={r['visible_geoms_by_type']} cam={r['cameras'][0]['mode']} "
              f"lights={[ (l['mode'], l['castshadow'], l['cutoff']) for l in r['lights']]} "
              f"cam_spread={r['frozen_corpus_motion']['cam_xpos_spread']} "
              f"fk={r['mj_kinematics_us']}us fwd={r['mj_forward_us']}us")


if __name__ == "__main__":
    main()
