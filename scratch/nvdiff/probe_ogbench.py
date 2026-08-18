"""The sanity gate: OGBench mazes, where the analytic ray-caster is interior-exact 1.0.

If nvdiffrast cannot match here it will not match anywhere, so this runs first and cheaply. It also
dumps the resolved scene facts (lights, camera, materials, textures) because every calibration below
depends on reading them off MuJoCo rather than off a source file.

One env per process: two MuJoCo envs alive together can return byte-identical frames from cameras
that provably differ.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import emit, save, score  # noqa: E402


def facts(m, scn, cam):
    import mujoco
    out = {"ngeom": int(m.ngeom), "nmat": int(m.nmat), "ntex": int(m.ntex),
           "nlight": int(scn.nlight), "extent": float(m.stat.extent),
           "znear": float(m.vis.map.znear), "zfar": float(m.vis.map.zfar),
           "offsamples": int(m.vis.quality.offsamples),
           "numslices": int(m.vis.quality.numslices),
           "numstacks": int(m.vis.quality.numstacks),
           "numquads": int(m.vis.quality.numquads),
           "global_fovy": float(m.vis.global_.fovy),
           "cam_pos": np.asarray(cam.pos).tolist(), "cam_fovy": cam.fovy,
           "cam_mat": np.asarray(cam.mat).reshape(3, 3).tolist(),
           "cam_near": cam.znear, "cam_far": cam.zfar,
           "light_fields": sorted(a for a in dir(scn.lights[0]) if not a.startswith("_")),
           "lights": [],
           "textures": [], "materials": [], "geomtypes": {}}
    for i in range(int(scn.nlight)):
        li = scn.lights[i]
        out["lights"].append({k: (np.asarray(getattr(li, k)).tolist()
                                  if hasattr(getattr(li, k), "__len__") else float(getattr(li, k)))
                              for k in ("pos", "dir", "ambient", "diffuse", "specular",
                                        "attenuation", "cutoff", "exponent")
                              if hasattr(li, k)})
        for k in ("headlight", "directional", "castshadow", "type"):
            if hasattr(li, k):
                out["lights"][-1][k] = int(getattr(li, k))
    for t in range(int(m.ntex)):
        out["textures"].append({"id": t, "type": int(m.tex_type[t]),
                                "w": int(m.tex_width[t]), "h": int(m.tex_height[t]),
                                "nchannel": int(m.tex_nchannel[t]) if hasattr(m, "tex_nchannel") else 3,
                                "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, t)})
    for k in range(int(m.nmat)):
        out["materials"].append({"id": k,
                                 "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, k),
                                 "texid": np.asarray(m.mat_texid[k]).tolist(),
                                 "texrepeat": np.asarray(m.mat_texrepeat[k]).tolist(),
                                 "texuniform": int(m.mat_texuniform[k]),
                                 "rgba": np.asarray(m.mat_rgba[k]).tolist(),
                                 "specular": float(m.mat_specular[k]),
                                 "shininess": float(m.mat_shininess[k]),
                                 "emission": float(m.mat_emission[k]),
                                 "reflectance": float(m.mat_reflectance[k])})
    for g in range(int(m.ngeom)):
        key = f"{int(m.geom_type[g])}/g{int(m.geom_group[g])}"
        out["geomtypes"][key] = out["geomtypes"].get(key, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loco", default="point")
    ap.add_argument("--maze", default="medium")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--states", type=int, default=8)
    ap.add_argument("--offsamples", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import gymnasium
    import mujoco
    import ogbench  # noqa: F401
    import torch

    from stable_envs.render import nvdiff as ND

    env = gymnasium.make(f"{args.loco}maze-{args.maze}-v0", render_mode="rgb_array",
                         ob_type="pixels", width=args.size, height=args.size)
    env.reset(options={"task_id": 1})
    u = env.unwrapped
    m, d = u.model, u.data
    m.vis.quality.offsamples = int(args.offsamples)
    cam_mjv = getattr(u, "custom_camera", None)
    assert cam_mjv is not None, "no custom_camera on this env; the camera convention must be measured"

    br = ND.BatchRenderer(m, res=args.size, device="cuda", textures=True, mipmap=True, skybox=True)
    cam, scn = br.resolve_from_mujoco(m, d, camera=cam_mjv)
    emit("facts", facts(m, scn, cam))
    emit("scene", {"geoms_drawn": br.scene.G, "verts": br.scene.V,
                   "tris": int(br.scene.face.shape[0]),
                   "textures": {int(k): v["shape"] + (str(v["resampled"]),)
                                for k, v in br.scene.textures.items()},
                   "skybox": None if br.skybox_tex is None else br.skybox_tex["shape"]})

    rng = np.random.default_rng(0)
    nq = m.nq
    q0 = np.asarray(d.qpos, np.float64).copy()
    states = []
    for _ in range(args.states):
        q = q0.copy()
        q[:2] += rng.uniform(-3.0, 3.0, 2)
        if nq > 7:
            q[7:] += rng.uniform(-0.3, 0.3, nq - 7)
        states.append(q)
    states = np.stack(states)

    ren = mujoco.Renderer(m, height=args.size, width=args.size)
    ref = []
    for q in states:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        ren.update_scene(d, camera=cam_mjv)
        ref.append(ren.render().copy())
    ref = np.stack(ref)

    # MuJoCo's own segmentation buffer, so the residual can be split by SURFACE instead of by guess
    ren.enable_segmentation_rendering()
    segs = []
    for q in states:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        ren.update_scene(d, camera=cam_mjv)
        segs.append(ren.render()[..., 0].copy())
    ren.disable_segmentation_rendering()
    segs = np.stack(segs)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    names = {g: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "") for g in range(m.ngeom)}
    walls = np.array([g for g in range(m.ngeom) if names[g].startswith("block_")])
    classes = {"floor": segs == floor, "walls": np.isin(segs, walls),
               "sky": segs < 0,
               "agent": ~np.isin(segs, np.concatenate([[floor], walls])) & (segs >= 0)}

    qt = torch.as_tensor(states, dtype=torch.float32)
    results = {}
    variants = [("full", dict(tex=True, mip=True, pol="pot")),
                ("exact_tex", dict(tex=True, mip=True, pol="exact")),
                ("no_tex", dict(tex=False, mip=True, pol="pot")),
                ("trunc", dict(tex=True, mip=True, pol="pot", quant="trunc"))]
    for name, kw in variants:
        if kw["pol"] != br.tex_policy:
            br = ND.BatchRenderer(m, res=args.size, device="cuda", tex_policy=kw["pol"])
            br.resolve_from_mujoco(m, d, camera=cam_mjv)
        br.use_tex, br.mipmap = kw["tex"], kw["mip"]
        br.quant = kw.get("quant", "round")
        ours = br.render_qpos(qt, cam=cam).cpu().numpy()
        row = {"all": score(ours, ref)}
        for cname, cmask in classes.items():
            row[cname] = score(ours, ref, cmask)
        results[name] = row
        emit(f"score.{name}", {k: {kk: vv for kk, vv in v.items()
                                   if kk in ("n", "mean_abs", "exact_frac", "within8_frac",
                                             "interior_exact", "interior_frac")}
                               for k, v in row.items()})
        if name == "full":
            np.save(f"/mnt/vast/home/randall.balestriero/stable_envs_logs/nvdiff/"
                    f"og_{args.loco}_{args.maze}{args.tag}_ours.npy", ours[:2])
            np.save(f"/mnt/vast/home/randall.balestriero/stable_envs_logs/nvdiff/"
                    f"og_{args.loco}_{args.maze}{args.tag}_ref.npy", ref[:2])
    save(f"ogbench_{args.loco}_{args.maze}{args.tag}.json",
         {"args": vars(args), "facts": facts(m, scn, cam), "scores": results})


if __name__ == "__main__":
    main()
