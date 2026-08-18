"""Where does an MJWarp frame's time actually go, on RoboCasa and HumanoidBench?

The premise of the whole alternative-backend search is that the ray tracer is the bottleneck. Before
writing a rasterizer this measures WHAT the ray tracer is paying for, using the one free control the
repo already has: ``RoboCasaCounterToSinkWrist`` is the SAME compiled kitchen as
``RoboCasaCounterToSink`` seen through a different camera, and it runs 2.3x faster. If the cost is
the scene, those two must cost the same; if the cost is what the camera sees, they must not.

Four decompositions, per (kitchen, camera):

1. **Per stage** -- kinematics / camlight / refit_bvh / render, each timed alone and synchronised,
   against the captured graph's total.
2. **Per render feature** -- shadows, textures, skybox switched off one at a time in a fresh context.
   Each is a different picture, so this is a cost attribution and NOT a proposal; it names which term
   of the shader the time is in.
3. **What the camera sees** -- MuJoCo's own segmentation and depth buffers at the same camera and
   resolution: how many distinct geoms land on screen, what fraction of pixels are sky, the mean and
   spread of depth. This is the quantity the wrist camera changes and the scene does not.
4. **What MOVES** -- per-pixel disagreement across states drawn from the env's own pool at the DEFAULT
   style. A pixel that is identical in every frame of the benchmark is a pixel a static plate could
   have painted once.

Run one env per process: an MJWarp render context is not cheap to have several of, and a CUDA error
poisons the process.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch


def timed(fn, reps=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# ---------------------------------------------------------------- geometry facts


def scene_facts(m, groups) -> dict:
    """Static description of what an enabled-group render has to draw."""
    import mujoco
    g = np.asarray(m.geom_group, np.int64)
    vis = np.where(np.isin(g, list(groups)))[0]
    types = np.asarray(m.geom_type, np.int64)[vis]
    names = {int(v): k for k, v in vars(mujoco.mjtGeom).items()
             if isinstance(v, int) and k.startswith("mjGEOM_")}
    by_type = {}
    for t in np.unique(types):
        by_type[names.get(int(t), str(int(t)))] = int((types == t).sum())
    meshid = np.asarray(m.geom_dataid, np.int64)[vis]
    meshes = sorted({int(x) for x, t in zip(meshid, types)
                     if x >= 0 and int(t) == int(mujoco.mjtGeom.mjGEOM_MESH)})
    tris = int(sum(int(m.mesh_facenum[i]) for i in meshes))
    tris_inst = int(sum(int(m.mesh_facenum[x]) for x, t in zip(meshid, types)
                        if x >= 0 and int(t) == int(mujoco.mjtGeom.mjGEOM_MESH)))
    verts = int(sum(int(m.mesh_vertnum[i]) for i in meshes))
    texbytes = int(np.asarray(m.tex_height, np.int64) @ np.asarray(m.tex_width, np.int64)
                   * 3) if m.ntex else 0
    return {"ngeom_total": int(m.ngeom), "ngeom_visible": int(len(vis)), "by_type": by_type,
            "nmesh_unique": len(meshes), "tri_unique": tris, "tri_instanced": tris_inst,
            "vert_unique": verts, "ntex": int(m.ntex), "tex_bytes_approx": texbytes,
            "nlight": int(m.nlight),
            "ncastshadow": int(np.asarray(m.light_castshadow, np.int64).sum())
            if m.nlight else 0,
            "nq": int(m.nq), "nbody": int(m.nbody), "nmesh_model": int(m.nmesh)}


def seen_facts(m, cam_id, res, qpos_rows, groups) -> dict:
    """Segmentation + depth from MuJoCo's own renderer: what this camera actually resolves."""
    import mujoco
    d = mujoco.MjData(m)
    vopt = mujoco.MjvOption()
    vopt.geomgroup[:] = 0
    for gg in groups:
        vopt.geomgroup[gg] = 1
    r = mujoco.Renderer(m, width=int(res), height=int(res))
    seen, sky, depths, rgbs = set(), [], [], []
    for row in qpos_rows:
        d.qpos[:] = row
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=int(cam_id), scene_option=vopt)
        r.enable_segmentation_rendering()
        seg = r.render()[..., 1].astype(np.int64)          # geom id per pixel, -1 for none
        r.disable_segmentation_rendering()
        r.enable_depth_rendering()
        dep = r.render().astype(np.float64)
        r.disable_depth_rendering()
        rgbs.append(r.render())
        u = np.unique(seg)
        seen.update(int(x) for x in u if x >= 0)
        sky.append(float((seg < 0).mean()))
        far = dep[dep < dep.max() * 0.999]
        depths.append((float(np.mean(dep)), float(np.median(dep)),
                       float(np.mean(far)) if far.size else float("nan")))
    rgbs = np.stack(rgbs)
    moved = (rgbs != rgbs[:1]).any(-1).any(0)               # pixel differs from state 0 anywhere
    return {"geoms_seen": len(seen), "sky_frac": float(np.mean(sky)),
            "depth_mean": float(np.mean([x[0] for x in depths])),
            "depth_median": float(np.mean([x[1] for x in depths])),
            "depth_mean_nonfar": float(np.mean([x[2] for x in depths])),
            "moving_pixel_frac": float(moved.mean()),
            "n_states": int(len(qpos_rows))}


# ---------------------------------------------------------------- timing


def time_context(m, cam_id, res, nworld, groups, qpos0, *, shadows=True, textures=True,
                 skybox=True, block_dim=None, reps=30) -> dict:
    """Build one context with the given flags and time the graph and each stage."""
    import mujoco
    import warp as wp

    import mujoco_warp as mjw
    import stable_envs.render.mjwarp as MW

    d0 = mujoco.MjData(m)
    d0.qpos[:] = qpos0
    mujoco.mj_forward(m, d0)
    wm = mjw.put_model(m, batch_sizes={k: int(nworld) for k in
                                       ("cam_pos", "cam_quat", "light_diffuse")})
    if block_dim is not None:
        wm.block_dim.render = int(block_dim)
    wd = mjw.put_data(m, d0, nworld=int(nworld))
    with MW.gl_rounding():
        rc = MW.create_render_context(
            m, nworld=int(nworld), cam_res=(int(res), int(res)), render_rgb=True,
            cam_active=[i == int(cam_id) for i in range(m.ncam)],
            enabled_geom_groups=list(groups), use_textures=textures, use_shadows=shadows,
            render_skybox=skybox)

        def frame():
            mjw.kinematics(wm, wd)
            mjw.camlight(wm, wd)
            mjw.refit_bvh(wm, wd, rc)
            mjw.render(wm, wd, rc)

        frame()
        wp.synchronize()
        with wp.ScopedCapture() as cap:
            frame()
        graph = cap.graph

    out = {"nworld": int(nworld), "res": int(res), "shadows": bool(shadows),
           "textures": bool(textures), "skybox": bool(skybox)}
    out["graph_ms"] = 1e3 * timed(lambda: (wp.capture_launch(graph), wp.synchronize()), reps)
    out["fps"] = nworld / (out["graph_ms"] / 1e3)
    for nm, fn in (("kinematics", lambda: mjw.kinematics(wm, wd)),
                   ("camlight", lambda: mjw.camlight(wm, wd)),
                   ("refit_bvh", lambda: mjw.refit_bvh(wm, wd, rc)),
                   ("render", lambda: mjw.render(wm, wd, rc))):
        out[f"{nm}_ms"] = 1e3 * timed(lambda f=fn: (f(), wp.synchronize()), max(6, reps // 3))
    frames = wp.to_torch(rc.rgb_data).view(torch.uint8).reshape(
        int(nworld), int(res), int(res), 4)[..., [2, 1, 0]].clone()
    out["_frames"] = frames
    del wd, wm, rc, graph
    return out


# ---------------------------------------------------------------- env plumbing


def load(name: str):
    """(model, cam_id, groups, qpos_pool, module) for one registered cell, without an env object."""
    import mujoco
    import stable_envs as se
    cls = se.REGISTRY[name]
    if name.startswith("RoboCasa"):
        from stable_envs.envs import robocasa as R
        s = R._scene(cls.KITCHEN)
        cid = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_CAMERA, cls.CAMERA)
        pool = np.asarray(R._yawed_pool(cls.KITCHEN), np.float64)
        return s.model, cid, R.VISIBLE_GROUPS, pool, cls.CAMERA
    if name.startswith("HumanoidBench"):
        from stable_envs.envs import humanoidbench as H
        b = H._build_bundle(cls.ROBOT, cls.TASK)
        pool = np.asarray(H._state_pool(cls.ROBOT, cls.TASK, b.nq), np.float64)
        return b.model, int(b.info["cam"]), H.VISIBLE_GROUPS, pool, H._camera_name(cls.TASK)
    raise SystemExit(f"unhandled {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--states", type=int, default=16)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import stable_envs as se                                        # noqa: F401
    m, cid, groups, pool, camname = load(a.env)
    rng = np.random.default_rng(0)
    rows = pool[rng.integers(0, len(pool), a.states)]

    res = {"env": a.env, "camera": camname, "res": a.res, "nworld": a.nworld,
           "gpu": torch.cuda.get_device_name(0)}
    res["scene"] = scene_facts(m, groups)
    print(json.dumps(res["scene"], indent=1), flush=True)

    variants = [dict(), dict(shadows=False), dict(textures=False), dict(skybox=False),
                dict(shadows=False, textures=False, skybox=False)]
    res["timing"] = []
    ref = None
    for v in variants:
        t = time_context(m, cid, a.res, a.nworld, groups, pool[0], reps=a.reps,
                         block_dim=32 if a.env.startswith("RoboCasa") else None, **v)
        fr = t.pop("_frames")
        if ref is None:
            ref = fr
            t["mean_abs_diff_vs_full"] = 0.0
        else:
            t["mean_abs_diff_vs_full"] = float(
                (fr.to(torch.float32) - ref.to(torch.float32)).abs().mean())
        res["timing"].append(t)
        print(json.dumps(t), flush=True)
        del fr
        torch.cuda.empty_cache()
    del ref
    torch.cuda.empty_cache()

    # what the camera sees -- MuJoCo's renderer, so it needs its own compilation-free pass
    res["seen"] = seen_facts(m, cid, a.res, rows, groups)
    print(json.dumps(res["seen"], indent=1), flush=True)

    out = a.out or (f"/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/altrender/"
                    f"decompose_{a.env}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
