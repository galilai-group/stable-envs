"""Where does MJWarp's 10 s of per-scene setup actually go, and how much of it is avoidable?

Both prior investigations treated setup as a fixed tax and amortized it. It is not fixed. This
prices every stage and then attacks the four that look reducible:

  1. **Prune the model to what is DRAWN.** ``create_render_context`` builds a mesh BVH for every
     mesh in the model -- ``for mid in range(nmesh)`` -- but the scene BVH it feeds contains only
     geoms in ``enabled_geom_groups``. RoboCasa renders group 1, which uses ~165 of ~350 meshes;
     the rest are robosuite collision meshes. Half the BVH work is for geometry nobody draws.
  2. **Reuse the compiled model.** ``mj_saveModel`` -> ``.mjb`` was dismissed at 528 MB per EPISODE.
     Per KITCHEN it is a different trade, and the size is measured here against what is actually in
     it (textures + mesh data, which the asset probe already showed are 7.4x shared across scenes).
  3. **Hold contexts in a pool.** One context per kitchen, kept alive across a dataset pass. Memory
     per context and whether an older context still renders correctly after newer ones are built.
  4. **Overlap.** Setup for kitchen N+1 on a second thread while kitchen N renders.

Run in ~/venvs/se3d (mujoco 3.11 + mujoco-warp 3.11).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import threading
import time

os.environ.setdefault("WARP_CACHE_PATH", "/mnt/vast/home/randall.balestriero/.cache/warp")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def used_meshes(mjm, groups) -> np.ndarray:
    """Mesh ids reachable from a geom in an enabled group. The rest are never traversed."""
    import mujoco
    enabled = np.isin(mjm.geom_group, list(groups))
    ids = mjm.geom_dataid[enabled & (mjm.geom_type == mujoco.mjtGeom.mjGEOM_MESH)]
    return np.unique(ids[ids >= 0])


def hide_zero_alpha(mjm, hidden_group=3):
    """The one-line pre-pass MJWarp needs on this class of scene (robocasa.md §7)."""
    alpha = mjm.geom_rgba[:, 3].copy()
    has_mat = mjm.geom_matid >= 0
    alpha[has_mat] = mjm.mat_rgba[mjm.geom_matid[has_mat], 3]
    n = int((alpha <= 0).sum())
    mjm.geom_group[alpha <= 0] = hidden_group
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmls", nargs="*", default=None)
    ap.add_argument("--corpus", default=os.path.join(OUT, "corpus"))
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--nworld", type=int, default=256)
    ap.add_argument("--cam", default="robot0_agentview_left")
    ap.add_argument("--pool", type=int, default=4, help="contexts to hold live at once")
    ap.add_argument("--out", default=os.path.join(OUT, "setup.json"))
    args = ap.parse_args()

    import mujoco
    import torch
    import warp as wp
    import mujoco_warp as mjw
    from mujoco_warp._src import bvh as mjw_bvh
    from mujoco_warp._src import render_util as mjw_ru

    def gpu_mb():
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1e6

    xmls = args.xmls
    if not xmls:
        xmls = sorted(glob.glob(os.path.join(args.corpus, "*", "ep000.xml")))
    if not xmls:
        xmls = [os.path.join(OUT, "pinned_scene.xml")]
    print("scenes:", xmls, flush=True)

    rec = {"res": args.res, "nworld": args.nworld, "mujoco": mujoco.__version__,
           "mujoco_warp": getattr(mjw, "__version__", "?"), "warp": wp.config.version,
           "gpu": torch.cuda.get_device_name(0), "scenes": xmls}

    # ---------------------------------------------------------------- 0. compile / .mjb economics
    x0 = xmls[0]
    mujoco.MjModel.from_xml_path(x0)                       # warm the NFS asset reads
    t0 = time.perf_counter(); mjm = mujoco.MjModel.from_xml_path(x0)
    rec["compile_warm_s"] = time.perf_counter() - t0
    rec["model"] = dict(nq=int(mjm.nq), ngeom=int(mjm.ngeom), nmesh=int(mjm.nmesh),
                        nmeshface=int(mjm.nmeshface), ntex=int(mjm.ntex),
                        texdata_MB=float(mjm.tex_data.size / 1e6),
                        meshvert_MB=float(mjm.mesh_vert.nbytes / 1e6),
                        meshface_MB=float(mjm.mesh_face.nbytes / 1e6))
    mjb = os.path.join(OUT, "cache_scene.mjb")
    mujoco.mj_saveModel(mjm, mjb, None)
    rec["mjb_bytes"] = os.path.getsize(mjb)
    t0 = time.perf_counter(); mujoco.MjModel.from_binary_path(mjb)
    rec["mjb_load_s"] = time.perf_counter() - t0
    print("compile warm %.3fs  mjb %.1f MB  load %.3fs" %
          (rec["compile_warm_s"], rec["mjb_bytes"] / 1e6, rec["mjb_load_s"]), flush=True)

    groups = [1]
    rec["n_zero_alpha_hidden"] = hide_zero_alpha(mjm)
    um = used_meshes(mjm, groups)
    rec["meshes_total"] = int(mjm.nmesh)
    rec["meshes_used_by_group1"] = int(um.size)
    print(f"meshes {mjm.nmesh} total, {um.size} reachable from group 1", flush=True)

    cam_id = max(0, mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, args.cam))
    cam_active = [i == cam_id for i in range(mjm.ncam)]
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    # ---------------------------------------------------------------- 1. stage breakdown
    stage = {"mesh_bvh_s": 0.0, "mesh_bvh_used_s": 0.0, "tex_s": 0.0, "n_bvh": 0}
    real_bvh, real_tex = mjw_bvh.build_mesh_bvh, mjw_ru.create_warp_texture

    def timed_bvh(m, mid, **kw):
        t = time.perf_counter(); r = real_bvh(m, mid, **kw); wp.synchronize()
        dt = time.perf_counter() - t
        stage["mesh_bvh_s"] += dt
        stage["n_bvh"] += 1
        if mid in um:
            stage["mesh_bvh_used_s"] += dt
        return r

    def timed_tex(m, i):
        t = time.perf_counter(); r = real_tex(m, i); wp.synchronize()
        stage["tex_s"] += time.perf_counter() - t
        return r

    mjw_bvh.build_mesh_bvh = timed_bvh
    mjw_ru.create_warp_texture = timed_tex

    base = gpu_mb()
    t0 = time.perf_counter(); m = mjw.put_model(mjm); wp.synchronize()
    rec["put_model_s"] = time.perf_counter() - t0
    rec["put_model_MB"] = gpu_mb() - base
    t0 = time.perf_counter(); d = mjw.put_data(mjm, mjd, nworld=args.nworld); wp.synchronize()
    rec["put_data_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    rc = mjw.create_render_context(mjm, nworld=args.nworld, cam_res=(args.res, args.res),
                                   render_rgb=True, use_textures=True, use_shadows=True,
                                   render_skybox=True, enabled_geom_groups=groups,
                                   cam_active=cam_active)
    wp.synchronize()
    rec["render_ctx_s"] = time.perf_counter() - t0
    rec["render_ctx_stages"] = dict(stage)
    rec["render_ctx_other_s"] = rec["render_ctx_s"] - stage["mesh_bvh_s"] - stage["tex_s"]
    rec["mesh_bvh_wasted_s"] = stage["mesh_bvh_s"] - stage["mesh_bvh_used_s"]
    print(json.dumps({k: rec[k] for k in
                      ("put_model_s", "put_data_s", "render_ctx_s", "render_ctx_stages",
                       "render_ctx_other_s", "mesh_bvh_wasted_s")}, indent=1), flush=True)

    # reference frame + throughput on this context, for the pool correctness check
    def capture(m_, d_, rc_):
        def frame():
            mjw.kinematics(m_, d_); mjw.camlight(m_, d_)
            mjw.refit_bvh(m_, d_, rc_); mjw.render(m_, d_, rc_)
        frame(); wp.synchronize()
        with wp.ScopedCapture() as cap:
            frame()
        wp.synchronize()
        return cap.graph

    g = capture(m, d, rc)
    bgra = wp.to_torch(rc.rgb_data).view(torch.uint8).reshape(args.nworld, args.res, args.res, 4)
    wp.capture_launch(g); wp.synchronize()
    ref0 = bgra[0, ..., [2, 1, 0]].clone()

    n_it = 20
    wp.synchronize(); t0 = time.perf_counter()
    for _ in range(n_it):
        wp.capture_launch(g)
    wp.synchronize()
    rec["fps"] = args.nworld * n_it / (time.perf_counter() - t0)
    print("fps", rec["fps"], flush=True)

    # ---------------------------------------------------------------- 2. pruned setup
    # Skip BVHs for meshes no enabled geom can reach. A degenerate stand-in keeps every index valid.
    stage2 = {"mesh_bvh_s": 0.0, "tex_s": 0.0, "n_bvh": 0}

    def pruned_bvh(mm, mid, **kw):
        t = time.perf_counter()
        r = real_bvh(mm, mid, **kw) if mid in um else _stub_bvh()
        wp.synchronize()
        stage2["mesh_bvh_s"] += time.perf_counter() - t
        stage2["n_bvh"] += 1
        return r

    _stub_cache = {}

    def _stub_bvh():
        if "m" not in _stub_cache:
            pts = wp.array([wp.vec3(0.0, 0.0, 0.0), wp.vec3(1e-6, 0.0, 0.0),
                            wp.vec3(0.0, 1e-6, 0.0)], dtype=wp.vec3)
            idx = wp.array([0, 1, 2], dtype=wp.int32)
            _stub_cache["m"] = wp.Mesh(points=pts, indices=idx)
        return _stub_cache["m"], wp.vec3(1e-6, 1e-6, 1e-6)

    def timed_tex2(mm, i):
        t = time.perf_counter(); r = real_tex(mm, i); wp.synchronize()
        stage2["tex_s"] += time.perf_counter() - t
        return r

    mjw_bvh.build_mesh_bvh = pruned_bvh
    mjw_ru.create_warp_texture = timed_tex2
    t0 = time.perf_counter()
    rc_p = mjw.create_render_context(mjm, nworld=args.nworld, cam_res=(args.res, args.res),
                                     render_rgb=True, use_textures=True, use_shadows=True,
                                     render_skybox=True, enabled_geom_groups=groups,
                                     cam_active=cam_active)
    wp.synchronize()
    rec["render_ctx_pruned_s"] = time.perf_counter() - t0
    rec["render_ctx_pruned_stages"] = dict(stage2)
    g_p = capture(m, d, rc_p)
    bgra_p = wp.to_torch(rc_p.rgb_data).view(torch.uint8).reshape(args.nworld, args.res, args.res, 4)
    wp.capture_launch(g_p); wp.synchronize()
    dif = (bgra_p[0, ..., [2, 1, 0]].to(torch.int16) - ref0.to(torch.int16)).abs()
    rec["pruned_identical"] = bool(dif.max().item() == 0)
    rec["pruned_maxdiff"] = int(dif.max().item())
    print("pruned ctx %.2fs (vs %.2fs), identical=%s" %
          (rec["render_ctx_pruned_s"], rec["render_ctx_s"], rec["pruned_identical"]), flush=True)
    mjw_bvh.build_mesh_bvh, mjw_ru.create_warp_texture = real_bvh, real_tex

    # ---------------------------------------------------------------- 3. context pool
    pool, pool_rows = [(mjm, m, d, rc, g, bgra)], []
    for x in xmls[1:args.pool]:
        before = gpu_mb()
        t0 = time.perf_counter()
        mm = mujoco.MjModel.from_xml_path(x)
        t_c = time.perf_counter() - t0
        hide_zero_alpha(mm)
        dd0 = mujoco.MjData(mm); mujoco.mj_forward(mm, dd0)
        t0 = time.perf_counter()
        mw = mjw.put_model(mm)
        dw = mjw.put_data(mm, dd0, nworld=args.nworld)
        cid = max(0, mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_CAMERA, args.cam))
        rcx = mjw.create_render_context(mm, nworld=args.nworld, cam_res=(args.res, args.res),
                                        render_rgb=True, use_textures=True, use_shadows=True,
                                        render_skybox=True, enabled_geom_groups=groups,
                                        cam_active=[i == cid for i in range(mm.ncam)])
        gg = capture(mw, dw, rcx)
        wp.synchronize()
        t_gpu = time.perf_counter() - t0
        bg = wp.to_torch(rcx.rgb_data).view(torch.uint8).reshape(args.nworld, args.res, args.res, 4)
        pool.append((mm, mw, dw, rcx, gg, bg))
        pool_rows.append({"xml": x, "compile_s": t_c, "gpu_setup_s": t_gpu,
                          "gpu_MB": gpu_mb() - before, "gpu_MB_total": gpu_mb()})
        print("pool +", os.path.basename(os.path.dirname(x)), pool_rows[-1], flush=True)
    rec["pool"] = pool_rows
    rec["pool_size"] = len(pool)
    # the oldest context must still be correct after the others exist
    wp.capture_launch(pool[0][4]); wp.synchronize()
    d0 = (pool[0][5][0, ..., [2, 1, 0]].to(torch.int16) - ref0.to(torch.int16)).abs()
    rec["pool_first_ctx_unchanged"] = bool(d0.max().item() == 0)
    # and all of them must render
    for i, (_, mw, dw, rcx, gg, bg) in enumerate(pool):
        wp.capture_launch(gg)
    wp.synchronize()
    rec["pool_all_render"] = True
    print("pool ok, first unchanged:", rec["pool_first_ctx_unchanged"], flush=True)

    # ---------------------------------------------------------------- 4. overlap setup with render
    # Can a second thread build the next kitchen while this one renders? Warp is stream-based; a
    # separate thread issuing allocations and BVH builds competes for the same device, so the
    # question is whether the sum is less than the serial sum, not whether it is free.
    if len(xmls) > 1:
        xn = xmls[min(len(xmls) - 1, args.pool)] if len(xmls) > args.pool else xmls[1]
        n_render = 40
        wp.synchronize(); t0 = time.perf_counter()
        for _ in range(n_render):
            wp.capture_launch(g)
        wp.synchronize()
        t_render_only = time.perf_counter() - t0

        holder = {}

        def build():
            t = time.perf_counter()
            mm = mujoco.MjModel.from_xml_path(xn)
            hide_zero_alpha(mm)
            dd = mujoco.MjData(mm); mujoco.mj_forward(mm, dd)
            mw = mjw.put_model(mm)
            _ = mjw.put_data(mm, dd, nworld=args.nworld)
            cid = max(0, mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_CAMERA, args.cam))
            _ = mjw.create_render_context(mm, nworld=args.nworld, cam_res=(args.res, args.res),
                                          render_rgb=True, use_textures=True, use_shadows=True,
                                          render_skybox=True, enabled_geom_groups=groups,
                                          cam_active=[i == cid for i in range(mm.ncam)])
            wp.synchronize()
            holder["t"] = time.perf_counter() - t
            holder["m"] = mw

        wp.synchronize(); t0 = time.perf_counter()
        th = threading.Thread(target=build); th.start()
        for _ in range(n_render):
            wp.capture_launch(g)
        wp.synchronize()
        t_render_during = time.perf_counter() - t0
        th.join()
        t_wall = time.perf_counter() - t0
        rec["overlap"] = {"render_only_s": t_render_only, "render_with_builder_s": t_render_during,
                          "builder_s": holder.get("t"), "wall_s": t_wall,
                          "serial_sum_s": t_render_only + (holder.get("t") or 0.0),
                          "speedup": (t_render_only + (holder.get("t") or 0.0)) / t_wall}
        print("overlap:", json.dumps(rec["overlap"]), flush=True)

    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
