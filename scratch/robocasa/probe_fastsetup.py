"""Attack MJWarp's per-scene setup cost directly, one term at a time.

Both prior investigations amortized the 10 s and neither asked what it is made of. Reading
``mujoco_warp`` 3.11's source gives three specific answers, each measured here against the same
RoboCasa kitchen so the reader can see which one carries the result:

  T1  **The texture upload is quadratic in the wrong variable.** ``render_util.create_warp_texture``
      does ``wp.array(mjm.tex_data, dtype=wp.uint8)`` *inside* the per-texture function, and
      ``mjm.tex_data`` is MuJoCo's single concatenated blob for ALL textures. RoboCasa has 76
      textures over 388.7 MB, so building a render context copies 76 x 388.7 MB = 29.5 GB
      host-to-device to use 388.7 MB of it. Hoisting that one array out is the whole fix.

  T2  **Mesh BVHs are built for every mesh in the model**, while the scene BVH only ever contains
      geoms in ``enabled_geom_groups``. RoboCasa renders group 1, which reaches ~165 of ~350
      meshes; the rest are robosuite collision hulls nothing draws.

  T3  **``put_model`` uploads a whole dynamics model to run a renderer.** The render kernels read
      about twenty ``Model`` fields (geom_type/dataid/matid/size/rgba, cam_*, light_*, mat_*,
      mesh_faceadr) and seven ``Data`` fields (geom_xpos/xmat, cam_xpos/xmat, light_xpos/xdir).
      MJX's own FFI shim already constructs exactly that. Here the cost is measured rather than
      re-implemented: ``put_model`` is timed against the size of what the renderer actually needs.

Also measured, because they change the amortization arithmetic rather than the setup:
  * a context POOL -- N kitchens live at once, memory each, and whether the oldest still renders;
  * OVERLAP -- building kitchen N+1 on a second thread while kitchen N renders.

Run in ~/venvs/se3d.
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


def hide_zero_alpha(mjm, hidden_group=3):
    a = mjm.geom_rgba[:, 3].copy()
    hm = mjm.geom_matid >= 0
    a[hm] = mjm.mat_rgba[mjm.geom_matid[hm], 3]
    n = int((a <= 0).sum())
    mjm.geom_group[a <= 0] = hidden_group
    return n


def used_meshes(mjm, groups):
    import mujoco
    en = np.isin(mjm.geom_group, list(groups))
    ids = mjm.geom_dataid[en & (mjm.geom_type == mujoco.mjtGeom.mjGEOM_MESH)]
    return set(int(i) for i in np.unique(ids[ids >= 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmls", nargs="*", default=None)
    ap.add_argument("--corpus", default=os.path.join(OUT, "corpus"))
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--nworld", type=int, default=256)
    ap.add_argument("--cam", default="robot0_agentview_left")
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(OUT, "fastsetup.json"))
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

    xmls = args.xmls or sorted(glob.glob(os.path.join(args.corpus, "*", "ep000.xml")))
    if not xmls:
        xmls = [os.path.join(OUT, "pinned_scene.xml")]
    print("scenes:", xmls, flush=True)

    rec = {"res": args.res, "nworld": args.nworld, "mujoco": mujoco.__version__,
           "mujoco_warp": getattr(mjw, "__version__", "?"), "warp": wp.config.version,
           "gpu": torch.cuda.get_device_name(0), "scenes": xmls}

    x0 = xmls[0]
    mujoco.MjModel.from_xml_path(x0)                                   # warm NFS
    t0 = time.perf_counter(); mjm = mujoco.MjModel.from_xml_path(x0)
    rec["compile_warm_s"] = time.perf_counter() - t0
    rec["n_zero_alpha_hidden"] = hide_zero_alpha(mjm)
    groups = [1]
    um = used_meshes(mjm, groups)
    rec["model"] = dict(nq=int(mjm.nq), ngeom=int(mjm.ngeom), nmesh=int(mjm.nmesh),
                        nmeshface=int(mjm.nmeshface), ntex=int(mjm.ntex),
                        texdata_MB=float(mjm.tex_data.size / 1e6),
                        meshes_used_by_group1=len(um))
    print(json.dumps(rec["model"]), "compile %.3fs" % rec["compile_warm_s"], flush=True)

    cam_id = max(0, mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, args.cam))
    cam_active = [i == cam_id for i in range(mjm.ncam)]
    mjd = mujoco.MjData(mjm); mujoco.mj_forward(mjm, mjd)

    real_bvh, real_tex = mjw_bvh.build_mesh_bvh, mjw_ru.create_warp_texture

    def ctx(nworld, patch_tex=False, prune_bvh=False, stats=None):
        """Build a render context with the named fixes applied, timed by stage."""
        st = stats if stats is not None else {}
        st.setdefault("bvh_s", 0.0); st.setdefault("tex_s", 0.0); st.setdefault("n_bvh", 0)
        blob = {}

        def tex_hoisted(m, i):
            t = time.perf_counter()
            if "a" not in blob:                          # T1: upload the blob ONCE, not per texture
                blob["a"] = wp.array(m.tex_data, dtype=wp.uint8)
            w, h = int(m.tex_width[i]), int(m.tex_height[i])
            out = wp.zeros((h, w, 4), dtype=float)
            wp.launch(mjw_ru._convert_texture_data, dim=(w, h),
                      inputs=[w, int(m.tex_adr[i]), int(m.tex_nchannel[i]), blob["a"]],
                      outputs=[out])
            r = wp.Texture2D(out, filter_mode=wp.TextureFilterMode.LINEAR)
            wp.synchronize(); st["tex_s"] += time.perf_counter() - t
            return r

        def tex_timed(m, i):
            t = time.perf_counter(); r = real_tex(m, i); wp.synchronize()
            st["tex_s"] += time.perf_counter() - t
            return r

        stub = {}

        def bvh_pruned(m, mid, **kw):
            t = time.perf_counter()
            if prune_bvh and mid not in um:              # T2: nothing enabled can reach this mesh
                if "m" not in stub:
                    stub["m"] = wp.Mesh(points=wp.array([wp.vec3(0.0, 0.0, 0.0),
                                                         wp.vec3(1e-6, 0.0, 0.0),
                                                         wp.vec3(0.0, 1e-6, 0.0)], dtype=wp.vec3),
                                        indices=wp.array([0, 1, 2], dtype=wp.int32))
                r = (stub["m"], wp.vec3(1e-6, 1e-6, 1e-6))
            else:
                r = real_bvh(m, mid, **kw)
            wp.synchronize(); st["bvh_s"] += time.perf_counter() - t; st["n_bvh"] += 1
            return r

        mjw_ru.create_warp_texture = tex_hoisted if patch_tex else tex_timed
        mjw_bvh.build_mesh_bvh = bvh_pruned
        t0 = time.perf_counter()
        rc = mjw.create_render_context(mjm, nworld=nworld, cam_res=(args.res, args.res),
                                       render_rgb=True, use_textures=True, use_shadows=True,
                                       render_skybox=True, enabled_geom_groups=groups,
                                       cam_active=cam_active)
        wp.synchronize()
        st["total_s"] = time.perf_counter() - t0
        st["other_s"] = st["total_s"] - st["bvh_s"] - st["tex_s"]
        mjw_bvh.build_mesh_bvh, mjw_ru.create_warp_texture = real_bvh, real_tex
        return rc, st

    def capture(m_, d_, rc_):
        def frame():
            mjw.kinematics(m_, d_); mjw.camlight(m_, d_)
            mjw.refit_bvh(m_, d_, rc_); mjw.render(m_, d_, rc_)
        frame(); wp.synchronize()
        with wp.ScopedCapture() as cap:
            frame()
        wp.synchronize()
        return cap.graph

    # ---------------------------------------------------------------- put_model / put_data
    base = gpu_mb()
    t0 = time.perf_counter(); m = mjw.put_model(mjm); wp.synchronize()
    rec["put_model_s"] = time.perf_counter() - t0
    rec["put_model_MB"] = gpu_mb() - base
    t0 = time.perf_counter(); d = mjw.put_data(mjm, mjd, nworld=args.nworld); wp.synchronize()
    rec["put_data_s"] = time.perf_counter() - t0
    if hasattr(mjw, "make_data"):
        t0 = time.perf_counter(); d2 = mjw.make_data(mjm, nworld=args.nworld); wp.synchronize()
        rec["make_data_s"] = time.perf_counter() - t0
        del d2
    else:
        rec["make_data_s"] = rec["put_data_s"]

    # ---------------------------------------------------------------- T0 baseline / T1 / T1+T2
    variants = {}
    for name, (pt, pb) in {"baseline": (False, False), "tex_hoisted": (True, False),
                           "tex_hoisted+bvh_pruned": (True, True)}.items():
        before = gpu_mb()
        rc, st = ctx(args.nworld, patch_tex=pt, prune_bvh=pb)
        st["gpu_MB"] = gpu_mb() - before
        g = capture(m, d, rc)
        bg = wp.to_torch(rc.rgb_data).view(torch.uint8).reshape(args.nworld, args.res, args.res, 4)
        wp.capture_launch(g); wp.synchronize()
        img = bg[0, ..., [2, 1, 0]].clone()
        n_it = 20
        wp.synchronize(); t0 = time.perf_counter()
        for _ in range(n_it):
            wp.capture_launch(g)
        wp.synchronize()
        st["fps"] = args.nworld * n_it / (time.perf_counter() - t0)
        variants[name] = st
        variants[name]["_img"] = img
        print(name, json.dumps({k: v for k, v in st.items() if not k.startswith("_")}), flush=True)

    ref = variants["baseline"].pop("_img")
    for name in ("tex_hoisted", "tex_hoisted+bvh_pruned"):
        img = variants[name].pop("_img")
        dd = (img.to(torch.int16) - ref.to(torch.int16)).abs()
        variants[name]["identical_to_baseline"] = bool(dd.max().item() == 0)
        variants[name]["maxdiff_vs_baseline"] = int(dd.max().item())
    rec["render_context"] = variants

    fast = variants["tex_hoisted+bvh_pruned"]
    rec["setup_baseline_s"] = (rec["compile_warm_s"] + rec["put_model_s"] + rec["put_data_s"]
                               + variants["baseline"]["total_s"])
    rec["setup_fast_s"] = (rec["compile_warm_s"] + rec["put_model_s"] + rec["make_data_s"]
                           + fast["total_s"])
    print("SETUP baseline %.2fs -> fast %.2fs" % (rec["setup_baseline_s"], rec["setup_fast_s"]),
          flush=True)

    # ---------------------------------------------------------------- pool of live contexts
    pool, rows = [(mjm, m, d)], []
    for x in xmls[1:args.pool]:
        before = gpu_mb()
        t0 = time.perf_counter()
        mm = mujoco.MjModel.from_xml_path(x)
        tc = time.perf_counter() - t0
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
        gg = capture(mw, dw, rcx); wp.synchronize()
        rows.append({"xml": x, "compile_s": tc, "gpu_setup_s": time.perf_counter() - t0,
                     "gpu_MB": gpu_mb() - before, "gpu_MB_total": gpu_mb()})
        pool.append((mm, mw, dw))
        print("pool +", os.path.basename(os.path.dirname(x)), json.dumps(rows[-1]), flush=True)
    rec["pool"] = rows

    # ---------------------------------------------------------------- overlap
    gbase = capture(m, d, rc)
    n_render = 60
    wp.synchronize(); t0 = time.perf_counter()
    for _ in range(n_render):
        wp.capture_launch(gbase)
    wp.synchronize()
    t_render = time.perf_counter() - t0
    xn = xmls[-1]
    hold = {}

    def build():
        t = time.perf_counter()
        mm = mujoco.MjModel.from_xml_path(xn)
        hide_zero_alpha(mm)
        dd = mujoco.MjData(mm); mujoco.mj_forward(mm, dd)
        _ = mjw.put_model(mm)
        cid = max(0, mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_CAMERA, args.cam))
        _ = mjw.create_render_context(mm, nworld=args.nworld, cam_res=(args.res, args.res),
                                      render_rgb=True, use_textures=True, use_shadows=True,
                                      render_skybox=True, enabled_geom_groups=groups,
                                      cam_active=[i == cid for i in range(mm.ncam)])
        wp.synchronize()
        hold["t"] = time.perf_counter() - t

    wp.synchronize(); t0 = time.perf_counter()
    th = threading.Thread(target=build); th.start()
    for _ in range(n_render):
        wp.capture_launch(gbase)
    wp.synchronize()
    th.join()
    wall = time.perf_counter() - t0
    rec["overlap"] = {"render_only_s": t_render, "builder_s": hold.get("t"), "wall_s": wall,
                      "serial_sum_s": t_render + (hold.get("t") or 0),
                      "speedup": (t_render + (hold.get("t") or 0)) / wall}
    print("overlap:", json.dumps(rec["overlap"]), flush=True)

    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
