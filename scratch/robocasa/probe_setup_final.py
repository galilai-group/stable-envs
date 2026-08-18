"""The corrected per-scene setup, end to end, with every fix applied at once — and its break-even K.

Four separate measurements said setup is not the fixed 10 s tax both prior investigations amortized:

  * ``create_render_context`` re-uploads the whole concatenated texture blob once per texture
    (``probe_fastsetup.py``: 7.60 s -> 0.106 s, bit-identical);
  * it builds a mesh BVH for every mesh, including the ones only collision geoms reference
    (1.23 s -> 0.87 s, bit-identical);
  * ``put_model`` is dominated by the collision half of the model, not by fixed overhead
    (``probe_minsetup.py``: 3.70 s full vs 0.048 s on a model stripped to the drawn geoms, against
    0.012 s on a two-geom model — so it is content, and most of the content is never drawn);
  * ``put_data`` tiles a host ``MjData`` nworld times where ``make_data`` allocates (5.68 -> 0.98 s).

This runs the whole corrected path once and reports the number that goes in the table, plus the
break-even K it implies against a CPU core of JPEG decode.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("WARP_CACHE_PATH", "/mnt/vast/home/randall.balestriero/.cache/warp")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# Load the package's mjwarp module by PATH: importing `stable_envs.render.mjwarp` normally would
# execute `stable_envs/__init__.py`, which pulls in gymnasium, and the MJWarp venv deliberately has
# neither gymnasium nor torch-side env deps. The module itself has no package-internal imports.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "se_mjwarp", "/mnt/vast/home/randall.balestriero/code/stable-envs/stable_envs/render/mjwarp.py")
_se_mjwarp = _ilu.module_from_spec(_spec)
import sys as _sys
_sys.modules["se_mjwarp"] = _se_mjwarp      # @dataclass resolves annotations via sys.modules
_spec.loader.exec_module(_se_mjwarp)
prepare_model, texture_gaps = _se_mjwarp.prepare_model, _se_mjwarp.texture_gaps
create_render_context = _se_mjwarp.create_render_context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=os.path.join(OUT, "corpus", "L52S03", "ep000.xml"))
    ap.add_argument("--stripped", default=os.path.join(OUT, "stripped_scene.xml"))
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--nworld", type=int, default=256)
    ap.add_argument("--cam", default="robot0_agentview_left")
    ap.add_argument("--T", type=int, default=500, help="frames per episode")
    ap.add_argument("--decode-fps", type=float, default=5137.0)
    ap.add_argument("--out", default=os.path.join(OUT, "setup_final.json"))
    args = ap.parse_args()

    import mujoco
    import torch
    import warp as wp
    import mujoco_warp as mjw
    from mujoco_warp._src import bvh as mjw_bvh

    rec = {"res": args.res, "nworld": args.nworld, "T": args.T, "decode_fps": args.decode_fps}
    out = {}
    for tag, path in (("full", args.xml), ("stripped", args.stripped)):
        mujoco.MjModel.from_xml_path(path)                          # warm
        t0 = time.perf_counter(); m = mujoco.MjModel.from_xml_path(path)
        t_compile = time.perf_counter() - t0
        prep = prepare_model(m, groups=(1,))
        um = set(int(i) for i in np.unique(
            m.geom_dataid[(m.geom_group == 1) & (m.geom_type == mujoco.mjtGeom.mjGEOM_MESH)])
            if int(i) >= 0)
        real = mjw_bvh.build_mesh_bvh
        stub = {}

        def pruned(mm, mid, **kw):
            if mid in um:
                return real(mm, mid, **kw)
            if "m" not in stub:
                stub["m"] = wp.Mesh(points=wp.array([wp.vec3(0.0, 0.0, 0.0),
                                                     wp.vec3(1e-6, 0.0, 0.0),
                                                     wp.vec3(0.0, 1e-6, 0.0)], dtype=wp.vec3),
                                    indices=wp.array([0, 1, 2], dtype=wp.int32))
            return stub["m"], wp.vec3(1e-6, 1e-6, 1e-6)

        t0 = time.perf_counter(); mw = mjw.put_model(m); wp.synchronize()
        t_put_model = time.perf_counter() - t0
        t0 = time.perf_counter(); dw = mjw.make_data(m, nworld=args.nworld); wp.synchronize()
        t_data = time.perf_counter() - t0
        cid = max(0, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, args.cam))
        mjw_bvh.build_mesh_bvh = pruned
        t0 = time.perf_counter()
        rc = create_render_context(m, nworld=args.nworld, cam_res=(args.res, args.res),
                                   render_rgb=True, use_textures=True, use_shadows=True,
                                   render_skybox=True, enabled_geom_groups=[1],
                                   cam_active=[i == cid for i in range(m.ncam)])
        wp.synchronize()
        t_ctx = time.perf_counter() - t0
        mjw_bvh.build_mesh_bvh = real

        def frame():
            mjw.kinematics(mw, dw); mjw.camlight(mw, dw)
            mjw.refit_bvh(mw, dw, rc); mjw.render(mw, dw, rc)

        frame(); wp.synchronize()
        t0 = time.perf_counter()
        with wp.ScopedCapture() as cap:
            frame()
        wp.synchronize()
        t_cap = time.perf_counter() - t0
        bg = wp.to_torch(rc.rgb_data).view(torch.uint8).reshape(args.nworld, args.res, args.res, 4)
        q = torch.as_tensor(np.tile(np.load(args.xml.replace(".xml", "_qpos.npy"))[:m.nq],
                                    (args.nworld, 1)), dtype=torch.float32, device="cuda")
        wp.to_torch(dw.qpos).copy_(q)
        wp.capture_launch(cap.graph); wp.synchronize()
        img = bg[0, ..., [2, 1, 0]].clone()
        n = 30
        wp.synchronize(); t0 = time.perf_counter()
        for _ in range(n):
            wp.capture_launch(cap.graph)
        wp.synchronize()
        fps = args.nworld * n / (time.perf_counter() - t0)

        setup = t_compile + t_put_model + t_data + t_ctx + t_cap
        loop = args.T / fps
        dec = args.T / args.decode_fps
        out[tag] = dict(ngeom=int(m.ngeom), nmesh=int(m.nmesh), ntex=int(m.ntex),
                        hidden_geoms=len(prep.hidden_geoms), hidden_sites=len(prep.hidden_sites),
                        compile_s=t_compile, put_model_s=t_put_model, data_s=t_data,
                        render_ctx_s=t_ctx, graph_capture_s=t_cap, setup_s=setup, fps=fps,
                        loop_s_per_episode=loop, decode_s_per_episode=dec,
                        cores_at_K_inf=dec / loop,
                        break_even_K=setup / max(dec - loop, 1e-9))
        out[tag]["_img"] = img
        print(tag, json.dumps({k: v for k, v in out[tag].items() if not k.startswith("_")},
                              indent=1), flush=True)
        del rc, mw, dw

    d = (out["stripped"]["_img"].to(torch.int16) - out["full"]["_img"].to(torch.int16)).abs()
    rec["stripped_identical_to_full"] = bool(d.max().item() == 0)
    rec["stripped_maxdiff"] = int(d.max().item())
    for k in out:
        out[k].pop("_img")
    rec["variants"] = out
    print("stripped == full:", rec["stripped_identical_to_full"], flush=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
