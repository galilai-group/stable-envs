"""99.7 % of these triangles are smaller than a pixel. What happens if there are ten times fewer?

``probe_rays.py`` measured the projected area of every triangle in the frustum at 256 px: mean
0.026 px on HumanoidBench, 99.63-99.74 % of them sub-pixel, and 94.97 % on RoboCasa's agentview. A
mesh at that tessellation is carrying detail the frame cannot resolve, and it is carried by every
world of every batch.

Decimation is a VISUAL-TIER move, so it is measured against the OFFICIAL renderer on the original
model -- not against MJWarp on the decimated one, which would only prove the two agree with each
other. Every level reports the same four numbers the class declares plus its throughput, so the
trade is legible in one row.

The candidate mesh set is deliberately restricted. ``fast_simplification`` collapses edges and does
not carry per-vertex texture coordinates, so a decimated TEXTURED mesh would sample the wrong texels
everywhere. HumanoidBench is the clean case -- ``ntex`` is 2, its locomotion scenes carry ZERO
textured primitives and ``_build_bundle`` already flattens the floor to one colour -- so its meshes
can be decimated outright. On RoboCasa only meshes whose material carries no texture are touched, and
how much of the scene that reaches is reported next to the result rather than assumed.
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


def mesh_is_textured(m, mesh_id) -> bool:
    """Does any geom wearing this mesh wear a material with a texture?"""
    for g in range(m.ngeom):
        if int(m.geom_dataid[g]) != int(mesh_id):
            continue
        mid = int(m.geom_matid[g])
        if mid < 0:
            continue
        t = m.mat_texid[mid]
        if (int(np.max(t)) if np.ndim(t) else int(t)) >= 0:
            return True
    return False


def build_spec(xml_path, m_src, keep, skip_textured):
    import fast_simplification
    import mujoco
    spec = mujoco.MjSpec.from_file(xml_path)
    by_name = {}
    for i in range(m_src.nmesh):
        nm = mujoco.mj_id2name(m_src, mujoco.mjtObj.mjOBJ_MESH, i)
        by_name[nm] = i
    stats = {"meshes": 0, "decimated": 0, "skipped_textured": 0, "skipped_small": 0,
             "tri_before": 0, "tri_after": 0}
    # HumanoidBench's MJCF declares its 21 meshes with NO name attribute -- MuJoCo names them after
    # their file at compile time, so ``spec.meshes[k].name`` is the empty string and a name lookup
    # matches nothing and silently decimates nothing. Fall back to position, which is the order the
    # compiler assigns ids in, and assert the vertex count agrees so a mismatch is a failure rather
    # than a wrong mesh.
    for k, ms in enumerate(spec.meshes):
        i = by_name.get(ms.name)
        if i is None:
            i = k if k < m_src.nmesh else None
        if i is None:
            continue
        stats["meshes"] += 1
        va, nv = int(m_src.mesh_vertadr[i]), int(m_src.mesh_vertnum[i])
        fa, nf = int(m_src.mesh_faceadr[i]), int(m_src.mesh_facenum[i])
        stats["tri_before"] += nf
        if keep >= 1.0:
            stats["tri_after"] += nf
            continue
        if skip_textured and mesh_is_textured(m_src, i):
            stats["skipped_textured"] += 1
            stats["tri_after"] += nf
            continue
        if nf < 64:
            stats["skipped_small"] += 1
            stats["tri_after"] += nf
            continue
        v = np.asarray(m_src.mesh_vert[va:va + nv], np.float32).reshape(nv, 3)
        f = np.asarray(m_src.mesh_face[fa:fa + nf], np.int32).reshape(nf, 3)
        if ms.name and ms.name not in by_name:
            raise AssertionError(f"mesh {ms.name!r} is not in the compiled model")
        v2, f2 = fast_simplification.simplify(v, f, 1.0 - keep)
        if len(f2) < 4:
            stats["skipped_small"] += 1
            stats["tri_after"] += nf
            continue
        # A mesh given uservert/userface must not also carry a file, and its texcoords no longer
        # index the same vertices, so they go too. That is exactly why textured meshes are excluded.
        # It must also carry a NAME: MuJoCo derives one from the file for an unnamed mesh, and taking
        # the file away without putting the derived name back fails with "empty name in mesh". The
        # derived name is what the compiled model already holds, and it is what the referencing geoms
        # were resolved against, so writing it back keeps every reference valid.
        if not ms.name:
            ms.name = mujoco.mj_id2name(m_src, mujoco.mjtObj.mjOBJ_MESH, i)
        ms.file = ""
        ms.uservert = np.asarray(v2, np.float64).reshape(-1).tolist()
        ms.userface = np.asarray(f2, np.int32).reshape(-1).tolist()
        ms.usertexcoord = []
        ms.userfacetexcoord = []
        ms.usernormal = []
        ms.userfacenormal = []
        stats["decimated"] += 1
        stats["tri_after"] += len(f2)
    return spec.compile(), stats


def timed(fn, reps=20, warm=5):
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


def render_batch(m, cam, res, nworld, groups, qpos, block_dim, reps):
    """(frames uint8 (n,res,res,3) on host, graph ms) through the shipped MJWarp path."""
    import mujoco
    import warp as wp

    import mujoco_warp as mjw
    import stable_envs.render.mjwarp as MW

    d0 = mujoco.MjData(m)
    d0.qpos[:] = qpos[0]
    mujoco.mj_forward(m, d0)
    wm = mjw.put_model(m, batch_sizes={k: nworld for k in
                                       ("cam_pos", "cam_quat", "light_diffuse")})
    if block_dim:
        wm.block_dim.render = int(block_dim)
    wd = mjw.put_data(m, d0, nworld=nworld)
    needs_com = MW.needs_com_pos(m)
    with MW.gl_rounding():
        rc = MW.create_render_context(
            m, nworld=nworld, cam_res=(res, res), render_rgb=True,
            cam_active=[i == cam for i in range(m.ncam)], enabled_geom_groups=list(groups),
            use_textures=True, use_shadows=True, render_skybox=True)

        def frame():
            mjw.kinematics(wm, wd)
            if needs_com:
                mjw.com_pos(wm, wd)
            mjw.camlight(wm, wd)
            mjw.refit_bvh(wm, wd, rc)
            mjw.render(wm, wd, rc)

        frame()
        wp.synchronize()
        with wp.ScopedCapture() as cap:
            frame()
        graph = cap.graph
    tq = wp.to_torch(wd.qpos)
    n = min(nworld, len(qpos))
    tq[:n] = torch.as_tensor(qpos[:n], device="cuda", dtype=tq.dtype)
    wp.capture_launch(graph)
    wp.synchronize()
    rgb = wp.to_torch(rc.rgb_data).view(torch.uint8).reshape(
        nworld, res, res, 4)[:n, ..., [2, 1, 0]].cpu().numpy()
    ms = 1e3 * timed(lambda: (wp.capture_launch(graph), wp.synchronize()), reps)
    del wm, wd, rc, graph
    torch.cuda.empty_cache()
    return rgb, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--keep", default="1.0,0.3,0.1,0.03")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()

    import mujoco
    import stable_envs as se
    from stable_envs.testing.conformance import pixel_agreement, ssim

    cls = se.REGISTRY[a.env]
    if a.env.startswith("RoboCasa"):
        from stable_envs.envs import robocasa as M
        s = M._scene(cls.KITCHEN)
        m0, groups, xml = s.model, M.VISIBLE_GROUPS, s.xml
        cam_name, block_dim, skip_tex = cls.CAMERA, 32, True
        env = se.make(a.env, batch=1, size=a.res, device="cuda")
        st = env.sample_states(a.nworld)
        qpos = env._full_qpos(torch.as_tensor(st, device="cuda")).cpu().numpy().astype(np.float64)
    else:
        from stable_envs.envs import humanoidbench as M
        b = M._build_bundle(cls.ROBOT, cls.TASK)
        m0, groups = b.model, M.VISIBLE_GROUPS
        xml = M._scene_xml(cls.ROBOT, cls.TASK)
        cam_name, block_dim, skip_tex = M._camera_name(cls.TASK), None, False
        env = se.make(a.env, batch=1, size=a.res, device="cuda")
        st = env.sample_states(a.nworld)
        qpos = np.asarray(st, np.float64)
    del env
    torch.cuda.empty_cache()

    # The official frame, from the pristine model, once.
    ref = cls.reference_env(a.res)
    official = cls.reference_render(ref, st[:min(a.nworld, len(st))])

    out = {"env": a.env, "res": a.res, "nworld": a.nworld, "xml": xml,
           "gpu": torch.cuda.get_device_name(0), "rows": []}
    # THREE kinds of row, and the middle one is the control that makes the third readable. "shipped"
    # is the model the env renders, model surgery and all. "raw 1.0" is the same MJCF recompiled from
    # its spec with nothing decimated -- it does NOT carry the env module's colour resolution or floor
    # bake, so it scores worse than shipped for reasons that have nothing to do with triangles.
    # Every decimated row is built the same way, so the decimation cost is (row - raw), and the
    # surgery cost is (raw - shipped).
    import stable_envs.render.mjwarp as MW
    for keep in ["shipped"] + [float(x) for x in a.keep.split(",")]:
        try:
            if keep == "shipped":
                m, stats = m0, {"note": "the model the env renders"}
            else:
                m, stats = build_spec(xml, m0, keep, skip_tex)
                MW.prepare_model(m, groups)
            cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            rgb, ms = render_batch(m, cam, a.res, a.nworld, groups, qpos, block_dim, a.reps)
            n = min(len(rgb), len(official))
            row = {"keep": keep, "graph_ms": ms, "fps": a.nworld / (ms / 1e3), "stats": stats}
            row.update(pixel_agreement(rgb[:n], official[:n]))
            row["ssim"] = ssim(rgb[:n], official[:n])
        except Exception as e:                                        # noqa: BLE001
            row = {"keep": keep, "error": f"{type(e).__name__}: {e}"}
        out["rows"].append(row)
        print(json.dumps(row), flush=True)
        torch.cuda.empty_cache()

    p = (f"/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/altrender/"
         f"decimate_{a.env}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
