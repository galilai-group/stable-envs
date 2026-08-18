"""Four follow-ups the main study leaves open.

  (a) BAKED-STATIC + ANALYTIC-AGENT. The top-down camera is *fixed per env*, so floor, walls, teleport
      pads and the skybox are a constant image. Bake that once from MuJoCo itself and only ray-cast the
      agent per frame: the static part is then exact by construction and the only thing under test is the
      agent composite. This is the shape the real renderer would take, so it is the number that matters.
  (b) The official ``visual-*`` OGBench envs do NOT use the top-down camera -- they use the ``back``
      camera, which tracks the agent's centre of mass, at 64x64. Same geometry question, different pose:
      re-run the segmentation check there.
  (c) Cross-node determinism: is MuJoCo's own EGL raster the same on two different GPUs? If it is not,
      "bit-exact" is not even well defined and only interior agreement can be claimed.
  (d) What a state->frame renderer owes beyond the raster: forward kinematics. Timed against the render.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raycast as RC  # noqa: E402
from shade import normals  # noqa: E402
from analytic_rgb import AMB, DIFF, SPEC, SHINE, LDIR, base_rgb, stats  # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out"
INF = RC.INF


def make2(loco, size, img=224, camera_name=None, decolor=True):
    import gymnasium
    import ogbench  # noqa: F401
    kw = dict(render_mode="rgb_array", ob_type="pixels", width=img, height=img)
    if camera_name:
        kw["camera_name"] = camera_name
    env = gymnasium.make(f"{loco}maze-{size}-v0", **kw)
    env.reset(options={"task_id": 1})
    u = env.unwrapped
    if decolor:
        sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/counterfactual-WM")
        from rjepa.envs.ogbench_maze import decolor_floor, mark_agent_red
        decolor_floor(u)
        if loco == "humanoid":
            mark_agent_red(u, scale=4.0)
    return env, u


def agent_geoms(m):
    import mujoco
    names = {g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)}
    return [g for g in range(m.ngeom)
            if not (names[g] or "").startswith(("block_", "teleport")) and names[g] != "floor"], names


def cast_subset(m, d, cam, H, W, geoms):
    import mujoco
    o, dr, fwd = RC.rays(cam, H, W)
    best_t = np.full(len(o), INF)
    best_g = np.full(len(o), -1, np.int32)
    for g in geoms:
        if m.geom_rgba[g][3] <= 0:
            continue
        gt = int(m.geom_type[g])
        pos = d.geom_xpos[g].astype(np.float64)
        mat = d.geom_xmat[g].reshape(3, 3).astype(np.float64)
        sz = m.geom_size[g].astype(np.float64)
        if gt == mujoco.mjtGeom.mjGEOM_PLANE:
            t = RC.hit_plane(o, dr, pos, mat, sz)
        elif gt == mujoco.mjtGeom.mjGEOM_BOX:
            t = RC.hit_box(o, dr, pos, mat, sz)
        elif gt == mujoco.mjtGeom.mjGEOM_SPHERE:
            t = RC.hit_sphere(o, dr, pos, sz[0])
        elif gt == mujoco.mjtGeom.mjGEOM_CAPSULE:
            t = RC.hit_capsule(o, dr, pos, mat, sz[0], sz[1])
        elif gt == mujoco.mjtGeom.mjGEOM_CYLINDER:
            t = RC.hit_cylinder(o, dr, pos, mat, sz[0], sz[1])
        else:
            continue
        upd = t < best_t
        best_t = np.where(upd, t, best_t)
        best_g = np.where(upd, g, best_g)
    return (best_g.reshape(H, W), best_t.reshape(H, W), o.reshape(H, W, 3), dr.reshape(H, W, 3))


def a_baked(loco, size, img=224, nstates=16):
    """(a) baked static background + analytic agent composite."""
    import mujoco
    import imageio.v2 as imageio
    env, u = make2(loco, size, img)
    m, d = u.model, u.data
    m.vis.quality.offsamples = 0
    r = mujoco.Renderer(m, width=img, height=img)
    r.update_scene(d, camera=u.custom_camera)
    cam = RC.avgcam(RC.glcam(r.scene, 0), RC.glcam(r.scene, 1))
    ag, names = agent_geoms(m)
    static = [g for g in range(m.ngeom) if g not in ag]

    # bake: hide the agent (alpha 0) and let MuJoCo draw the static scene once
    keep = m.geom_rgba[ag].copy()
    m.geom_rgba[ag, 3] = 0.0
    r2 = mujoco.Renderer(m, width=img, height=img)
    r2.update_scene(d, camera=u.custom_camera)
    bg_rgb = r2.render().copy()
    m.geom_rgba[ag] = keep
    _, bg_t, o, dr = cast_subset(m, d, cam, img, img, static)

    V = -dr / np.linalg.norm(dr, axis=-1, keepdims=True)
    hv = LDIR + V
    hv = hv / np.linalg.norm(hv, axis=-1, keepdims=True)

    acc, accint = [], []
    for s in range(nstates):
        for _ in range(11):
            env.step(u.action_space.sample())
        r.update_scene(d, camera=u.custom_camera)
        ref = r.render().copy()
        gid, t, _, _ = cast_subset(m, d, cam, img, img, ag)
        vis = (t < INF / 2) & (t < bg_t)
        out = bg_rgb.astype(np.float64).copy()
        if vis.any():
            p = o + t[..., None] * dr
            n = normals(m, d, np.where(vis, gid, -1), p)
            ndl = np.clip(n @ LDIR, 0, None)
            ndh = np.clip((n * hv).sum(-1), 0, None)
            S = AMB + DIFF * ndl + SPEC * ndh ** SHINE
            base = np.zeros((img, img, 3))
            for g in np.unique(gid[vis]):
                base[vis & (gid == g)] = base_rgb(m, int(g))
            col = np.clip(base * S[..., None], 0, 1) * 255.0
            out[vis] = col[vis]
        mine = np.clip(np.round(out), 0, 255).astype(np.uint8)
        acc.append(stats(mine, ref))
        b = np.zeros(vis.shape, bool)
        b[1:, :] |= vis[1:, :] != vis[:-1, :]
        b[:-1, :] |= vis[1:, :] != vis[:-1, :]
        b[:, 1:] |= vis[:, 1:] != vis[:, :-1]
        b[:, :-1] |= vis[:, 1:] != vis[:, :-1]
        dd = np.abs(mine.astype(np.int16) - ref.astype(np.int16)).max(-1)
        accint.append(dict(agent_px=int(vis.sum()), boundary_px=int(b.sum()),
                           exact_nonboundary=float((dd[~b] == 0).mean()),
                           bad_nonboundary=int((dd[~b] > 0).sum()),
                           max_nonboundary=int(dd[~b].max())))
        if s == 0:
            imageio.imwrite(os.path.join(OUT, f"baked_{loco}_{size}.png"),
                            np.concatenate([ref, mine, np.clip(np.abs(mine.astype(np.int16) -
                                                                     ref.astype(np.int16)) * 8, 0, 255)
                                            .astype(np.uint8)], 1))
    env.close()
    return dict(env=f"{loco}maze-{size}", mode="baked_static+analytic_agent", nstates=nstates,
                vs_mujoco_noaa={k: round(float(np.mean([x[k] for x in acc])), 6) for k in acc[0]},
                interior={k: round(float(np.mean([x[k] for x in accint])), 4) for k in accint[0]})


def b_back_camera(loco, size, img=64, nstates=6):
    """(b) the official visual-* configuration: 'back' tracking camera, 64x64."""
    import mujoco
    import imageio.v2 as imageio
    env, u = make2(loco, size, img, camera_name="back", decolor=False)
    m, d = u.model, u.data
    m.vis.quality.offsamples = 0
    r = mujoco.Renderer(m, width=img, height=img)
    res = dict(env=f"{loco}maze-{size}", camera="back", img=img)
    per = []
    for s in range(nstates):
        for _ in range(11):
            env.step(u.action_space.sample())
        r.update_scene(d, camera=u.custom_camera)
        cam = RC.avgcam(RC.glcam(r.scene, 0), RC.glcam(r.scene, 1))
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=u.custom_camera)
        seg = r.render()
        r.disable_segmentation_rendering()
        gt = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1).astype(np.int32)
        gid, t, _, _ = RC.cast(m, d, cam, img, img)[:2] + (None, None)
        gid = np.where(t >= INF / 2, -1, gid)
        b = np.zeros(gt.shape, bool)
        b[1:, :] |= gt[1:, :] != gt[:-1, :]
        b[:-1, :] |= gt[1:, :] != gt[:-1, :]
        b[:, 1:] |= gt[:, 1:] != gt[:, :-1]
        b[:, :-1] |= gt[:, 1:] != gt[:, :-1]
        per.append(dict(cam_pos=[round(x, 3) for x in cam["pos"]],
                        cam_fwd=[round(x, 3) for x in cam["forward"]],
                        id_all=float((gid == gt).mean()),
                        id_interior=float((gid == gt)[~b].mean()),
                        interior_bad=int(((gid != gt) & ~b).sum()),
                        bg_frac=float((gt < 0).mean())))
        if s == 0:
            r.update_scene(d, camera=u.custom_camera)
            imageio.imwrite(os.path.join(OUT, f"back_{loco}_{size}.png"), r.render())
    res["per_state"] = per
    res["id_interior_mean"] = float(np.mean([p["id_interior"] for p in per]))
    res["interior_bad_mean"] = float(np.mean([p["interior_bad"] for p in per]))
    env.close()
    return res


def c_determinism(loco="point", size="large", img=224, n=8):
    """(c) hash of MuJoCo frames for a fixed state sequence; compare across nodes/jobs."""
    import mujoco
    env, u = make2(loco, size, img)
    m, d = u.model, u.data
    rng = np.random.default_rng(0)
    hs = []
    for i in range(n):
        qpos = np.asarray(u.init_qpos, np.float64).copy()
        qpos[:2] = rng.uniform(-4, 20, 2)
        if m.nq > 2:
            qpos[2:] = np.asarray(u.init_qpos, np.float64)[2:] + rng.normal(0, 0.05, m.nq - 2)
        u.set_state(qpos, np.zeros(m.nv))
        f = u.render()
        hs.append(hashlib.sha256(np.ascontiguousarray(f)).hexdigest()[:16])
    env.close()
    return dict(host=socket.gethostname(), env=f"{loco}maze-{size}", hashes=hs)


def d_fk_cost(loco, size, img=224, n=500):
    """(d) mj_kinematics (state -> geom poses) versus a full MuJoCo render."""
    import mujoco
    env, u = make2(loco, size, img)
    m, d = u.model, u.data
    r = mujoco.Renderer(m, width=img, height=img)
    for _ in range(5):
        r.update_scene(d, camera=u.custom_camera)
        r.render()
    t0 = time.time()
    for _ in range(n):
        mujoco.mj_kinematics(m, d)
    t_fk = (time.time() - t0) / n
    t0 = time.time()
    for _ in range(n // 5):
        r.update_scene(d, camera=u.custom_camera)
        r.render()
    t_r = (time.time() - t0) / (n // 5)
    env.close()
    return dict(env=f"{loco}maze-{size}", fk_us=t_fk * 1e6, render_ms=t_r * 1e3,
                fk_fps=1.0 / t_fk, render_fps=1.0 / t_r)


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {}
    print("== (a) baked static + analytic agent ==", flush=True)
    out["baked"] = []
    for loco, size in [("point", "medium"), ("point", "large"), ("point", "giant"),
                       ("point", "teleport"), ("ant", "large"), ("ant", "giant"),
                       ("humanoid", "large")]:
        try:
            r = a_baked(loco, size)
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            continue
        out["baked"].append(r)
        print(json.dumps(r), flush=True)

    print("\n== (b) back tracking camera (official visual-*) ==", flush=True)
    out["back"] = []
    for loco, size in [("ant", "large"), ("humanoid", "large")]:
        try:
            r = b_back_camera(loco, size)
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            continue
        out["back"].append(r)
        print(json.dumps(r), flush=True)

    print("\n== (c) determinism ==", flush=True)
    out["determinism"] = c_determinism()
    print(json.dumps(out["determinism"]), flush=True)

    print("\n== (d) FK vs render ==", flush=True)
    out["fk"] = []
    for loco, size in [("point", "large"), ("ant", "large"), ("humanoid", "large")]:
        try:
            r = d_fk_cost(loco, size)
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            continue
        out["fk"].append(r)
        print(json.dumps(r), flush=True)

    with open(os.path.join(OUT, f"general_{socket.gethostname()}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
