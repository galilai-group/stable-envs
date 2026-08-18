"""Two fresh ogbench envs, driven to the SAME qpos, render differently. Find out where and why.

``test_state_is_sufficient`` builds two reference instances and compares. It fails by up to 44/255 on
every maze cell, while a single instance driven through two different histories is bit-identical -- so
whatever differs is a property of the INSTANCE, not of the state. This localises it: same qpos, two
models, diff the frames, report which pixels move and which model fields differ.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out"
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")


def run(loco="point", maze="medium", size=224):
    import mujoco
    from stable_envs.envs.ogbench import _build_reference
    a = _build_reference(loco, maze, size)
    b = _build_reference(loco, maze, size)
    ua, ub = a.unwrapped, b.unwrapped
    ma, mb = ua.model, ub.model
    q = np.asarray(ma.qpos0, np.float64).copy()
    q[:2] = [2.0, 6.0]
    fa, fb = [], []
    for u, out in ((ua, fa), (ub, fb)):
        u.set_state(q, np.zeros(u.model.nv))
        out.append(u.render().copy())
    A, B = fa[0], fb[0]
    d = np.abs(A.astype(np.int16) - B.astype(np.int16)).max(-1)
    res = dict(env=f"{loco}maze-{maze}", max=int(d.max()), frac=float((d > 0).mean()))
    ij = np.argwhere(d > 0)
    if len(ij):
        res["bbox"] = [int(ij[:, 0].min()), int(ij[:, 0].max()),
                       int(ij[:, 1].min()), int(ij[:, 1].max())]
        i, j = ij[0]
        res["example"] = dict(px=[int(i), int(j)], a=A[i, j].tolist(), b=B[i, j].tolist())
        # which geom owns the differing pixels?
        r = mujoco.Renderer(ma, width=size, height=size)
        r.enable_segmentation_rendering()
        r.update_scene(ua.data, camera=ua.custom_camera)
        seg = r.render()
        gid = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        names = {}
        for g in np.unique(gid[d > 0]):
            names[int(g)] = mujoco.mj_id2name(ma, mujoco.mjtObj.mjOBJ_GEOM, int(g)) if g >= 0 else "bg"
        res["geoms_of_differing_px"] = names

    # do the two MODELS differ anywhere that could matter?
    diffs = {}
    for f in ("geom_pos", "geom_size", "geom_rgba", "geom_quat", "mat_rgba", "light_pos", "light_dir",
              "light_ambient", "light_diffuse", "light_specular", "qpos0", "cam_pos", "cam_fovy",
              "body_pos", "body_quat"):
        if not hasattr(ma, f):
            continue
        x, y = np.asarray(getattr(ma, f)), np.asarray(getattr(mb, f))
        if x.shape == y.shape and not np.array_equal(x, y):
            diffs[f] = float(np.abs(x - y).max())
    attr = "tex_rgb" if hasattr(ma, "tex_rgb") else "tex_data"
    ta, tb = np.asarray(getattr(ma, attr)), np.asarray(getattr(mb, attr))
    if ta.shape == tb.shape and not np.array_equal(ta, tb):
        diffs[attr] = dict(max=int(np.abs(ta.astype(int) - tb.astype(int)).max()),
                           n_bytes_differ=int((ta != tb).sum()), n_bytes=int(ta.size))
        # which texture region?
        for t in range(ma.ntex):
            a0 = int(ma.tex_adr[t])
            n = 3 * int(ma.tex_height[t]) * int(ma.tex_width[t])
            nd = int((ta[a0:a0 + n] != tb[a0:a0 + n]).sum())
            if nd:
                diffs.setdefault("tex_by_name", {})[
                    mujoco.mj_id2name(ma, mujoco.mjtObj.mjOBJ_TEXTURE, t) or f"tex{t}"] = nd
    res["model_field_diffs"] = diffs
    res["cam_a"] = dict(lookat=ua.custom_camera.lookat.tolist(),
                        distance=float(ua.custom_camera.distance),
                        azimuth=float(ua.custom_camera.azimuth),
                        elevation=float(ua.custom_camera.elevation))
    res["cam_b"] = dict(lookat=ub.custom_camera.lookat.tolist(),
                        distance=float(ub.custom_camera.distance),
                        azimuth=float(ub.custom_camera.azimuth),
                        elevation=float(ub.custom_camera.elevation))
    return res


def stale_pose_check(loco="ant", maze="large", size=224):
    """Is ``data.xpos`` stale after ``set_state``? The PushT-class bug, tested directly.

    If the state-restore path wrote ``qpos`` without a forward pass, ``data.xpos`` — which is what the
    renderer draws — would still hold the pose from the PREVIOUS qpos. That predicts three things, all
    checked here: ``xpos`` after ``set_state`` disagrees with a fresh ``mj_kinematics``; the same qpos
    renders differently depending on which qpos preceded it; and A,B,A gives frame 3 != frame 1.
    """
    import inspect

    import mujoco
    from gymnasium.envs.mujoco import MujocoEnv
    from stable_envs.envs.ogbench import _build_reference
    env = _build_reference(loco, maze, size)
    u = env.unwrapped
    m, d = u.model, u.data
    src = inspect.getsource(MujocoEnv.set_state)
    res = dict(env=f"{loco}maze-{maze}",
               set_state_calls_forward=("mj_forward" in src),
               set_state_source=src.strip().splitlines()[-3:])

    rng = np.random.default_rng(0)
    qa = np.asarray(m.qpos0, np.float64).copy()
    qa[:2] = [2.0, 6.0]
    if m.nq > 7:
        qa[7:] += rng.uniform(-0.3, 0.3, m.nq - 7)
    qb = np.asarray(m.qpos0, np.float64).copy()
    qb[:2] = [14.0, 18.0]
    if m.nq > 7:
        qb[7:] += rng.uniform(-0.3, 0.3, m.nq - 7)

    # 1. does xpos agree with a fresh kinematics pass right after set_state?
    u.set_state(qb, np.zeros(m.nv))
    u.set_state(qa, np.zeros(m.nv))
    xpos_after_set = d.geom_xpos.copy()
    d.qpos[:] = qa
    mujoco.mj_kinematics(m, d)
    res["xpos_stale_by"] = float(np.abs(xpos_after_set - d.geom_xpos).max())

    # 2. A, B, A -- frame 3 must equal frame 1
    frames = []
    for q in (qa, qb, qa):
        u.set_state(q, np.zeros(m.nv))
        frames.append(u.render().copy())
    res["ABA_max_diff"] = int(np.abs(frames[0].astype(np.int16) - frames[2].astype(np.int16)).max())

    # 3. same qpos reached from two different predecessors
    u.set_state(qb, np.zeros(m.nv))
    u.set_state(qa, np.zeros(m.nv))
    f1 = u.render().copy()
    for _ in range(7):
        env.step(u.action_space.sample())
    u.set_state(qa, np.zeros(m.nv))
    f2 = u.render().copy()
    res["different_predecessor_max_diff"] = int(np.abs(f1.astype(np.int16) - f2.astype(np.int16)).max())
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "two_instances.log"), "w")
    for loco, maze in (("ant", "large"), ("point", "medium"), ("humanoid", "large")):
        try:
            log.write(json.dumps(stale_pose_check(loco, maze), indent=1) + "\n")
            log.flush()
        except Exception:  # noqa: BLE001
            import traceback
            log.write(f"FAIL stale {loco}-{maze}\n{traceback.format_exc()}\n")
            log.flush()
    for loco, maze in (("point", "medium"), ("point", "teleport"), ("ant", "large")):
        try:
            r = run(loco, maze)
        except Exception:  # noqa: BLE001
            import traceback
            log.write(f"FAIL {loco}-{maze}\n{traceback.format_exc()}\n")
            log.flush()
            continue
        log.write(json.dumps(r, indent=1) + "\n")
        log.flush()
    log.close()


if __name__ == "__main__":
    main()
