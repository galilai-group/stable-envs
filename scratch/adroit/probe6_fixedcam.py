"""Is there a PURE official Adroit render, and if so what fraction of it can MJWarp draw?

Two questions, because the family needs both answers to be portable and the first one has a real
chance of rescuing it:

  Q1. Every Adroit XML defines <camera name="fixed"> and <camera name="vil_camera">, and gymnasium's
      MujocoEnv accepts camera_name=. A model camera is a constant of the model, so it should be
      immune to the median(geom_xpos) pathology that makes the DEFAULT free camera depend on when
      the viewer was first built. Tested the only way that settles it: one env per process, varying
      both the reset seed and the number of actions taken before the first render, all restoring the
      same canonical state.

  Q2. Given a pure camera, how much of the frame is geometry MJWarp cannot draw? From probe1 the
      undrawable set is: textured non-mesh/non-plane geoms (sample_texture derives UVs for PLANE and
      MESH only) and translucent geoms (the shader packs a constant alpha of 255). Split by whether
      the geom MOVES, because a static one is baked through MuJoCo itself and is exact regardless.
      probe1's coverage numbers were measured through a guessed camera and are void; these are
      measured through the camera actually under test.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import itertools
import json
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

REF_ID = {"Door": "AdroitHandDoor-v1", "Hammer": "AdroitHandHammer-v1",
          "Pen": "AdroitHandPen-v1", "Relocate": "AdroitHandRelocate-v1"}
PER_EPISODE = {"Door": "frame", "Hammer": "nail_board", "Pen": "target", "Relocate": "Object"}
SIZE = 480
TMP = "/mnt/vast/home/randall.balestriero/se_logs/adroit"
CAM = "fixed"
#: (reset seed, actions before the first render)
CASES = ((0, 0), (2, 0), (3, 0), (0, 25), (7, 60))


def worker(scene, seed, warmup, out_path):
    import gymnasium, gymnasium_robotics, mujoco                                # noqa: F401
    gymnasium.register_envs(gymnasium_robotics)
    env = gymnasium.make(REF_ID[scene], render_mode="rgb_array", width=SIZE, height=SIZE,
                         camera_name=CAM)
    u = env.unwrapped
    u.reset(seed=int(seed))
    env.action_space.seed(0)
    for _ in range(int(warmup)):
        u.step(env.action_space.sample())
    u.render()

    st = np.load(os.path.join(TMP, f"canon_{scene}.npz"))
    u.model.body_pos[:] = st["body_pos"]
    u.model.body_quat[:] = st["body_quat"]
    u.model.site_pos[:] = st["site_pos"]
    u.data.qpos[:] = st["qpos"]
    u.data.qvel[:] = st["qvel"]
    mujoco.mj_forward(u.model, u.data)
    frame = np.ascontiguousarray(u.render())
    c = u.mujoco_renderer.viewer.cam
    np.savez(out_path, frame=frame, camtype=int(c.type), fixedcamid=int(c.fixedcamid),
             lookat=np.asarray(c.lookat, np.float64))


def coverage(scene):
    """Pixel share of drawable / undrawable, moving / static, through the FIXED camera."""
    import mujoco
    import gymnasium, gymnasium_robotics                                        # noqa: F401
    from stable_envs.render import mjwarp as MW
    gymnasium.register_envs(gymnasium_robotics)

    env = gymnasium.make(REF_ID[scene])                    # no viewer
    u = env.unwrapped
    m = u.model
    u.reset(seed=0)
    env.action_space.seed(0)

    gname = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom{i}"
    pe_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PER_EPISODE[scene])
    subtree = set()
    for b in range(m.nbody):
        x = b
        while x > 0:
            if x == pe_bid:
                subtree.add(b); break
            x = int(m.body_parentid[x])

    groups = MW.DEFAULT_GROUPS
    visible = [g for g in range(m.ngeom) if int(m.geom_group[g]) in groups]
    alpha = MW.effective_alpha(m, "geom")
    gaps = {g for g, _, _ in MW.texture_gaps(m, groups)}
    translucent = {g for g in visible if 0.0 < alpha[g] < 1.0}
    invisible = {g for g in visible if alpha[g] <= 0.0}
    undrawable = (gaps | translucent) - invisible
    moves = [g for g in visible
             if int(m.body_weldid[m.geom_bodyid[g]]) != 0 or int(m.geom_bodyid[g]) in subtree
             or int(m.body_mocapid[m.geom_bodyid[g]]) >= 0]
    static = [g for g in visible if g not in moves]

    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAM)
    rend = mujoco.Renderer(m, width=SIZE, height=SIZE)
    cov = {k: 0.0 for k in ("moving_undrawable", "static_undrawable", "moving_ok", "static_ok",
                            "background")}
    n = 12
    for i in range(n):
        for _ in range(int(np.random.default_rng(i).integers(0, 26))):
            u.step(env.action_space.sample())
        rend.enable_segmentation_rendering()
        rend.update_scene(u.data, camera=cid)
        seg = rend.render()
        rend.disable_segmentation_rendering()
        ids, objs = seg[..., 0].astype(np.int64), seg[..., 1].astype(np.int64)
        gpix = ids == int(mujoco.mjtObj.mjOBJ_GEOM)
        tot = gpix.size
        sets = {"moving_undrawable": [g for g in moves if g in undrawable],
                "static_undrawable": [g for g in static if g in undrawable],
                "moving_ok": [g for g in moves if g not in undrawable],
                "static_ok": [g for g in static if g not in undrawable]}
        for k, sel in sets.items():
            a = np.asarray(sel, np.int64) if sel else np.zeros(0, np.int64)
            cov[k] += float((np.isin(objs, a) & gpix).sum() / tot) / n
        cov["background"] += float((~gpix).sum() / tot) / n
        u.reset(seed=200 + i)
    return {"coverage": {k: round(v, 5) for k, v in cov.items()},
            "moving_undrawable_geoms": sorted(gname(g) for g in moves if g in undrawable),
            "translucent": {gname(g): round(float(alpha[g]), 3) for g in sorted(translucent)}}


def drive():
    out = {}
    for scene in REF_ID:
        frames, meta = {}, {}
        for seed, warm in CASES:
            p = os.path.join(TMP, f"fix_{scene}_{seed}_{warm}.npz")
            subprocess.run([sys.executable, __file__, scene, str(seed), str(warm), p], check=True,
                           env={**os.environ, "MUJOCO_GL": "egl"})
            z = np.load(p)
            frames[(seed, warm)] = z["frame"]
            meta[(seed, warm)] = (int(z["camtype"]), int(z["fixedcamid"]))
        worst = {"pair": None, "differing_pixel_frac": 0.0, "max_abs": 0}
        n_id = 0
        for a, b in itertools.combinations(CASES, 2):
            d = np.abs(frames[a].astype(np.int16) - frames[b].astype(np.int16)).max(-1)
            if np.array_equal(frames[a], frames[b]):
                n_id += 1
            f = float((d > 0).mean())
            if f > worst["differing_pixel_frac"]:
                worst = {"pair": f"{a} vs {b}", "differing_pixel_frac": round(f, 5),
                         "max_abs": int(d.max())}
        res = {"camera": CAM, "cam_type_fixedcamid": {str(k): v for k, v in meta.items()},
               "n_case_pairs": len(list(itertools.combinations(CASES, 2))),
               "n_pairs_bit_identical": n_id, "worst_pair": worst}
        res.update(coverage(scene))
        out[scene] = res
        print(f"\n===== {scene} =====")
        print(json.dumps(res, indent=2))
    with open(os.path.join(TMP, "probe6.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    if sys.argv[1] == "--drive":
        drive()
    else:
        worker(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
