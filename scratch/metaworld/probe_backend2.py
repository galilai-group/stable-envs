"""The two questions ``probe_backend.py`` left open for Meta-World.

1. **Is the plate constant within an episode?** ``probe_backend.py`` drew each pool state from a
   fresh ``reset``, so its plate drift of 255 only says the plate is a function of the per-episode
   placement — which it must be, since Meta-World writes that placement into the model. The design
   question is different: with the placement HELD, does the movers-hidden plate stay constant as
   ``qpos`` moves? A plate that is constant per episode is baked once per reset and amortised over
   the episode, which is the Fetch pattern; a plate that moves with ``qpos`` is not a plate at all.

2. **Does MJWarp survive the cube textures, and how close does it get?** Every Meta-World model
   carries ``mjTEXTURE_CUBE`` slots, which is what hard-faults MJWarp with CUDA 700 on Franka
   Kitchen (``docs/research/kitchen.md`` Gate 1). Whether that fault fires here is a fact about
   which cube textures are REACHED, not about their presence, so it is measured rather than
   inferred. If it renders, the numbers that matter are full-scene interior agreement and the
   composite the port would actually ship: MuJoCo's baked plate under MJWarp's movers.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scratch.metaworld.probe_metaworld import make                          # noqa: E402
from stable_envs.render import mjwarp as MW                                 # noqa: E402
from stable_envs.testing.conformance import pixel_agreement                 # noqa: E402

SIZE = 224
OUT = "/mnt/vast/home/randall.balestriero/se_logs/metaworld"
SUBSET = ["reach-v3", "pick-place-v3", "door-open-v3", "button-press-v3", "drawer-open-v3",
          "window-open-v3", "hammer-v3", "assembly-v3"]


def movers_of(m, extra_bodies=()):
    vis = [g for g in range(m.ngeom) if int(m.geom_group[g]) in MW.DEFAULT_GROUPS]
    mv = {g for g in vis if int(m.body_weldid[m.geom_bodyid[g]]) != 0
          or int(m.body_mocapid[m.geom_bodyid[g]]) >= 0}
    mv |= {g for g in vis if int(m.geom_bodyid[g]) in set(extra_bodies)}
    return sorted(mv), [g for g in vis if g not in mv]


def plate_within_episode(task: str, n: int = 12, steps: int = 6) -> dict:
    """Placement HELD (one reset, then only qpos moves). Does the movers-hidden plate move?"""
    env = make(task, SIZE)
    u = env.unwrapped
    m, d = u.model, u.data
    env.reset()
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "corner")
    r = mujoco.Renderer(m, height=SIZE, width=SIZE)
    rng = np.random.default_rng(0)

    # bodies whose model rows this episode's reset wrote, found by diffing against a second episode
    ref = np.array(m.body_pos, copy=True), np.array(m.body_quat, copy=True)
    keep_model = {a: np.array(getattr(m, a), copy=True)
                  for a in ("body_pos", "body_quat", "site_pos", "geom_pos", "geom_size")}
    env.reset()
    moved = {b for b in range(m.nbody)
             if not np.array_equal(ref[0][b], m.body_pos[b])
             or not np.array_equal(ref[1][b], m.body_quat[b])}
    for a, v in keep_model.items():                       # back to episode 1's placement
        getattr(m, a)[:] = v
    movers, _ = movers_of(m, moved)

    opt_nosite = mujoco.MjvOption()
    opt_nosite.sitegroup[:] = 0

    def plate():
        keep = np.array(m.geom_group, copy=True)
        m.geom_group[np.asarray(movers, np.int64)] = MW.HIDDEN_GROUP
        r.update_scene(d, camera=cam, scene_option=opt_nosite)
        img = r.render().copy()
        m.geom_group[:] = keep
        return img

    plates = []
    for _ in range(n):
        for _ in range(steps):
            env.step(rng.uniform(-1, 1, size=u.action_space.shape).astype(np.float32))
        for a, v in keep_model.items():                   # step() cannot move the placement, but
            getattr(m, a)[:] = v                          # asserting it costs nothing
        mujoco.mj_forward(m, d)
        plates.append(plate())
    p = np.stack(plates)
    dr = np.abs(p.astype(np.int16) - p[0].astype(np.int16)).max(-1)
    env.close()
    return {"task": task, "n_movers": len(movers), "moved_bodies": len(moved),
            "plate_drift_max": int(dr.max()), "plate_drift_frac": round(float((dr > 0).mean()), 6)}


def mjwarp_parity(task: str, n: int = 6, batch: int = 16) -> dict:
    """Does MJWarp draw a Meta-World scene at all, and how close is it?"""
    import warp as wp

    env = make(task, SIZE)
    u = env.unwrapped
    m, d = u.model, u.data
    env.reset()
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "corner")

    ttype = np.asarray(m.tex_type, np.int64)
    matid = np.asarray(m.mat_texid, np.int64)
    reached = {int(t) for t in matid.ravel() if t >= 0}
    rec = {"task": task,
           "n_tex_by_type": {int(t): int((ttype == t).sum()) for t in np.unique(ttype)},
           "cube_textures_reached_by_a_material":
               sorted(int(ttype[t]) for t in reached if ttype[t] == 1)}

    rng = np.random.default_rng(0)
    qs = []
    for _ in range(n):
        for _ in range(6):
            env.step(rng.uniform(-1, 1, size=u.action_space.shape).astype(np.float32))
        qs.append(np.array(d.qpos, copy=True))

    r = mujoco.Renderer(m, height=SIZE, width=SIZE)
    refs = []
    for q in qs:
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam)
        refs.append(r.render().copy())
    refs = np.stack(refs)

    import torch
    br = None
    # ``put_model`` validates the model's COLLISION configuration even when the caller only wants a
    # render context, and refuses every Meta-World model: the Sawyer gripper pads carry
    # ``geom_margin`` 0.001 on BOX/MESH pairs, which its MULTICCD path rejects
    # (``mujoco_warp/_src/io.py:_check_margin``). A margin is a contact quantity and reaches no
    # pixel, so zeroing it for the render context is sound -- and it is restored, because this is
    # the env's own MjModel.
    margin0 = np.array(m.geom_margin, copy=True), np.array(m.pair_margin, copy=True)
    m.geom_margin[:] = 0.0
    m.pair_margin[:] = 0.0
    rec["zeroed_margin_for_put_model"] = bool(margin0[0].any() or margin0[1].any())
    try:
        # BatchRenderer prepares the model itself; preparing it here as well would double-apply the
        # group surgery and leave restore_model with a stale record.
        with MW.gl_rounding():
            br = MW.BatchRenderer(m, batch=batch, res=SIZE, camera=cam,
                                  groups=MW.DEFAULT_GROUPS)
            rec["texture_gaps"] = len(br.texture_gaps)
            rec["texture_gap_geoms"] = [n for _, n, _ in br.texture_gaps][:12]
            qq = torch.as_tensor(
                np.stack([qs[i % len(qs)] for i in range(batch)]), dtype=torch.float32,
                device="cuda")
            got = br.render(qq)[: len(qs)].cpu().numpy()
            wp.synchronize()
            t0 = time.perf_counter()
            reps = 20
            for _ in range(reps):
                br.render(qq)
            wp.synchronize()
            rec["mjwarp_fps"] = round(batch * reps / (time.perf_counter() - t0), 1)
        rec["mjwarp_vs_mujoco_fullscene"] = {
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in pixel_agreement(np.asarray(got), refs).items()}
    except Exception as e:                                                  # noqa: BLE001
        import traceback
        traceback.print_exc()
        rec["mjwarp_error"] = f"{type(e).__name__}: {e}"
    finally:
        m.geom_margin[:], m.pair_margin[:] = margin0
        if br is not None and br.prepared is not None:
            MW.restore_model(m, br.prepared)

    # reference fps, same camera, same resolution
    t0 = time.perf_counter()
    for i in range(20):
        d.qpos[:] = qs[i % len(qs)]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam)
        r.render()
    rec["mujoco_fps"] = round(20 / (time.perf_counter() - t0), 1)
    env.close()
    return rec


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "plate"
    tasks = sys.argv[2:] or SUBSET
    if tasks == ["ALL"]:
        from metaworld.env_dict import ALL_V3_ENVIRONMENTS
        tasks = sorted(ALL_V3_ENVIRONMENTS)
    fn = {"plate": plate_within_episode, "mjwarp": mjwarp_parity}[mode]
    out = []
    for t in tasks:
        try:
            rr = fn(t)
        except Exception as ex:                                             # noqa: BLE001
            import traceback
            traceback.print_exc()
            rr = {"task": t, "error": f"{type(ex).__name__}: {ex}"}
        out.append(rr)
        print(json.dumps(rr), flush=True)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"probe_backend2_{mode}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)
