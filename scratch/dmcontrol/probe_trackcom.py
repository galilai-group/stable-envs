"""The six ``trackcom`` DM Control domains through a FULL-SCENE backend, one domain per process.

``docs/research/dmcontrol.md`` S8 rules these out because camera 0 rides the body, so the plate the
shipped renderer bakes is not a constant: hide every mover and the "static" frame still moves by
51-220/255. That is a limitation of the plate, and a backend that draws the whole scene every frame
does not have it -- ``docs/research/mujoco_locomotion.md`` S6.2 measures MJWarp's ``mjCAMLIGHT_TRACKCOM``
camera landing on MuJoCo's ``cam_xpos`` to **0.0**, provided ``com_pos`` runs between ``kinematics``
and ``camlight`` (``kinematics`` does not write ``subtree_com``, and omitted it freezes the camera at
its ``put_data`` pose with no error at all).

So the question this probe asks is not "does the camera work" but "what is the frame worth". It
measures, per domain:

  * the camera control first -- MJWarp's ``cam_xpos`` against MuJoCo's, and the spread over states,
    because a frozen camera produces a plausible-looking picture of the right scene from the wrong
    place and every other number here would be measured on it;
  * whole-frame agreement with ``physics.render()``, in the three quantities the VISUAL tier declares
    (``mean_abs``, ``within8_frac``, ``ssim``) plus the interior-exact number the tier reports for
    honesty;
  * where the disagreement is, off MuJoCo's own segmentation buffer, split into skybox / floor /
    bodies -- because DM Control is 62-82% SKYBOX where Gymnasium locomotion is 52-55%, and the
    checker floor that decides that family covers 14-35% here;
  * an ablation ladder that both renderers see identically, so "it is the floor texture" is a
    measurement rather than an inference.

One domain per process. A CUDA 700 poisons the interpreter, two MuJoCo envs alive together can return
byte-identical frames from cameras that provably differ, and EGL contexts are never released.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WARP_CACHE_PATH", os.path.expanduser("~/.cache/warp"))

import numpy as np

VISIBLE_GROUPS = (0, 1, 2)
DEFAULT_RGBA = np.array([0.5, 0.5, 0.5, 1.0])


def model_of(p):
    return p.model.ptr if hasattr(p.model, "ptr") else p.model


def interior_mask(frames):
    import scipy.ndimage as ndi
    out = []
    for f in frames:
        g = f.astype(np.int16)
        flat = (np.abs(np.diff(g, axis=0)).sum(-1)[:, :-1]
                + np.abs(np.diff(g, axis=1)).sum(-1)[:-1]) == 0
        out.append(ndi.minimum_filter(flat, size=3))
    return np.stack(out)


def ssim(a, b, win=7):
    """Mean SSIM on luminance, the same eight lines ``testing/conformance.ssim`` uses."""
    import scipy.ndimage as ndi
    f = lambda x: ndi.uniform_filter(x, size=win, mode="nearest")          # noqa: E731
    x = a.astype(np.float64) @ [0.299, 0.587, 0.114]
    y = b.astype(np.float64) @ [0.299, 0.587, 0.114]
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ux, uy = f(x), f(y)
    vx, vy = f(x * x) - ux * ux, f(y * y) - uy * uy
    vxy = f(x * y) - ux * uy
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))
    return float(s.mean())


def agreement(a, b, mask=None):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    if mask is not None:
        sel = d[mask]
        if sel.size == 0:
            return {"n": 0}
        return {"n": int(sel.size), "mean_abs": round(float(sel.mean()), 4),
                "exact_frac": round(float((sel == 0).mean()), 6),
                "within8_frac": round(float((sel <= 8).mean()), 6)}
    res = {"mean_abs": round(float(d.mean()), 4), "exact_frac": round(float((d == 0).mean()), 6),
           "within8_frac": round(float((d <= 8).mean()), 6), "max_abs": int(d.max()),
           "ssim": round(ssim(a, b), 6)}
    m = interior_mask(b)
    di = d[:, :-1, :-1][m]
    res["interior_frac"] = round(float(m.mean()), 6)
    res["interior_exact"] = round(float((di == 0).mean()), 6) if di.size else None
    res["interior_mean"] = round(float(di.mean()), 4) if di.size else None
    return res


def sample_qpos(domain, task, n, seed=0, steps=40):
    """The domain's own ``initialize_episode`` plus a random-action rollout -- ``sample_states``."""
    from dm_control import suite
    env = suite.load(domain, task, task_kwargs={"random": int(seed)})
    spec = env.action_spec()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(n)):
        env.reset()
        for _ in range(int(rng.integers(0, steps))):
            env.step(rng.uniform(spec.minimum, spec.maximum))
        out.append(np.asarray(env.physics.data.qpos, np.float64))
    return np.stack(out)


def main() -> None:
    import mujoco
    from dm_control import suite

    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--size", type=int, default=84)
    ap.add_argument("--states", type=int, default=32)
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/logs/dmc2")
    a = ap.parse_args()
    out = {"domain": a.domain, "task": a.task, "camera": a.camera, "size": a.size,
           "states": a.states}

    from stable_envs.render import mjwarp as MW

    env = suite.load(a.domain, a.task, task_kwargs={"random": 0})
    env.reset()                       # so the task's initialize_episode model edits are APPLIED
    p = env.physics
    m = model_of(p)
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), a.size)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), a.size)
    d = mujoco.MjData(m)

    # the one colour case MJWarp reads differently from MuJoCo: a geom carrying BOTH its own rgba and
    # a material. MuJoCo prefers the geom's; MJWarp reads the material's. Counted rather than assumed,
    # because on Gymnasium locomotion it was worth interior 0.016 -> 0.939.
    own = [int(g) for g in range(m.ngeom)
           if int(m.geom_matid[g]) >= 0 and not np.allclose(m.geom_rgba[g], DEFAULT_RGBA)
           and not np.allclose(m.geom_rgba[g], m.mat_rgba[int(m.geom_matid[g])])]
    out["geoms_with_own_rgba_and_material"] = own
    out["texture_gaps"] = [list(t) for t in MW.texture_gaps(m, VISIBLE_GROUPS)]
    out["needs_com_pos"] = bool(MW.needs_com_pos(m))
    out["cam_mode"] = int(m.cam_mode[a.camera])
    out["offsamples"] = int(m.vis.quality.offsamples)

    q = sample_qpos(a.domain, a.task, a.states)

    # ---- reference: mujoco.Renderer, measured == physics.render() in docs/research/dmcontrol.md S3
    r = mujoco.Renderer(m, height=a.size, width=a.size)
    opt = mujoco.MjvOption()

    def ref_frames(qs):
        fr, cams = [], []
        for row in qs:
            d.qpos[:] = row
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=a.camera, scene_option=opt)
            fr.append(r.render().copy())
            # .copy(): d.cam_xpos[i] is a VIEW that the next mj_forward overwrites
            cams.append(np.array(d.cam_xpos[a.camera], np.float64, copy=True))
        return np.stack(fr), np.stack(cams)

    ref, ref_cam = ref_frames(q)
    out["ref_camera_spread"] = round(float(np.abs(ref_cam - ref_cam[0]).max()), 6)

    # ---- segmentation, to attribute the residual ------------------------------------------------
    r.enable_segmentation_rendering()
    seg = []
    for row in q:
        d.qpos[:] = row
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=a.camera, scene_option=opt)
        seg.append(r.render().copy()[..., :2].astype(np.int32))
    r.disable_segmentation_rendering()
    seg = np.stack(seg)
    objid, objtype = seg[..., 0], seg[..., 1]
    is_geom = objtype == int(mujoco.mjtObj.mjOBJ_GEOM)
    floor_ids = [g for g in range(m.ngeom)
                 if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)]
    m_sky = objid < 0
    m_floor = is_geom & np.isin(objid, floor_ids)
    m_body = is_geom & ~np.isin(objid, floor_ids)
    out["surface_share"] = {"skybox": round(float(m_sky.mean()), 4),
                            "floor": round(float(m_floor.mean()), 4),
                            "bodies": round(float(m_body.mean()), 4)}

    # ---- MJWarp ---------------------------------------------------------------------------------
    def build(**kw):
        with MW.gl_rounding():
            br = MW.BatchRenderer(m, batch=int(a.states), res=a.size, camera=a.camera,
                                  groups=VISIBLE_GROUPS, capture=False, **kw)
            br.render(_t(q))                        # compile the kernel under the patch
        # BatchRenderer leaves the model PREPARED (zero-alpha geometry moved out of the drawn
        # groups) and this probe shares one MjModel with the reference renderer. MJWarp read the
        # geometry once at context creation, so restoring now costs it nothing and stops the
        # ablation ladder measuring the reference against a model the pre-pass had edited.
        if br.prepared is not None:
            MW.restore_model(m, br.prepared)
        return br

    def _t(x):
        import torch
        return torch.as_tensor(np.asarray(x, np.float32), device="cuda")

    try:
        br = build()
    except Exception as e:                    # put_model refuses some models on non-render grounds
        out["mjwarp_error"] = f"{type(e).__name__}: {e}"
        jdump(a, out)
        return

    ours = br.render(_t(q)).cpu().numpy()
    import warp as wp
    cam_w = wp.to_torch(br.d.cam_xpos).cpu().numpy()[:len(q), a.camera]
    out["camera_vs_mujoco_max"] = round(float(np.abs(cam_w - ref_cam).max()), 8)
    out["camera_spread_mjwarp"] = round(float(np.abs(cam_w - cam_w[0]).max()), 6)

    out["mjwarp"] = agreement(ours, ref)
    out["mjwarp_by_surface"] = {"skybox": agreement(ours, ref, m_sky),
                                "floor": agreement(ours, ref, m_floor),
                                "bodies": agreement(ours, ref, m_body)}

    # the far tenth of the floor: the band where minification decides, and the one number that
    # separates "a filter disagreement" from "a wrong picture"
    if m_floor.any():
        rows = np.zeros_like(m_floor)
        h = m_floor.shape[1]
        top = int(round(h * 0.10))
        rows[:, :top] = True                 # a trackcom camera looks along the ground: far == top
        far = m_floor & rows
        if far.any():
            out["floor_far_band"] = agreement(ours, ref, far)
            lum = lambda x: (x.astype(np.float64) @ [0.299, 0.587, 0.114])   # noqa: E731
            out["far_band_luminance_std"] = {
                "reference": round(float(lum(ref)[far].std()), 3),
                "mjwarp": round(float(lum(ours)[far].std()), 3)}

    # ---- ladder: both renderers see the same edit -----------------------------------------------
    if a.ladder:
        lad = {}
        keep_tex = np.array(m.mat_texid, copy=True)
        keep_ref = np.array(m.mat_reflectance, copy=True)
        for name, edit in (("L1_no_reflectance", lambda: m.mat_reflectance.__setitem__(
                            slice(None), 0.0)),
                           ("L3_no_textures", lambda: m.mat_texid.__setitem__(slice(None), -1))):
            m.mat_texid[:] = keep_tex
            m.mat_reflectance[:] = keep_ref
            edit()
            rr, _ = ref_frames(q)
            b2 = build(skybox=(name != "L3_no_textures"))
            lad[name] = agreement(b2.render(_t(q)).cpu().numpy(), rr)
        m.mat_texid[:] = keep_tex
        m.mat_reflectance[:] = keep_ref
        out["ladder"] = lad

    # ---- throughput -----------------------------------------------------------------------------
    if a.bench:
        import time

        import torch
        out["bench"] = {}
        for B in (1, 16, 64, 256, 1024):
            qb = _t(np.repeat(q, int(np.ceil(B / len(q))), 0)[:B])
            with MW.gl_rounding():
                bb = MW.BatchRenderer(m, batch=B, res=a.size, camera=a.camera,
                                      groups=VISIBLE_GROUPS, capture=True)
            for _ in range(3):
                bb.render(qb)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            reps = max(3, int(200 // max(1, B // 16)))
            for _ in range(reps):
                bb.render(qb)
            torch.cuda.synchronize()
            out["bench"][B] = round(B * reps / (time.perf_counter() - t0), 1)
            del bb
        t0 = time.perf_counter()
        ref_frames(q[:16])
        out["bench"]["dm_control"] = round(16 / (time.perf_counter() - t0), 1)

    jdump(a, out)


def jdump(a, out):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"track_{a.domain}-{a.task}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    print(f"@@wrote {path}")


if __name__ == "__main__":
    main()
