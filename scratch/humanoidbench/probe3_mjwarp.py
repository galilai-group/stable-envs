"""MJWarp on one HumanoidBench scene, with every correction this repo knows about, plus the ladder.

ONE TASK PER PROCESS. A CUDA 700 poisons the process, and two MuJoCo envs alive at once can return
byte-identical frames from cameras that provably differ.

The camera is TRACKCOM on every task but ``kitchen``, so there is no static plate (probe2 measures the
"static" frame moving by 54/255 over 21.7 % of pixels) and this is a full-scene comparison. The whole
question is therefore the floor: it is 70 % of the frame, it is a PLANE (so MJWarp can UV it), and its
texture is a DEGENERATE checker -- ``rgb1 == rgb2 == 0.9`` with a one-texel ``mark="edge"`` at 0.7 --
which is a very different object from the locomotion suite's full-contrast checker.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/humanoidbench")

from hb_common import (LOGS, MjwRenderer, agreement, cam_id, interior_mask, jsave, make_env,
                       mj_render, mj_segment, renderer, resolved_model, sample_qpos, set_state)

SIZE = 256
NSTATE = 16


def build_mjw(m, cam, res, nworld, **kw):
    return MjwRenderer(m, cam, res, nworld, **kw)


def flat_texture(m, texid):
    adr = int(m.tex_adr[texid])
    w, h, c = int(m.tex_width[texid]), int(m.tex_height[texid]), int(m.tex_nchannel[texid])
    n = w * h * c
    old = np.array(m.tex_data[adr:adr + n], copy=True)
    px = old.reshape(-1, c)
    m.tex_data[adr:adr + n] = np.tile(np.round(px.mean(0)).astype(np.uint8), (w * h, 1)).reshape(-1)
    return old


def attribute(got, ref, seg, m):
    import mujoco
    objid, objtype = seg[..., 0], seg[..., 1]
    isgeom = objtype == int(mujoco.mjtObj.mjOBJ_GEOM)
    plane = np.array([i for i in range(int(m.ngeom)) if int(m.geom_type[i]) == 0], np.int64)
    floor = isgeom & np.isin(objid, plane)
    classes = {"background": ~isgeom, "floor": floor, "bodies": isgeom & ~floor}
    im = interior_mask(ref)
    dif = np.abs(got.astype(np.int16) - ref.astype(np.int16)).max(-1)
    di = dif[:, :-1, :-1]
    tot = float(di[im].astype(np.float64).sum()) or 1.0
    out = {}
    for tag, msk in classes.items():
        mm = msk[:, :-1, :-1] & im
        out[tag] = dict(
            frame_frac=round(float(msk.mean()), 5),
            all_exact=round(float((dif[msk] == 0).mean()), 6) if msk.any() else None,
            interior_share=round(float(mm.sum() / max(im.sum(), 1)), 5),
            interior_exact=round(float((di[mm] == 0).mean()), 6) if mm.any() else None,
            interior_mean=round(float(di[mm].mean()), 4) if mm.any() else None,
            share_of_interior_error=round(float(di[mm].astype(np.float64).sum() / tot), 5))
    return out


def bands(got, ref, seg, m, frac=0.1):
    """Near/far floor bands: the measurement that decides a minification hypothesis."""
    import mujoco
    objid, objtype = seg[..., 0], seg[..., 1]
    isgeom = objtype == int(mujoco.mjtObj.mjOBJ_GEOM)
    plane = np.array([i for i in range(int(m.ngeom)) if int(m.geom_type[i]) == 0], np.int64)
    floor = isgeom & np.isin(objid, plane)
    rows = np.where(floor.any((0, 2)))[0]
    if rows.size == 0:
        return {}
    n = max(1, int(round(frac * rows.size)))
    lum = lambda a: a.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    out = {}
    for tag, rr in (("near", rows[-n:]), ("far", rows[:n])):
        msk = np.zeros_like(floor)
        msk[:, rr, :] = floor[:, rr, :]
        if not msk.any():
            continue
        d = np.abs(got.astype(np.int16) - ref.astype(np.int16)).max(-1)
        out[tag] = dict(frame_frac=round(float(msk.mean()), 5),
                        exact=round(float((d[msk] == 0).mean()), 6),
                        mean=round(float(d[msk].mean()), 4),
                        ref_lum_std=round(float(lum(ref)[msk].std()), 3),
                        got_lum_std=round(float(lum(got)[msk].std()), 3))
    return out


def main():
    task = sys.argv[1]
    robot = sys.argv[2] if len(sys.argv) > 2 else "h1hand"
    import mujoco
    res = {"task": task, "robot": robot, "size": SIZE, "nstate": NSTATE}

    env, u = make_env(task, robot, SIZE)
    cam_name = u.camera_name
    path = u.fullpath
    Q = sample_qpos(task, robot, NSTATE, seed=0, steps=60)

    # ---- the surgical model, and the proof that the surgery is inert ------------------------
    m, mat0 = resolved_model(path, SIZE)
    d = mujoco.MjData(m)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    ref_off = mj_render(u.model, u.data, cam_id(u), SIZE, Q, tag="official")
    ref = mj_render(m, d, cam, SIZE, Q, tag="surgical")
    dd = np.abs(ref.astype(np.int16) - ref_off.astype(np.int16))
    res["surgery_inert"] = dict(exact=round(float((dd.max(-1) == 0).mean()), 6), max_abs=int(dd.max()))
    res["n_colors"] = int(len(np.unique(ref.reshape(-1, 3), axis=0)))
    env.close()

    seg = mj_segment(m, d, cam, SIZE, Q, tag="seg")

    # ---- L0: the official scene --------------------------------------------------------------
    R = build_mjw(m, cam, SIZE, NSTATE)
    res["texture_gaps"] = [g[1] for g in R.texture_gaps]
    res["needs_com"] = bool(R.needs_com)
    res["prepared"] = R.prepared.summary() if R.prepared else None
    got = R.render(Q)
    res["L0"] = agreement(got, ref)
    res["L0_by_class"] = attribute(got, ref, seg, m)
    res["L0_bands"] = bands(got, ref, seg, m)

    # camera check: with and without com_pos
    R.render(Q, com=False)
    cam_frozen = R.cam_xpos()[:NSTATE, cam].copy()
    R.render(Q, com=True)
    cam_moving = R.cam_xpos()[:NSTATE, cam].copy()
    mj_cam = []
    for q in Q:
        set_state(m, d, q)
        mj_cam.append(np.array(d.cam_xpos[cam], np.float64))
    mj_cam = np.stack(mj_cam)
    res["camera"] = dict(
        name=cam_name, mode=int(m.cam_mode[cam]),
        ref_spread=round(float(np.abs(mj_cam - mj_cam[0]).max()), 5),
        frozen_spread=round(float(np.abs(cam_frozen - cam_frozen[0]).max()), 6),
        frozen_err=round(float(np.abs(cam_frozen - mj_cam).max()), 5),
        com_spread=round(float(np.abs(cam_moving - cam_moving[0]).max()), 5),
        com_err=round(float(np.abs(cam_moving - mj_cam).max()), 8))

    # ---- throughput --------------------------------------------------------------------------
    speed = {}
    for B in (1, 16, 64, 256):
        try:
            Rb = build_mjw(m, cam, SIZE, B, capture=True)
            qq = Q[np.arange(B) % NSTATE]
            Rb.upload(qq)
            Rb.wp.capture_launch(Rb.graph)
            Rb.wp.synchronize()
            t0 = time.perf_counter()
            reps = max(3, int(600 / B) + 1)
            for _ in range(reps):
                Rb.wp.capture_launch(Rb.graph)
            Rb.wp.synchronize()
            speed[f"mjw_b{B}"] = round(B * reps / (time.perf_counter() - t0), 1)
            if B == 16:
                t0 = time.perf_counter()
                for _ in range(20):
                    Rb._frame()
                Rb.wp.synchronize()
                speed["mjw_b16_uncaptured"] = round(16 * 20 / (time.perf_counter() - t0), 1)
            del Rb
        except Exception as e:                                              # pragma: no cover
            speed[f"mjw_b{B}"] = f"{type(e).__name__}: {e}"
    r = renderer(m, SIZE, "surgical")
    set_state(m, d, Q[0])
    r.update_scene(d, camera=int(cam))
    r.render()
    t0 = time.perf_counter()
    for i in range(60):
        set_state(m, d, Q[i % NSTATE])
        r.update_scene(d, camera=int(cam))
        r.render()
    speed["mujoco_egl"] = round(60 / (time.perf_counter() - t0), 1)
    res["throughput"] = speed

    # ---- the ladder --------------------------------------------------------------------------
    ladder = {}

    # L1: shadows off on BOTH sides
    r = renderer(m, SIZE, "surgical")
    out = []
    try:
        for q in Q:
            set_state(m, d, q)
            r.update_scene(d, camera=int(cam))
            r.scene.flags[int(mujoco.mjtRndFlag.mjRND_SHADOW)] = 0
            out.append(np.ascontiguousarray(r.render()))
    finally:
        r.update_scene(d, camera=int(cam))
        r.scene.flags[int(mujoco.mjtRndFlag.mjRND_SHADOW)] = 1
    ref_ns = np.stack(out)
    R1 = build_mjw(m, cam, SIZE, NSTATE, shadows=False)
    g1 = R1.render(Q)
    ladder["L1_noshadow"] = agreement(g1, ref_ns)
    ladder["L1_noshadow_by_class"] = attribute(g1, ref_ns, seg, m)
    del R1

    # L2: the floor texture flattened to its own mean on both sides
    texids = sorted({int(np.max(m.mat_texid[i])) if np.ndim(m.mat_texid[i]) else int(m.mat_texid[i])
                     for i in range(m.nmat)} - {-1})
    tex2d = [t for t in texids if int(m.tex_type[t]) == int(mujoco.mjtTexture.mjTEXTURE_2D)]
    saved = {t: flat_texture(m, t) for t in tex2d}
    _RC = renderer.__globals__["_RCACHE"]
    for k in [k for k in _RC if k[0] == id(m)]:
        _RC.pop(k)                                        # the MjrContext caches texture data
    ref_flat = mj_render(m, d, cam, SIZE, Q, tag="flat")
    R2 = build_mjw(m, cam, SIZE, NSTATE)
    g2 = R2.render(Q)
    ladder["L2_flat_texture"] = agreement(g2, ref_flat)
    ladder["L2_flat_texture_by_class"] = attribute(g2, ref_flat, seg, m)
    del R2
    for t, old in saved.items():
        adr = int(m.tex_adr[t])
        m.tex_data[adr:adr + old.size] = old
    for k in [k for k in _RC if k[0] == id(m)]:
        _RC.pop(k)

    # L3: no textures at all on either side (the material path only)
    keep_tex = np.array(m.mat_texid, copy=True)
    m.mat_texid[:] = -1
    ref_nt = mj_render(m, d, cam, SIZE, Q, tag="notex")
    R3 = build_mjw(m, cam, SIZE, NSTATE)
    g3 = R3.render(Q)
    ladder["L3_no_texture"] = agreement(g3, ref_nt)
    ladder["L3_no_texture_by_class"] = attribute(g3, ref_nt, seg, m)
    del R3

    # L4: L3 + specular 0
    keep_spec = np.array(m.mat_specular, copy=True)
    m.mat_specular[:] = 0.0
    ref_ns2 = mj_render(m, d, cam, SIZE, Q, tag="nospec")
    R4 = build_mjw(m, cam, SIZE, NSTATE, specular=False)
    g4 = R4.render(Q)
    ladder["L4_no_texture_no_specular"] = agreement(g4, ref_ns2)
    del R4
    m.mat_specular[:] = keep_spec
    m.mat_texid[:] = keep_tex
    res["ladder"] = ladder

    jsave(f"{LOGS}/probe3_{robot}_{task}.json", res)
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
