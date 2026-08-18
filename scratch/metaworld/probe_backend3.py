"""Meta-World through the Fetch design: MuJoCo bakes the plate, MJWarp draws only the movers.

``probe_backend2.py`` reproduced Franka Kitchen's Gate 1 here — MJWarp hard-faults with CUDA 700 on
``mjTEXTURE_CUBE``, and every Meta-World model carries 2-4 cube textures reached by a material. On
the kitchen that was fatal. Here it may not be, and the reason is what the plate does:

  in the composite, a STATIC pixel comes from MuJoCo's bake. MJWarp has to get static geometry's
  DEPTH right, so that a mover behind the table is occluded — but never its COLOUR, because that
  colour is thrown away. So the cube textures can be neutralised in the MJWarp model at no cost to
  the composite, as long as no MOVER wears one.

Configurations, each in its own process because a CUDA 700 poisons the context for everything after
it (``probe_backend2`` measured one real fault followed by seven "failed to allocate 4 bytes"):

  ``full``       every texture as authored          — does the fault reproduce?
  ``nocube``     cube textures neutralised          — does it then render, and how close full-scene?
  ``composite``  ``nocube`` + MuJoCo's baked plate  — the number a port would ship, plus fps.
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


def neutralise_cube_textures(m) -> dict:
    """Point every material away from a ``mjTEXTURE_CUBE`` slot, keeping the skybox.

    Returns what to restore. The material keeps its ``mat_rgba``, so the geom is drawn flat in the
    colour MuJoCo would modulate the texture by — which is the right fallback for a surface whose
    pixels the composite takes from the plate anyway.
    """
    ttype = np.asarray(m.tex_type, np.int64)
    tex = np.array(m.mat_texid, copy=True)
    hit = 0
    flat = m.mat_texid.reshape(m.nmat, -1) if m.mat_texid.ndim > 1 else m.mat_texid.reshape(-1, 1)
    for i in range(flat.shape[0]):
        for j in range(flat.shape[1]):
            t = int(flat[i, j])
            if t >= 0 and ttype[t] == int(mujoco.mjtTexture.mjTEXTURE_CUBE):
                flat[i, j] = -1
                hit += 1
    return {"mat_texid": tex, "n_neutralised": hit}


def movers_and_plate(env, u, m, d, cam, res):
    """The mover set (geometry MJWarp must draw) and MuJoCo's baked plate for this episode."""
    ref = np.array(m.body_pos, copy=True), np.array(m.body_quat, copy=True)
    keep = {a: np.array(getattr(m, a), copy=True)
            for a in ("body_pos", "body_quat", "site_pos", "geom_pos", "geom_size")}
    env.reset()
    moved = {b for b in range(m.nbody)
             if not np.array_equal(ref[0][b], m.body_pos[b])
             or not np.array_equal(ref[1][b], m.body_quat[b])}
    for a, v in keep.items():
        getattr(m, a)[:] = v
    mujoco.mj_forward(m, d)

    vis = [g for g in range(m.ngeom) if int(m.geom_group[g]) in MW.DEFAULT_GROUPS]
    movers = sorted({g for g in vis if int(m.body_weldid[m.geom_bodyid[g]]) != 0
                     or int(m.body_mocapid[m.geom_bodyid[g]]) >= 0}
                    | {g for g in vis if int(m.geom_bodyid[g]) in moved})

    r = mujoco.Renderer(m, height=res, width=res)
    o = mujoco.MjvOption()
    o.sitegroup[:] = 0
    g0 = np.array(m.geom_group, copy=True)
    m.geom_group[np.asarray(movers, np.int64)] = MW.HIDDEN_GROUP
    r.update_scene(d, camera=cam, scene_option=o)
    plate = r.render().copy()
    m.geom_group[:] = g0
    return movers, plate, r


def run(task: str, config: str, n: int = 8, batch: int = 16) -> dict:
    import torch
    import warp as wp

    import mujoco_warp as mjw

    env = make(task, SIZE)
    u = env.unwrapped
    m, d = u.model, u.data
    env.reset()
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "corner")
    rec = {"task": task, "config": config}

    movers, plate, r = movers_and_plate(env, u, m, d, cam, SIZE)
    rec["n_movers"] = len(movers)

    # does any MOVER wear a cube texture? that is the only place neutralising costs anything
    ttype = np.asarray(m.tex_type, np.int64)
    flat = m.mat_texid.reshape(m.nmat, -1) if m.mat_texid.ndim > 1 else m.mat_texid.reshape(-1, 1)
    cube_mats = {i for i in range(flat.shape[0])
                 for j in range(flat.shape[1])
                 if int(flat[i, j]) >= 0 and ttype[int(flat[i, j])] == 1}
    rec["movers_wearing_a_cube_texture"] = sorted(
        (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}")
        for g in movers if int(m.geom_matid[g]) in cube_mats)

    rng = np.random.default_rng(0)
    qs = []
    for _ in range(n):
        for _ in range(6):
            env.step(rng.uniform(-1, 1, size=u.action_space.shape).astype(np.float32))
        qs.append(np.array(d.qpos, copy=True))

    refs = []
    for q in qs:
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam)
        refs.append(r.render().copy())
    refs = np.stack(refs)

    t0 = time.perf_counter()
    for i in range(20):
        d.qpos[:] = qs[i % n]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam)
        r.render()
    rec["mujoco_fps"] = round(20 / (time.perf_counter() - t0), 1)

    # margin is a contact quantity that reaches no pixel; put_model rejects the model without this
    margin0 = np.array(m.geom_margin, copy=True), np.array(m.pair_margin, copy=True)
    m.geom_margin[:] = 0.0
    m.pair_margin[:] = 0.0
    undo = neutralise_cube_textures(m) if config in ("nocube", "composite") else None
    rec["n_cube_texture_slots_neutralised"] = undo["n_neutralised"] if undo else 0
    prep = MW.prepare_model(m, MW.DEFAULT_GROUPS)

    try:
        d0 = mujoco.MjData(m)
        mujoco.mj_forward(m, d0)
        wm = mjw.put_model(m)
        wd = mjw.put_data(m, d0, nworld=batch)
        with MW.gl_rounding():
            rc = MW.create_render_context(
                m, nworld=batch, cam_res=(SIZE, SIZE), render_rgb=True, render_seg=True,
                cam_active=[i == cam for i in range(m.ncam)],
                enabled_geom_groups=list(MW.DEFAULT_GROUPS), use_textures=True,
                use_shadows=True, render_skybox=True, enable_specular=True)

            def frame():
                mjw.kinematics(wm, wd)
                mjw.camlight(wm, wd)
                mjw.refit_bvh(wm, wd, rc)
                mjw.render(wm, wd, rc)

            frame()
            wp.synchronize()
        bgra = wp.to_torch(rc.rgb_data).view(torch.uint8).reshape(batch, SIZE, SIZE, 4)
        seg = wp.to_torch(rc.seg_data).reshape(batch, SIZE, SIZE, 2)[..., 0]
        qp = wp.to_torch(wd.qpos)
        qq = torch.as_tensor(np.stack([qs[i % n] for i in range(batch)]),
                             dtype=qp.dtype, device=qp.device)
        qp.copy_(qq)
        frame()
        wp.synchronize()

        rgb = bgra[..., [2, 1, 0]]
        got = rgb[:n].cpu().numpy()
        rec["mjwarp_vs_mujoco_fullscene"] = {
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in pixel_agreement(got, refs).items()}

        if config == "composite":
            mv_ids = torch.as_tensor(np.asarray(movers, np.int64), device=seg.device,
                                     dtype=seg.dtype)
            pl = torch.as_tensor(plate, device=rgb.device)
            mask = torch.isin(seg, mv_ids)
            comp = torch.where(mask.unsqueeze(-1), rgb, pl)[:n].cpu().numpy()
            rec["composite_vs_mujoco"] = {
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in pixel_agreement(comp, refs).items()}
            rec["mover_mask_frac"] = round(float(mask[:n].float().mean().item()), 5)

        # Un-captured, the four kernel launches per frame are dominated by launch overhead at these
        # scene sizes; the shipped design replays a captured graph, so that is what is timed.
        with wp.ScopedCapture() as capg:
            frame()
        graph = capg.graph
        for _ in range(3):
            wp.capture_launch(graph)
        wp.synchronize()
        t0 = time.perf_counter()
        reps = 50
        for _ in range(reps):
            wp.capture_launch(graph)
        wp.synchronize()
        rec["mjwarp_fps"] = round(batch * reps / (time.perf_counter() - t0), 1)
    except Exception as e:                                                  # noqa: BLE001
        import traceback
        traceback.print_exc()
        rec["mjwarp_error"] = f"{type(e).__name__}: {e}"
    finally:
        MW.restore_model(m, prep)
        if undo:
            m.mat_texid[:] = undo["mat_texid"]
        m.geom_margin[:], m.pair_margin[:] = margin0
    env.close()
    return rec


if __name__ == "__main__":
    task, config = sys.argv[1], sys.argv[2]
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    try:
        rr = run(task, config, batch=bs)
    except Exception as ex:                                                 # noqa: BLE001
        import traceback
        traceback.print_exc()
        rr = {"task": task, "config": config, "error": f"{type(ex).__name__}: {ex}"}
    print("RESULT " + json.dumps(rr), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"b3_{task}_{config}_{bs}.json"), "w") as f:
        json.dump(rr, f, indent=1)
