"""Phase 1: the REGISTERED cell (``DMCQuadrupedWalk``, camera 2, 224 px, SWM's default appearance and
density) on CPU against the official renderer under the declared 3.11.0 reference pair, per residual.

Three modes, because the declared reference (mujoco 3.11.0 + dm_control 1.0.44) lives in a venv
without torch and the cell lives in one with mujoco 3.12.0:

  ours      (se6-venv)      sample 64 states from the cell's own ``sample_states`` (seed 100, the
                            first ``visual_calib.py`` draw), render them through ``se.make(...,
                            device="cpu")`` at the registered size (and 84 px supplementally), and
                            record a digest of the compiled model arrays the cell draws.
  official  (mj311 venv)    rebuild SWM's model with the same transcription (``_swm_quadruped_env``'s
                            lines, repeated here because stable_envs cannot import without torch),
                            ASSERT its digest equals the cell's, and render ``physics.render`` at
                            camera 2 with the segmentation buffer at each state (round trip asserted).
  compare   (se6-venv)      ``conformance.pixel_agreement`` / ``ssim`` plus the per-surface split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)


def model_digest(m) -> str:
    """The arrays the frame depends on; identical across mujoco 3.11/3.12 for this model (measured)."""
    h = hashlib.sha256()
    for name in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_bodyid", "geom_group",
                 "body_pos", "body_quat", "body_parentid", "jnt_type", "jnt_qposadr", "qpos0",
                 "cam_pos", "cam_quat", "cam_mode", "cam_fovy", "light_pos", "light_dir", "light_mode",
                 "light_diffuse", "light_castshadow", "site_group", "site_size", "site_pos"):
        h.update(np.ascontiguousarray(np.asarray(getattr(m, name))).tobytes())
    # the material/texture tables by NAME (their ids are permuted by the mjcf round trip)
    import mujoco
    for i in sorted(range(m.nmat), key=lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i)):
        h.update(str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i)).encode())
        h.update(np.asarray(m.mat_rgba[i]).round(6).tobytes())
        h.update(np.asarray(m.mat_reflectance[i]).round(6).tobytes())
    for t in sorted(range(m.ntex), key=lambda t: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, t)):
        a, n = int(m.tex_adr[t]), int(m.tex_width[t]) * int(m.tex_height[t]) * int(m.tex_nchannel[t])
        h.update(np.asarray(m.tex_data[a:a + n]).tobytes())
    h.update(np.asarray(m.body_mass).round(9).tobytes())
    return h.hexdigest()[:16]


def swm_quadruped_env_standalone(seed: int):
    """``stable_envs.envs.dmcontrol._swm_quadruped_env``, line for line, without stable_envs."""
    import tempfile

    from dm_control import mjcf
    from dm_control.rl import control
    from dm_control.suite import quadruped
    from dm_control.suite.wrappers import action_scale
    speed = 0.5
    xml = quadruped.make_model(floor_size=20 * speed).replace(b'file="./common/', b'file="common/')
    model = mjcf.from_xml_string(xml, model_dir=os.path.dirname(quadruped.__file__),
                                 assets=quadruped.common.ASSETS or {})
    grid = model.find("texture", "grid")
    grid.rgb1 = np.array([0.1, 0.2, 0.3])
    grid.rgb2 = np.array([0.2, 0.3, 0.4])
    model.find("material", "self").rgba = np.array([0.6, 0.3, 0.3, 1.0], np.float32)
    floor = model.find("geom", "floor")
    fr = (np.array([1.0, 0.005, 0.0001], np.float32) if floor.friction is None
          else np.asarray(floor.friction, np.float32).copy())
    fr[0] = 1.0
    floor.friction = fr
    model.find("geom", "torso").density = 1000.0
    model.find("geom", "foot_back_left").density = 1000.0
    model.find("light", "light").diffuse = 0.7 * np.ones(3, np.float32)
    with tempfile.TemporaryDirectory(prefix="se_swm_quadruped_ref_") as td:
        mjcf.export_with_assets(model, td, out_file_name="quadruped.xml")
        physics = quadruped.Physics.from_xml_path(os.path.join(td, "quadruped.xml"))
    task = quadruped.Move(desired_speed=speed, random=seed)
    env = control.Environment(physics, task, time_limit=20, control_timestep=0.02)
    return action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)


def mode_ours(a):
    import importlib.metadata as md

    import mujoco
    import torch

    import stable_envs as se
    torch.set_num_threads(a.threads)
    env = se.make("DMCQuadrupedWalk", batch=8, device="cpu")
    cls = type(env)
    states = env.sample_states(a.n, rng=np.random.default_rng(a.seed))
    m = cls._model()
    out = {"mujoco": mujoco.__version__, "dm_control": md.version("dm_control"),
           "torch": torch.__version__, "model_digest": model_digest(m), "camera": cls.CAMERA,
           "default_size": cls.default_size, "n": int(a.n), "seed": a.seed, "sizes": {}}
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "states.npz"), states=states)
    import time
    for size in (cls.default_size, 84):
        t0 = time.perf_counter()
        fr = env.render(torch.as_tensor(states), size=size).cpu().numpy()
        wall = time.perf_counter() - t0
        np.savez_compressed(os.path.join(a.out, f"ours_cpu_cam{cls.CAMERA}_{size}px.npz"), frames=fr)
        # render-only throughput, local, 1 thread: not a certification number
        env_b = se.make("DMCQuadrupedWalk", batch=64, device="cpu", size=size)
        st = torch.as_tensor(states[:64])
        env_b.render(st)
        fps = {}
        for B in (1, 16, 64):
            reps = 3 if B == 64 else 5
            t1 = time.perf_counter()
            for _ in range(reps):
                env_b.render(st[:B])
            fps[f"B{B}"] = round(B * reps / (time.perf_counter() - t1), 1)
        out["sizes"][str(size)] = {"wall_s_64_states": round(wall, 2), "fps_local_1thread": fps}
        print(size, out["sizes"][str(size)], flush=True)
    json.dump(out, open(os.path.join(a.out, "ours_provenance.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "sizes"}))


def mode_official(a):
    import importlib.metadata as md

    import mujoco
    env = swm_quadruped_env_standalone(0)
    env.reset()
    p = env.physics
    m = p.model.ptr
    want = json.load(open(os.path.join(a.out, "ours_provenance.json")))
    dg = model_digest(m)
    assert dg == want["model_digest"], f"model digest {dg} != the cell's {want['model_digest']}"
    states = np.load(os.path.join(a.out, "states.npz"))["states"].astype(np.float64)
    cam = int(want["camera"])
    tag = f"mj{mujoco.__version__.replace('.', '')}"
    for size in (int(want["default_size"]), 84):
        rs = mujoco.Renderer(m, height=size, width=size)
        rs.enable_segmentation_rendering()
        frames, segs = [], []
        for st in states:
            with p.reset_context():
                p.data.qpos[:] = st[:m.nq]
                p.data.qvel[:] = 0
            assert np.abs(np.asarray(p.data.qpos) - st[:m.nq]).max() <= 1e-9
            frames.append(p.render(height=size, width=size, camera_id=cam).copy())
            rs.update_scene(p.data.ptr, camera=cam)
            segs.append(rs.render().copy())
        np.savez_compressed(os.path.join(a.out, f"official_{tag}_cam{cam}_{size}px.npz"),
                            frames=np.stack(frames), segs=np.stack(segs))
        print("wrote", tag, size, flush=True)
    json.dump({"mujoco": mujoco.__version__, "dm_control": md.version("dm_control"),
               "model_digest": dg, "MUJOCO_GL": os.environ.get("MUJOCO_GL")},
              open(os.path.join(a.out, f"official_provenance_{tag}.json"), "w"), indent=1)


def per_surface(ours, official, segs, size):
    from scipy import ndimage
    d = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
    objid, objtype = segs[..., 0], segs[..., 1]
    geom = objtype == 5
    floor, body = geom & (objid == 0), geom & (objid > 0)
    ell = geom & (objid == 3)
    dil = 16 if size >= 224 else 6
    near = np.zeros_like(floor)
    for i in range(len(ours)):
        mk = ndimage.binary_dilation(body[i], iterations=dil)
        ys, xs = np.nonzero(body[i])
        if ys.size:
            h = ys.max() - ys.min()
            ext = np.zeros_like(mk)
            ext[ys.min():min(size, ys.max() + h + dil), max(0, xs.min() - dil):min(size, xs.max() + dil)] = True
            mk |= ext
        near[i] = mk
    masks = {"sky": objid < 0, "floor": floor, "floor_near_body": floor & near, "floor_far": floor & ~near,
             "body": body, "ellipsoid": ell, "site": objtype == 6}
    tot = max(int(d.sum()), 1)
    out = {}
    for k, msk in masks.items():
        n = int(msk.sum())
        out[k] = {"share": float(msk.mean()), "mean_abs": float(d[msk].mean()) if n else None,
                  "exact": float((d[msk] == 0).mean()) if n else None,
                  "within8": float((d[msk] <= 8).mean()) if n else None,
                  "resid_share": float(d[msk].sum() / tot)}
    return out


def mode_compare(a):
    from stable_envs.testing import conformance as C
    want = json.load(open(os.path.join(a.out, "ours_provenance.json")))
    cam = int(want["camera"])
    res = {"provenance_ours": want, "cells": {}}
    for size in (int(want["default_size"]), 84):
        ours = np.load(os.path.join(a.out, f"ours_cpu_cam{cam}_{size}px.npz"))["frames"]
        for tag in ("mj3110", "mj3120"):
            f = os.path.join(a.out, f"official_{tag}_cam{cam}_{size}px.npz")
            if not os.path.exists(f):
                continue
            ref = np.load(f)
            r = C.pixel_agreement(ours, ref["frames"])
            r["ssim"] = C.ssim(ours, ref["frames"])
            r["per_surface"] = per_surface(ours, ref["frames"], ref["segs"], size)
            res["cells"][f"{size}px_vs_{tag}"] = r
            ps = r["per_surface"]
            print(f"{size}px vs {tag}: mean {r['mean_abs']:.3f} exact {r['exact_frac']:.4f} within8 "
                  f"{r['within8_frac']:.4f} ssim {r['ssim']:.4f} interior_exact {r.get('interior_exact_frac')} | "
                  + " ".join(f"{k} {v['mean_abs']:.2f}/{v['resid_share']:.3f}" for k, v in ps.items()
                             if v["mean_abs"] is not None), flush=True)
        try:
            from PIL import Image
            ref = np.load(os.path.join(a.out, f"official_mj3110_cam{cam}_{size}px.npz"))["frames"]
            diff = np.clip(np.abs(ours[:6].astype(np.int16) - ref[:6].astype(np.int16)) * 8, 0, 255).astype(np.uint8)
            sheet = np.concatenate([np.concatenate(list(ref[:6]), 1), np.concatenate(list(ours[:6]), 1),
                                    np.concatenate(list(diff), 1)], 0)
            Image.fromarray(sheet).save(os.path.join(a.out, f"sheet_{size}px_official311_ours_diffx8.png"))
        except Exception as e:  # noqa: BLE001
            print("sheet failed:", e)
    json.dump(res, open(os.path.join(a.out, "cell_vs_official.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("ours", "official", "compare"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    {"ours": mode_ours, "official": mode_official, "compare": mode_compare}[a.mode](a)


if __name__ == "__main__":
    main()
