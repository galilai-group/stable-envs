"""Add depth and geom-id planes to a split that already has its cheap and official halves.

Same economy as ``gen_lr.py``: the official frames cost 30 f/s through ``mujoco.Renderer`` and the
cheap render costs 19,700, so a split is re-rendered rather than regenerated. The states and the
targets are read back from disk and reused byte for byte, which is also what makes the aux run and
the RGB-only run comparable -- they are scored on the same frames, not on two draws that happen to
come from the same generator.

The render has to be chunked at exactly the context's world count. ``RoboCasaBase.render`` sizes its
MJWarp context to the chunk, writes ``nworld`` frames into one buffer and moves on, so the depth and
segmentation buffers hold **only the last chunk** once a bigger batch has gone through. Rendering
exactly ``MJW_CHUNK`` states at a time and reading the buffers after each call is what makes the aux
planes line up with the frames instead of being the last 256 repeated.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

import aux as AX
import data as D
from probe_budget_frontier import ctx_cache, env_module, family_of


def context_for(mod, family, cls, res, nworld):
    """The cached ``_Ctx`` the env just rendered through. Both families build SINGLE-camera
    contexts, so the active-camera index is 0 (see ``aux.read_aux``)."""
    if family == "RoboCasa":
        return mod._context(cls.KITCHEN, cls.CAMERA, res, nworld,
                            int(getattr(cls, "TEX_LOD", 1)), getattr(cls, "BACKFACE_CULL", None))
    return mod._context(cls.ROBOT, cls.TASK, res, mod.MJW_CHUNK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--split", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--depth-scale", type=float, default=8.0)
    a = ap.parse_args()

    import stable_envs as se
    cls = se.REGISTRY[a.env]
    mod = env_module(a.env)
    family = family_of(a.env)
    d = D.split_dir(a.env, a.split, a.tag)
    shards = sorted(glob.glob(os.path.join(d, "states_*.npy")),
                    key=lambda p: int(p.rsplit("_", 1)[1][:-4]))
    if not shards:
        raise SystemExit(f"no states under {d}")
    meta0 = json.load(open(os.path.join(d, "meta_0.json")))
    res = int(meta0["render_size"])
    lod = int(meta0.get("tex_lod", 1))
    chunk = int(mod.MJW_CHUNK)

    orig = AX.patch_context(mod, lod)
    try:
        env = se.make(a.env, batch=1, size=256, device="cuda", seed=0)
        for sp in shards:
            k = sp.rsplit("_", 1)[1][:-4]
            states = np.load(sp)
            sty = None
            stp = os.path.join(d, f"style_{k}.npy")
            if os.path.exists(stp):
                sty = np.load(stp)
            depth = np.empty((len(states), res, res), np.uint8)
            seg = np.empty((len(states), res, res), np.int32)
            t0 = time.perf_counter()
            for i in range(0, len(states), chunk):
                j = min(i + chunk, len(states))
                env.render(states[i:j], size=res, style=None if sty is None else sty[i:j])
                n = j - i
                nw = 1 if n <= 1 else min(chunk, 1 << (n - 1).bit_length())
                c = context_for(mod, family, cls, res, nw)
                dd, ss = AX.read_aux(c, n, res, 0, a.depth_scale)
                depth[i:j] = dd.cpu().numpy()
                seg[i:j] = ss.cpu().numpy()
            np.save(os.path.join(d, f"depth_{k}.npy"), depth)
            np.save(os.path.join(d, f"segid_{k}.npy"), seg.astype(np.int16))
            print(f"shard {k}: {len(states)} aux planes in {time.perf_counter() - t0:.1f}s  "
                  f"depth mean {depth.mean():.1f}  distinct geoms {len(np.unique(seg))}  "
                  f"background frac {float((seg == 0).mean()):.3f}", flush=True)
        del env
    finally:
        mod.MW.create_render_context = orig
        ctx_cache(mod).clear()
        torch.cuda.empty_cache()

    # The per-geom descriptor table belongs to the SCENE, not to the split, so it is written once
    # beside the data and loaded by whatever consumes it. A checkpoint trained on one table and
    # evaluated against another built by a different revision of ``geom_features`` would be a silent
    # mismatch, so the column names travel with it.
    scene_model = (mod._scene(cls.KITCHEN).model if family == "RoboCasa"
                   else mod._build_bundle(cls.ROBOT, cls.TASK).model)
    feats = AX.geom_features(scene_model)
    np.save(os.path.join(D.ROOT, a.env, "geom_features.npy"), feats)
    with open(os.path.join(D.ROOT, a.env, "geom_features.json"), "w") as f:
        json.dump({"columns": list(AX.FEATURE_NAMES), "ngeom": int(scene_model.ngeom),
                   "rows": int(feats.shape[0]), "depth_scale_m": a.depth_scale,
                   "texgen_gap_geoms": int((feats[:, 4] > 0).sum())}, f, indent=1)
    print(f"wrote geom_features {feats.shape}, {int((feats[:, 4] > 0).sum())} geoms flagged as "
          f"textured primitives MJWarp cannot UV", flush=True)


if __name__ == "__main__":
    main()
