"""M1a/M1b: the CPU backend on the frozen Step-4 states against OUR MJWarp frames and the official ones.

Reads the texture-fidelity workstream's frame stacks (``localize312/frames/<cfg>.npz``: ``frozen_ours``,
``frozen_official``, ``frozen_seg`` at the frozen corpus states, in corpus order), renders the same states
through ``se.make(cfg, device="cpu")``, and reports per-surface agreement (sky / floor / body by MuJoCo's
segmentation buffer). Writes JSON + a contact sheet per config.

    python scratch/step6/cpu-3d/compare_frozen.py --frames-dir <dir> --configs DMCCheetahRun --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def stats(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)[mask]
    if d.size == 0:
        return {"px": 0}
    return {"px": int(d.size), "mean_abs": float(d.mean()), "exact": float((d == 0).mean()),
            "within1": float((d <= 1).mean()), "within2": float((d <= 2).mean()),
            "within8": float((d <= 8).mean()), "max": int(d.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--configs", default="DMCCheetahRun,DMCHopperHop,DMCWalkerWalk,DMCHumanoidStand")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--frac-bits", type=int, default=None)
    ap.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(1)
    import stable_envs as se
    from stable_envs.render import cpu3d
    from stable_envs.testing.conformance import pixel_agreement, ssim
    if a.frac_bits is not None:
        cpu3d.TEX_FRAC_BITS = a.frac_bits
    res = {}
    for cfg in a.configs.split(","):
        z = np.load(a.frames_dir / f"{cfg}.npz")
        ours_gpu, off, seg = z["frozen_ours"][:a.n], z["frozen_official"][:a.n], z["frozen_seg"][:a.n]
        st = np.load(ROOT / f"examples/results/step4/states/{cfg}.npz")["states"][:a.n]
        env = se.make(cfg, batch=len(st), device="cpu")
        t0 = time.perf_counter()
        cpu = env.render(st).cpu().numpy()
        dt = time.perf_counter() - t0
        gid = seg[..., 0].astype(np.int64)
        sky = gid < 0
        floor = gid == env._cpu_scene(env.size).plane_geom
        body = ~sky & ~floor
        r = {"n": int(len(st)), "render_s": dt, "fps_1thread_local": len(st) / dt,
             "vs_mjwarp": {"all": stats(cpu, ours_gpu, np.ones_like(sky)), "sky": stats(cpu, ours_gpu, sky),
                           "floor": stats(cpu, ours_gpu, floor), "body": stats(cpu, ours_gpu, body)},
             "vs_official": {"all": stats(cpu, off, np.ones_like(sky)), "sky": stats(cpu, off, sky),
                             "floor": stats(cpu, off, floor), "body": stats(cpu, off, body)},
             "mjwarp_vs_official": {"all": stats(ours_gpu, off, np.ones_like(sky))},
             "declared_metric_cpu_vs_official": {**pixel_agreement(cpu, off), "ssim": ssim(cpu, off)},
             "declared_metric_mjwarp_vs_official": {**pixel_agreement(ours_gpu, off), "ssim": ssim(ours_gpu, off)},
             "frac_bits": cpu3d.TEX_FRAC_BITS}
        # which pixels disagree with MJWarp by more than a level, by surface (is it structure or a level?)
        d = np.abs(cpu.astype(np.int16) - ours_gpu.astype(np.int16)).max(-1)
        r["vs_mjwarp"]["gt2_share_by_surface"] = {"sky": float((d[sky] > 2).mean()) if sky.any() else None,
                                                  "floor": float((d[floor] > 2).mean()) if floor.any() else None,
                                                  "body": float((d[body] > 2).mean()) if body.any() else None}
        res[cfg] = r
        v = r["vs_mjwarp"]
        print(f"{cfg:18s} {len(st)} states {dt:.2f}s ({len(st)/dt:.0f} f/s local)  vs MJWarp: all mean {v['all']['mean_abs']:.3f} exact {v['all']['exact']:.3f} within1 {v['all']['within1']:.3f} | "
              f"sky {v['sky'].get('mean_abs', float('nan')):.3f}/{v['sky'].get('within1', float('nan')):.3f} floor {v['floor'].get('mean_abs', float('nan')):.3f}/{v['floor'].get('within1', float('nan')):.3f} body {v['body'].get('mean_abs', float('nan')):.3f}/{v['body'].get('within1', float('nan')):.3f} max {v['all']['max']}  || "
              f"cpu vs official mean {r['declared_metric_cpu_vs_official']['mean_abs']:.3f} w8 {r['declared_metric_cpu_vs_official']['within8_frac']:.3f} ssim {r['declared_metric_cpu_vs_official']['ssim']:.3f} "
              f"(mjwarp {r['declared_metric_mjwarp_vs_official']['mean_abs']:.3f}/{r['declared_metric_mjwarp_vs_official']['within8_frac']:.3f}/{r['declared_metric_mjwarp_vs_official']['ssim']:.3f})", flush=True)
        if a.sheet:
            from PIL import Image
            rows = []
            for i in (0, 1, 2, 3):
                dd = np.clip(np.abs(cpu[i].astype(np.int16) - ours_gpu[i].astype(np.int16)) * 8, 0, 255).astype(np.uint8)
                rows.append(np.concatenate([cpu[i], ours_gpu[i], dd, off[i]], 1))
            Image.fromarray(np.concatenate(rows, 0)).resize((4 * 84 * 3, 4 * 84 * 3), Image.NEAREST).save(a.out / f"{cfg}_sheet.png")
    (a.out / "compare_frozen.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
