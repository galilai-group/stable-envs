"""Cross-backend for DMCQuadrupedWalk: the CPU backend vs OUR MJWarp frames on the same states.

``--frames`` is the ``frames/<config>.npz`` stack ``scratch/step6/texture-fidelity/localize.py``
writes on the cuda job (``frozen_ours`` = MJWarp frames, ``frozen_official``, ``frozen_seg`` at the
corpus states in corpus order); the states are the Step-7 corpus (``--states``). Per-surface split as
``measure_cell.per_surface``. Same claim as the cpu-3d workstream's: the same picture to within a
level, not bit-identity (cpu-3d findings: mean 0.004-0.018/255, >= 0.9997 within one level).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="DMCQuadrupedWalk")
    ap.add_argument("--frames", required=True, help="localize.py frames/<config>.npz from the cuda job")
    ap.add_argument("--states", required=True, help="the Step-7 corpus <config>.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    import torch

    import stable_envs as se
    from measure_cell import per_surface
    from stable_envs.testing import conformance as C
    torch.set_num_threads(a.threads)
    z = np.load(a.frames)
    cuda, official, seg = z["frozen_ours"], z["frozen_official"], z["frozen_seg"]
    states = np.load(a.states)["states"][:len(cuda)]
    env = se.make(a.config, batch=len(states), device="cpu")
    cpu = env.render(torch.as_tensor(states)).cpu().numpy()
    size = cpu.shape[1]
    d = np.abs(cpu.astype(np.int16) - cuda.astype(np.int16)).max(-1)
    res = {"config": a.config, "n": int(len(states)), "size": size,
           "cpu_vs_cuda": {"mean_abs": float(d.mean()), "exact": float((d == 0).mean()),
                           "within1": float((d <= 1).mean()), "within2": float((d <= 2).mean()),
                           "max": int(d.max()), "gt2_px": int((d > 2).sum()),
                           "per_surface": per_surface(cpu, cuda, seg, size)},
           "cpu_vs_official": dict(C.pixel_agreement(cpu, official), ssim=C.ssim(cpu, official),
                                   per_surface=per_surface(cpu, official, seg, size)),
           "cuda_vs_official": dict(C.pixel_agreement(cuda, official), ssim=C.ssim(cuda, official),
                                    per_surface=per_surface(cuda, official, seg, size))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    x = res["cpu_vs_cuda"]
    print(f"cpu vs cuda: mean {x['mean_abs']:.4f} exact {x['exact']:.5f} within1 {x['within1']:.6f} "
          f"max {x['max']} gt2 {x['gt2_px']} px")
    for k in ("cpu_vs_official", "cuda_vs_official"):
        y = res[k]
        print(f"{k}: mean {y['mean_abs']:.3f} within8 {y['within8_frac']:.4f} ssim {y['ssim']:.4f}")


if __name__ == "__main__":
    main()
