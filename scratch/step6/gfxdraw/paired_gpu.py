"""R as a PAIRED measurement, for the same reason S_cpu needed one.

``bench_step3`` times the official CPU renderer once, after the whole batch sweep. In the GFXDRAW-2
sweep that denominator moved by a factor of two between the two arms of ONE job on ONE node --
Acrobot's official renderer measured 428.2 frames/s in the arm with the kernel off and 229.8 in the
arm with it on, CartPole 488.4 against 246.1 -- while MountainCar's barely moved (448.3 against
439.1). So R came out anywhere between 325 and 756 depending on which reading was used, and the
class-R ``GPU_FAST`` bar of 500 sits inside that range. That is not a measurement.

The mechanism is contention: the reference is a single-threaded pygame loop on the same node whose
CPU is feeding the GPU, and it is timed in a block rather than interleaved. So this interleaves it:
a burst of official frames, then a burst of ours, round after round in one process, reporting the
median of the per-ROUND ratio. Drift slower than a round cancels; faster drift shows as spread.

Both arms in one process too, by flipping ``use_fused_render`` -- the "before" is this tree with the
kernel off, which is the code 6213673 shipped.

    python scratch/step6/gfxdraw/paired_gpu.py --batch 4096 --out <path>
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
CLASSIC = ("Acrobot", "CartPole", "MountainCar", "MountainCarContinuous", "Pendulum")


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default=",".join(CLASSIC))
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--ref-frames", type=int, default=24)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import stable_envs as se
    from stable_envs.envs.classic_control import _Classic
    from stable_envs.testing import conformance as C

    doc = {"host": socket.gethostname(), "cpu": platform.processor() or platform.machine(),
           "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "torch": torch.__version__, "batch": a.batch, "rounds": a.rounds, "inner": a.inner,
           "stable_envs_file": se.__file__, "envs": {}}
    print(f"{doc['host']}  {doc['device_name']}  B={a.batch}\n  tree {se.__file__}")

    for name in a.envs.split(","):
        if name not in se.REGISTRY:
            continue
        pool = np.load(ROOT / f"examples/results/step4/states/{name}.npz")["states"]
        ref_states = np.asarray(pool[np.arange(a.ref_frames) % len(pool)], np.float32)
        rec: dict = {}
        for arm, fused in (("before", False), ("after", True)):
            _Classic.use_fused_render = fused
            env = se.make(name, batch=a.batch, device="cuda")
            cls = type(env)
            st = np.asarray(pool[np.arange(a.batch) % len(pool)], np.float32)
            out = torch.empty((a.batch, *env.native, 3), dtype=torch.uint8, device="cuda")
            ref = C.reference_env(cls, env.size)
            cls.reference_render(ref, ref_states[:4])
            env.render(st, out=out)
            _sync()
            ours, theirs, ratios = [], [], []
            for _ in range(a.rounds):
                t0 = time.perf_counter()
                cls.reference_render(ref, ref_states)
                r = a.ref_frames / (time.perf_counter() - t0)
                inner = []
                for _ in range(a.inner):
                    _sync()
                    t0 = time.perf_counter()
                    env.render(st, out=out)
                    _sync()
                    inner.append(time.perf_counter() - t0)
                s = a.batch / statistics.median(inner)
                theirs.append(r)
                ours.append(s)
                ratios.append(s / r)
            rec[arm] = {"official_fps_median": statistics.median(theirs),
                        "official_fps_min": min(theirs), "official_fps_max": max(theirs),
                        "se_fps_median": statistics.median(ours),
                        "se_fps_min": min(ours), "se_fps_max": max(ours),
                        "R_paired_median": statistics.median(ratios),
                        "R_paired_min": min(ratios), "R_paired_max": max(ratios),
                        "digest": C.frames_digest(env.render(st[:64]).cpu().numpy()),
                        "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
            print(f"[{name:22s} {arm:6s}] official {rec[arm]['official_fps_median']:7.1f} "
                  f"({rec[arm]['official_fps_min']:.0f}-{rec[arm]['official_fps_max']:.0f})  "
                  f"ours {rec[arm]['se_fps_median']:10,.0f}  "
                  f"R {rec[arm]['R_paired_median']:7.1f} "
                  f"({rec[arm]['R_paired_min']:.0f}-{rec[arm]['R_paired_max']:.0f})", flush=True)
            del env, out
            torch.cuda.empty_cache()
        rec["speedup"] = rec["after"]["se_fps_median"] / rec["before"]["se_fps_median"]
        rec["frames_identical"] = rec["after"]["digest"] == rec["before"]["digest"]
        doc["envs"][name] = rec
        print(f"[{name:22s}] {rec['speedup']:.2f}x   frames identical: {rec['frames_identical']}")
    _Classic.use_fused_render = True
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(f"wrote {a.out}")
    bad = [k for k, v in doc["envs"].items() if not v["frames_identical"]]
    if bad:
        print(f"FAILED: the kernel changed pixels on {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
