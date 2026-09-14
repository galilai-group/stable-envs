"""The mandatory GPU sweep with the fused classic-control kernel ON and OFF, in one process.

The "before" arm for GFXDRAW-2 is not another tree: it is this tree with ``use_fused_render``
False, which is exactly the code Step 6 shipped at 6213673. Flipping a class default rather than
switching ``PYTHONPATH`` makes the two arms same-card and same-process by construction, and removes
the one failure mode that costs a GPU job — measuring the same tree twice.

Everything timed here goes through ``examples/bench_step3.measure_env``, so the batch list, the
repetition count, the per-cell budget, the official-renderer baseline and the decode bar are the
frozen Step 3 procedure and not a second one.

    python scratch/step6/gfxdraw/sweep_fused.py --device cuda --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples"))

CLASSIC = ("Acrobot", "CartPole", "MountainCar", "MountainCarContinuous", "Pendulum")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--envs", default=",".join(CLASSIC))
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--max-cell-s", type=float, default=30.0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", required=True, help="directory; one shard per arm")
    a = ap.parse_args()

    import torch

    import bench_step3 as B3
    from bench_gpu_vs_cpu import set_device
    import stable_envs as se
    from stable_envs.envs.classic_control import _Classic
    from stable_envs.render import gfxdraw_classic as FC

    if a.threads:
        torch.set_num_threads(a.threads)
    set_device(a.device)
    print(f"stable_envs from {se.__file__}")
    print(f"triton available: {FC.HAVE_TRITON}; device {a.device}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path("/tmp") / "gfx-sweep"
    scratch.mkdir(parents=True, exist_ok=True)
    universe = B3.matched_universe()
    batches = B3.batches_for(a.device, optional=False)
    print(f"batches {batches}")

    for arm, fused in (("before", False), ("after", True)):
        _Classic.use_fused_render = fused
        doc = {"schema_version": 1, "step": 6, "shard": f"gfxdraw-{arm}",
               "what_this_is": f"the frozen Step 3 sweep with the fused classic kernel "
                               f"{'ON' if fused else 'OFF'}; both arms one process, one card",
               "arm": arm, "fused": fused, "stable_envs_file": se.__file__,
               "provenance": B3.provenance(a.device, torch.get_num_threads()),
               "batches": batches, "envs": []}
        for name in a.envs.split(","):
            rec = B3.measure_env(name, universe.get(name, [name]), a.device, batches, a.reps,
                                 8.0, a.max_cell_s, 60, 40, "native", scratch)
            doc["envs"].append(rec)
            cells = {c["batch"]: c.get("fps") for c in (rec.get("cells") or []) if c.get("fps")}
            best = max(cells.values()) if cells else 0
            ref = ((rec.get("official_cpu") or {}).get("set_state_render") or {}).get("fps")
            print(f"[{arm:6s}] {name:22s} best {best:10,.0f} f/s   ref {ref or 0:7.1f}   "
                  f"R {best / ref if ref else 0:8.1f}", flush=True)
        p = out / f"gpu_{arm}_{doc['provenance']['host'].split('.')[0]}.json"
        p.write_text(json.dumps(doc, indent=1))
        print(f"wrote {p}")
    _Classic.use_fused_render = True


if __name__ == "__main__":
    main()
