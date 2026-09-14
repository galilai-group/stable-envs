"""cpu-3d: gfxdraw's paired Step-3 shard (scratch/step6/gfxdraw/certify_shard.py) relabelled for this workstream, with the tree commit from STABLE_ENVS_GIT_COMMIT when the tree has no .git, reps/thread env recorded, and the 1024 cell kept (max-cell-s 60).

A Step-3-shaped certification shard whose DENOMINATOR is paired, on one host, in one process.

`examples/bench_step3.py` produces the shard schema `scratch/step3/certify.py` labels tiers from,
and everything about it is right except one thing on a shared node: it times the official renderer
**once, in a block, after the whole batch sweep**. On `node2338` at load average 47 that single
reading moved 2.3x between the two arms of one job, and `S_cpu` moved with it — far enough to
relabel four configurations a tier up. The same happened on `gpu2110` for R, far enough to publish
`GPU_FAST` for two rows that a paired measurement puts at 347-390 against a bar of 500.

So this emits the same schema with the reference measured the way `paired_cpu.py`/`paired_gpu.py`
measure it: a reference burst **between every batch cell**, spread across the whole run rather than
blocked at the end, reported as the median of those readings with their spread kept beside it. The
cells, the decode bar and the provenance are the frozen Step-3 machinery unchanged — this is not a
second benchmark, it is the same benchmark with its denominator sampled properly.

Everything a later reader needs to accept or reject the row travels with it: host, load average,
torch thread count, the tree actually imported, and the git commit of that tree.

    python scratch/step6/gfxdraw/certify_shard.py --device cpu  --threads 1 --out <path>
    python scratch/step6/gfxdraw/certify_shard.py --device cuda --arm before --out <path>
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples"))

CLASSIC = ("DMCCheetahRun", "DMCHopperHop", "DMCWalkerWalk", "DMCHumanoidStand")


def git_commit(tree: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30).stdout.strip() or "unknown"
    except Exception:                                                     # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--envs", default=",".join(CLASSIC))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--max-cell-s", type=float, default=60.0)
    ap.add_argument("--max-out-gb", type=float, default=8.0)
    ap.add_argument("--ref-frames", type=int, default=24)
    ap.add_argument("--decode-frames", type=int, default=40)
    ap.add_argument("--arm", default="after", choices=["before", "after"],
                    help="'before' forces the per-primitive path (classic control only)")
    #: The reference is now sampled across the whole run, so its SPREAD is evidence about the node
    #: and not just noise to average away. node2339 returned 4.52x on Acrobot and 10.37x on
    #: CartPole while node1829 and gpu2264 returned 1.02-1.11x; a ratio built on the first is not a
    #: measurement, and the only reason that is visible at all is that the spread is recorded. A
    #: shard over the bound is written and marked, never silently dropped: it is the evidence that
    #: the node was unusable, and a missing file would look like a job that never ran.
    ap.add_argument("--max-reference-spread", type=float, default=1.25)
    #: Abort BEFORE measuring anything if the node is already busy. The spread check catches a
    #: perturbed reference after the fact and costs a whole job to find out; this costs a second.
    #: 4 on a 48-core node is generous -- an exclusive allocation settles near 0 -- and it is a
    #: pre-condition on the measurement, not a threshold on the result.
    ap.add_argument("--max-load", type=float, default=0.0,
                    help="abort and write a rejected shard if the 1-minute load average exceeds "
                         "this at start; 0 disables the check")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch

    import bench_step3 as B3
    from bench_gpu_vs_cpu import gpu_throughput, make_env, set_device
    from bench_vs_decode import decode_baseline
    import stable_envs as se
    from stable_envs.testing import conformance as C

    if a.threads:
        torch.set_num_threads(a.threads)
    set_device(a.device)
    try:
        from stable_envs.envs.classic_control import _Classic
        _Classic.use_fused_render = (a.arm == "after")
    except ImportError:
        pass

    load0 = os.getloadavg()
    if a.max_load and load0[0] > a.max_load:
        doc = {"schema_version": 1, "step": 6, "shard": f"cpu-3d-paired-{a.arm}",
               "arm": a.arm, "envs": [],
               "certifiable": False,
               "not_certifiable_because":
                   f"the node's 1-minute load average was {load0[0]:.1f} at the start of this job, "
                   f"over the {a.max_load} bound, so the official renderer and the decode bar would "
                   f"have been timed against co-tenants. Nothing was measured; aborting before the "
                   f"measurement is cheaper than discovering it in the reference spread afterwards.",
               "load_average_at_start": load0,
               "provenance": {**B3.provenance(a.device, torch.get_num_threads()),
                              "load_average": load0,
                              "stable_envs_file": se.__file__,
                              "tree_commit": git_commit(Path(se.__file__).resolve().parents[1])}}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(doc, indent=1))
        print(f"ABORTED: load average {load0[0]:.1f} > {a.max_load} at start; wrote {a.out}")
        return 2

    batches = B3.batches_for(a.device, optional=False)
    universe = B3.matched_universe()
    scratch = Path(os.environ.get("SCR", "/tmp")) / ".cache"
    scratch.mkdir(parents=True, exist_ok=True)
    doc = {"schema_version": 1, "step": 6, "shard": f"cpu-3d-paired-{a.arm}",
           "what_this_is": "the frozen Step 3 sweep with the official renderer measured INTERLEAVED "
                           "with ours instead of in a block after it; same schema, same cells, same "
                           "decode bar, a denominator that is a median over the whole run",
           "arm": a.arm, "batches": batches,
           "provenance": {**B3.provenance(a.device, torch.get_num_threads()),
                          "load_average": load0,
                          "load_average_at_end": None,
                          "stable_envs_file": se.__file__,
                          "tree_commit": (git_commit(Path(se.__file__).resolve().parents[1]) if git_commit(Path(se.__file__).resolve().parents[1]) != "unknown" else os.environ.get("STABLE_ENVS_GIT_COMMIT", "unknown")),
                          "official_renderer_timing": "paired: one reference burst between every "
                                                      "batch cell, median of the readings",
                          "reps": a.reps, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"), "max_load_policy": ("checked" if a.max_load else "disabled: batch nodes are shared and the reference is PAIRED (interleaved bursts, median, spread recorded) so load enters the ratio symmetrically; the spread bound rejects a node whose load moved the reference"), "git_commit_source": ("git" if git_commit(Path(se.__file__).resolve().parents[1]) != "unknown" else "env:STABLE_ENVS_GIT_COMMIT")},
           "envs": []}
    print(f"{doc['provenance']['host']}  {a.device}  {torch.get_num_threads()} thread(s)  "
          f"arm={a.arm}  load {os.getloadavg()[0]:.1f}\n  tree {se.__file__}\n  batches {batches}")

    for name in a.envs.split(","):
        if name not in se.REGISTRY:
            print(f"[{name}] not registered")
            continue
        cls = se.REGISTRY[name]
        probe = make_env(name, 1, None, None)
        pool = np.asarray(np.load(ROOT / f"examples/results/step4/states/{name}.npz")["states"],
                          np.float32)
        frame0 = probe.render(pool[:1])
        hw = (int(frame0.shape[1]), int(frame0.shape[2]))
        rec = {"env": name, "projected_to": universe.get(name, [name]), "device": a.device,
               "size": int(probe.size), "supersample": int(probe.ss), "shape": f"{hw[0]}x{hw[1]}",
               "backend": getattr(probe, "backend", None), "renderer_class": type(probe).__name__,
               "reference_id": getattr(cls, "reference_id", None),
               "fidelity_tier": getattr(cls, "fidelity_tier", None),
               "state_dim": int(getattr(cls, "state_dim", 0) or 0),
               "states_source": str(ROOT / f"examples/results/step4/states/{name}.npz")}
        del probe

        ref_env = C.reference_env(cls, rec["size"])
        ref_states = pool[np.arange(a.ref_frames) % len(pool)]
        cls.reference_render(ref_env, ref_states[:4])

        def ref_burst():
            t0 = time.perf_counter()
            cls.reference_render(ref_env, ref_states)
            return a.ref_frames / (time.perf_counter() - t0)

        refs, cells, fps_seen = [ref_burst()], [], None
        for b in batches:
            if fps_seen and a.max_cell_s > 0 and b / fps_seen > a.max_cell_s:
                cells.append({"batch": b, "kind": "not_measured_here",
                              "skipped": f"projected {b / fps_seen:,.0f} s per repetition at the "
                                         f"{fps_seen:,.0f} frames/s already measured, over the "
                                         f"{a.max_cell_s:.0f} s per-cell budget"})
                continue
            cell = gpu_throughput(name, b, None, None, a.reps, a.max_out_gb, pool, hw)
            cells.append(cell)
            if cell.get("fps"):
                fps_seen = cell["fps"]
            refs.append(ref_burst())                 # between every cell, not once at the end
        rec["cells"] = cells
        med = statistics.median(refs)
        rec["official_cpu"] = {
            "frames": a.ref_frames, "size": rec["size"],
            "set_state_render": {"fps": med, "seconds_per_frame": 1.0 / med,
                                 "us_per_frame": 1e6 / med},
            "paired": {"readings": len(refs), "fps_median": med, "fps_min": min(refs),
                       "fps_max": max(refs), "spread": max(refs) / min(refs)}}
        f = None
        for c in cells:
            if c.get("fps"):
                e2 = make_env(name, 1, None, None)
                f = np.asarray(e2.render(pool[:1])[0].cpu(), np.uint8)
                del e2
                break
        rec["decode"] = decode_baseline(f, 90, a.decode_frames, scratch) if f is not None else None
        doc["envs"].append(rec)
        best = max((c["fps"] for c in cells if c.get("fps")), default=0)
        print(f"[{name:22s}] best {best:10,.0f} f/s   official {med:7.1f} "
              f"({min(refs):.0f}-{max(refs):.0f}, spread {max(refs)/min(refs):.2f}x)   "
              f"ratio {best/med:8.2f}   decode {(rec['decode'] or {}).get('decode_fps_per_core', 0):.0f}",
              flush=True)

    worst = max(((e["official_cpu"].get("paired") or {}).get("spread") or 1.0)
                for e in doc["envs"]) if doc["envs"] else 1.0
    doc["reference_spread_worst"] = worst
    doc["certifiable"] = worst <= a.max_reference_spread
    if not doc["certifiable"]:
        doc["not_certifiable_because"] = (
            f"the official renderer's throughput varied {worst:.2f}x across the interleaved "
            f"readings on this host, over the {a.max_reference_spread}x bound. S_cpu/R is a ratio "
            f"to that number, so a tier computed from this shard would describe the node's load "
            f"rather than the renderer. The numerator (our frames/s) is unaffected.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1))
    doc["provenance"]["load_average_at_end"] = os.getloadavg()
    doc["load_average_at_start"] = load0
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(f"wrote {a.out}  worst reference spread {worst:.2f}x  "
          f"certifiable={doc['certifiable']}  load {load0[0]:.1f} -> "
          f"{doc['provenance']['load_average_at_end'][0]:.1f}")
    return 0 if doc["certifiable"] else 3


if __name__ == "__main__":
    sys.exit(main())
