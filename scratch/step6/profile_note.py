"""Produce a Step 6 section-8 profile note from instrumentation, not from impressions.

Section 6 forbids editing code that "looks slow" and section 8 requires a written profile before
any patch. This produces that note mechanically: it times the REAL public rendering path, attributes
the time, counts kernel launches, records peak memory, and writes both JSON and the markdown note in
the shape section 8 asks for.

Three properties matter more than the pretty output:

**It measures the caller-visible path.** ``env.render(states)`` and nothing smaller. Section 9 is
explicit that profiling a helper the callers do not use and reporting it as a renderer improvement
is not evidence.

**It synchronises.** A CUDA timing that forgets to synchronise measures launch latency and calls it
throughput. Timings here are CUDA-event based on GPU and perf-counter based on CPU, with an explicit
synchronise before and after every timed region.

**It separates set_state from render.** These are different phases with different bottlenecks --
per-env state unpacking in Python is a classic structural problem, and folding it into "render time"
hides it. Both are reported.

Attribution is op-level, from ``torch.profiler``, and is labelled as such. It is not hand-placed
phase instrumentation inside the renderer: that would be an edit, and this runs before any edit is
allowed. Where a workstream needs finer phases than the operator trace gives, it should add real
timers in its own branch and say so in its note.

    python scratch/step6/profile_note.py --env PushT --device cuda --batch 256 \
        --out examples/results/step6/<workstream>/profile_before/
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Operator-name fragments mapped to the section-8 phase vocabulary. Deliberately coarse: this is an
#: attribution HINT over an operator trace, and a note that claims more precision than the trace
#: supports is worse than one that admits the granularity.
PHASE_HINTS = (
    ("host->device", ("copy_", "to", "pin_memory", "_to_copy")),
    ("allocation", ("empty", "zeros", "ones", "full", "new_")),
    ("kernel/raster", ("triton", "conv", "gemm", "mm", "bmm", "index", "gather", "scatter",
                       "where", "clamp", "mul", "add", "sub", "div", "cat", "stack")),
    ("output conversion", ("uint8", "byte", "permute", "contiguous", "clip", "round")),
    ("synchronization", ("synchronize", "item", "cpu")),
)


def classify(name: str) -> str:
    low = name.lower()
    for phase, frags in PHASE_HINTS:
        if any(f in low for f in frags):
            return phase
    return "other"


def rel(p: Path) -> str:
    """Repo-relative when it can be, absolute when it cannot.

    ``Path.relative_to`` raises for anything outside ROOT -- including a path the caller passed in
    relative form, which resolves against the CWD. Step 4 hit exactly this and grew the same
    helper; reusing it here rather than rediscovering it a third time.
    """
    p = p.resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return "unknown"


def versions() -> dict:
    import importlib.metadata as md
    out = {}
    for p in ("torch", "triton", "numpy", "mujoco", "dm-control", "mujoco-warp", "warp-lang"):
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = "absent"
    return out


def time_call(fn, device: str, repeat: int, warmup: int) -> dict:
    """Wall time per call, synchronised. Returns milliseconds, not a mean with no spread."""
    import torch

    cuda = device.startswith("cuda") and torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        if cuda:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            samples.append(a.elapsed_time(b))
        else:
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1e3)
    return {
        "ms_median": statistics.median(samples),
        "ms_min": min(samples),
        "ms_max": max(samples),
        "ms_stdev": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "samples": len(samples),
    }


def op_trace(fn, device: str) -> dict:
    """One profiled iteration, aggregated by the section-8 phase vocabulary."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU]
    cuda = device.startswith("cuda") and torch.cuda.is_available()
    if cuda:
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts, record_shapes=False, profile_memory=True) as prof:
        fn()
        if cuda:
            torch.cuda.synchronize()

    evs = prof.key_averages()
    total_cpu = sum(e.self_cpu_time_total for e in evs) or 1
    dev_attr = "self_device_time_total" if hasattr(evs[0], "self_device_time_total") else \
               "self_cuda_time_total"
    total_dev = sum(getattr(e, dev_attr, 0) or 0 for e in evs)
    phases: dict[str, dict] = {}
    for e in evs:
        ph = classify(e.key)
        d = phases.setdefault(ph, {"cpu_us": 0.0, "device_us": 0.0, "calls": 0})
        d["cpu_us"] += e.self_cpu_time_total
        d["device_us"] += getattr(e, dev_attr, 0) or 0
        d["calls"] += e.count
    top = sorted(evs, key=lambda e: -(getattr(e, dev_attr, 0) or e.self_cpu_time_total))[:15]
    return {
        "phases": {k: {**v, "cpu_pct": 100 * v["cpu_us"] / total_cpu,
                       "device_pct": (100 * v["device_us"] / total_dev) if total_dev else None}
                   for k, v in sorted(phases.items(), key=lambda kv: -kv[1]["device_us"])},
        "total_cpu_us": total_cpu, "total_device_us": total_dev,
        "op_count": sum(e.count for e in evs),
        "top_ops": [{"op": e.key, "calls": e.count,
                     "cpu_us": e.self_cpu_time_total,
                     "device_us": getattr(e, dev_attr, 0) or 0} for e in top],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 256])
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--workstream", default="")
    # The CPU certification budget is ONE thread (section 3.3), and 102 of the Step 3 CPU rows were
    # measured that way. Profiling on all cores and comparing against a one-thread baseline would
    # be the "different machines for numerator and denominator" invariant in section 12, wearing a
    # thread count instead of a hostname.
    ap.add_argument("--threads", type=int, default=None,
                    help="torch thread budget; use 1 to match the CPU certification metric")
    ap.add_argument("--states-dir", type=Path,
                    default=ROOT / "examples" / "results" / "step4" / "states",
                    help="frozen Step 4 corpora, read-only; profiled on real reachable states")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np
    import torch
    import stable_envs as se

    if args.threads:
        torch.set_num_threads(args.threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    note: dict = {
        "workstream": args.workstream, "env": args.env, "device": args.device,
        "baseline_commit": git_commit(), "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu": platform.processor() or platform.machine(),
        "threads": torch.get_num_threads(), "versions": versions(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"), "batches": {},
    }

    env = se.make(args.env, batch=max(args.batch), size=args.size, device=args.device)
    note["size"] = int(getattr(env, "size", args.size) or 0)
    note["state_dim"] = int(env.state_dim)

    # Profile the states the frozen corpus holds, so the note describes the same scenes Step 4
    # certified rather than a convenient synthetic pose.
    npz = args.states_dir.expanduser().resolve() / f"{args.env}.npz"
    if npz.is_file():
        pool = np.load(npz)["states"]
        note["states_source"] = rel(npz)
    else:
        pool = env.sample_states(64)
        note["states_source"] = "env.sample_states (no frozen corpus for this env)"

    for B in args.batch:
        idx = np.arange(B) % len(pool)
        states = np.asarray(pool[idx], np.float32)
        e = se.make(args.env, batch=B, size=args.size, device=args.device)

        # set_state and render are timed apart: per-env state unpacking in Python is a structural
        # bottleneck in its own right and folding it into "render" is how it stays hidden.
        rec = {"batch": B}
        rec["set_state_ms"] = time_call(lambda: e.set_state(states), args.device,
                                        args.repeat, args.warmup)
        rec["render_ms"] = time_call(lambda: e.render(states), args.device,
                                     args.repeat, args.warmup)
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            e.render(states)
            torch.cuda.synchronize()
            rec["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
        rec["trace"] = op_trace(lambda: e.render(states), args.device)
        m = rec["render_ms"]["ms_median"]
        rec["fps"] = 1000.0 * B / m if m else None
        note["batches"][str(B)] = rec
        print(f"B={B:<6} render {m:8.2f} ms  ({rec['fps']:9.1f} fps)  "
              f"set_state {rec['set_state_ms']['ms_median']:7.2f} ms  "
              f"ops {rec['trace']['op_count']}")

    args.out.mkdir(parents=True, exist_ok=True)
    # HOST IN THE FILENAME. Without it an Oscar run silently overwrites a laptop run of the same
    # env and the directory becomes a mix of two machines -- which is section 12's "different
    # machines for numerator and denominator", arrived at by filename collision rather than by
    # choice. Acrobot read 109 fps locally and 20 fps on Oscar; one file cannot hold both.
    # ``note``, not ``n``: the scaffolding commit wrote ``n["host"]`` here, which is the parameter
    # name of ``render_note`` and is not bound in this function. Every run therefore printed its
    # timings, raised ``NameError`` and wrote NO file -- the profile existed only in the SLURM log.
    host = note["host"].split(".")[0]
    tag = f"{args.env}_{args.device.replace(':', '')}_{host}"
    (args.out / f"{tag}.json").write_text(json.dumps(note, indent=2))
    (args.out / f"{tag}.md").write_text(render_note(note))
    print(f"\nwrote {args.out / (tag + '.json')} and {tag}.md")


def render_note(n: dict) -> str:
    """The section-8 note, generated. A hand-written one is a hand-written one."""
    L = [f"# Profile note — {n['env']} on {n['device']}", "",
         f"- workstream: `{n['workstream'] or '(unset)'}`",
         f"- baseline commit: `{n['baseline_commit'][:12]}`",
         f"- host: {n['host']}  ·  gpu: {n['gpu'] or 'none'}  ·  cpu: {n['cpu']}  "
         f"·  threads: {n['threads']}",
         f"- frame: {n['size']}px  ·  state_dim: {n['state_dim']}",
         f"- states: `{n['states_source']}`",
         f"- torch {n['versions'].get('torch')}  ·  triton {n['versions'].get('triton')}", ""]
    for B, rec in n["batches"].items():
        t, r = rec["trace"], rec["render_ms"]
        L += [f"## batch {B}", "",
              f"render **{r['ms_median']:.2f} ms** median "
              f"(min {r['ms_min']:.2f}, max {r['ms_max']:.2f}, sd {r['ms_stdev']:.2f}) "
              f"= **{rec['fps']:.0f} fps**  ·  "
              f"set_state {rec['set_state_ms']['ms_median']:.2f} ms"
              + (f"  ·  peak VRAM {rec['peak_vram_mib']:.0f} MiB" if "peak_vram_mib" in rec else ""),
              "",
              "| phase (op-level attribution) | device us | device % | cpu us | calls |",
              "|---|---|---|---|---|"]
        for ph, v in t["phases"].items():
            dp = f"{v['device_pct']:.1f}" if v["device_pct"] is not None else "-"
            L.append(f"| {ph} | {v['device_us']:.0f} | {dp} | {v['cpu_us']:.0f} | {v['calls']} |")
        L += ["", f"{t['op_count']} operator calls per render. Top by device time:", "",
              "| op | calls | device us | cpu us |", "|---|---|---|---|"]
        for o in t["top_ops"][:8]:
            L.append(f"| `{o['op']}` | {o['calls']} | {o['device_us']:.0f} | {o['cpu_us']:.0f} |")
        L.append("")
    L += ["## largest bottleneck", "", "_to be completed by the workstream from the table above._",
          "", "## structural hypothesis", "", "_..._", "",
          "## proposed intervention", "", "_..._", "",
          "## correctness/fidelity risk", "", "_..._", "",
          "## affected Step 4 configurations to replay", "",
          "_from `examples/results/step6/plan.json`, field `step4_replay_configs`._", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
