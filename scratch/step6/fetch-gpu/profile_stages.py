"""Stage-resolved profile of the PUBLIC Fetch render path, to explain BATCH_SCALES.

Step 6D asks one question: why does ``fps(4096)/fps(1)`` reach only 7-12x on the Fetch MJWarp path
when the frozen qualifier is 20x, while the throughput ratio R already passes at 5.3 against a bar
of 4. ``scratch/step6/profile_note.py`` answers "where does the wall time go" at OPERATOR
granularity; that is not enough here, because the operator trace cannot tell a per-frame cost that
amortises from one that does not. This adds the two things it is missing:

1. **A wall-time partition of ``render()`` that sums to the whole.** Every stage boundary is a
   function the env already calls -- ``_plate``, ``_camera_batch``, ``_context``, ``_upload``,
   ``torch.cuda.synchronize``, ``warp.capture_launch``, ``warp.synchronize`` -- wrapped from the
   OUTSIDE, so no renderer or env code is edited (this workstream is authorised to profile only).
   Everything between two wrapped calls is charged to a named GAP, so the partition is exhaustive
   and a reader can see what is unattributed instead of trusting a residual.

2. **A device-side partition from the kineto trace.** Kernel names are bucketed into the MJWarp
   ray-trace megakernel, the MJWarp scene-update kernels (kinematics / camlight / refit_bvh), and
   the torch composite that turns the raw buffer into the frame (channel swap, mover lookup,
   where). Memcpys are counted WITH THEIR BYTES and split by direction, which ``key_averages``
   does not expose -- that is the only way to answer "is static geometry or texture re-uploaded
   per frame".

The decisive experiment is ``--sizes``: the SAME code path at several resolutions. ``render(size=)``
is a public argument, so this is still the caller-visible path. If BATCH_SCALES passes at 84 px and
fails at 480 px with no code change, the failure is occupancy at the frozen resolution and not a
structural defect, and that changes what a patch could possibly do.

Usage (one GPU job, mujoco 3.11 Fetch stack):

    python scratch/step6/fetch-gpu/profile_stages.py --env FetchReach \
        --batch 1 16 64 256 1024 4096 --sizes 84 120 240 480 \
        --out examples/results/step6/fetch-gpu/profile_stages/
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import socket
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------------------------
# Outside-in instrumentation. Nothing here edits stable_envs; every patch is installed on an
# already-imported module object and removed in ``restore``.
# ---------------------------------------------------------------------------------------------

EVENTS: list = []          # (label, t_enter, t_exit)
ON = False
_PATCHES: list = []


def patch(owner, attr: str, label: str) -> None:
    orig = getattr(owner, attr)

    def wrapper(*a, **k):
        if not ON:
            return orig(*a, **k)
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            EVENTS.append((label, t0, time.perf_counter()))

    wrapper.__wrapped__ = orig
    setattr(owner, attr, wrapper)
    _PATCHES.append((owner, attr, orig))


def restore() -> None:
    while _PATCHES:
        owner, attr, orig = _PATCHES.pop()
        setattr(owner, attr, orig)


def install(fetch_mod) -> None:
    import warp as wp

    import torch
    patch(fetch_mod, "_context", "context")
    patch(fetch_mod.FetchBase, "_upload", "upload")
    patch(fetch_mod.FetchBase, "_camera_batch", "camera_batch")
    patch(fetch_mod.FetchBase, "_plate", "plate")
    patch(torch.cuda, "synchronize", "torch_sync")
    patch(wp, "capture_launch", "graph_launch")
    patch(wp, "synchronize", "warp_sync")


def partition(t0: float, t1: float) -> dict:
    """Charge every microsecond of ``[t0, t1]`` to a named stage or a named gap between stages."""
    ev = sorted(EVENTS, key=lambda e: e[1])
    out: dict = defaultdict(float)
    calls: dict = defaultdict(int)
    prev_label, prev_end = "enter", t0
    for label, a, b in ev:
        out[f"gap:{prev_label}->{label}"] += a - prev_end
        out[label] += b - a
        calls[label] += 1
        prev_label, prev_end = label, b
    out[f"gap:{prev_label}->return"] += t1 - prev_end
    return {"ms": {k: v * 1e3 for k, v in sorted(out.items(), key=lambda kv: -kv[1])},
            "calls": dict(calls), "total_ms": (t1 - t0) * 1e3}


# ---------------------------------------------------------------------------------------------
# Device side
# ---------------------------------------------------------------------------------------------

KERNEL_BUCKETS = (
    ("mjwarp_raytrace_megakernel", ("render_megakernel",)),
    ("mjwarp_scene_update", ("kinematics", "camlight", "com_pos", "refit", "bvh", "_mjw",
                             "transform", "local_to_global")),
    ("torch_composite_output", ("index_elementwise", "elementwise_kernel", "where_kernel",
                                "gpu_kernel", "vectorized_elementwise", "CatArrayBatched",
                                "unrolled_elementwise", "copy_device")),
)


def bucket(name: str) -> str:
    low = name.lower()
    for b, frags in KERNEL_BUCKETS:
        if any(f.lower() in low for f in frags):
            return b
    return "kernel_other"


def kineto(fn, tmpdir: Path) -> dict:
    """One profiled iteration -> kernel buckets, kernel count, memcpy count AND BYTES by direction."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    path = tmpdir / "trace.json"
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    tr = json.loads(raw.decode("utf-8", "replace"))
    evs = tr["traceEvents"] if isinstance(tr, dict) else tr

    kern: dict = defaultdict(lambda: {"calls": 0, "device_us": 0.0})
    per_kernel: dict = defaultdict(lambda: {"calls": 0, "device_us": 0.0})
    mem: dict = defaultdict(lambda: {"calls": 0, "bytes": 0, "device_us": 0.0})
    for e in evs:
        cat = (e.get("cat") or "").lower()
        dur = float(e.get("dur") or 0.0)
        name = e.get("name") or ""
        if cat in ("kernel", "gpu_user_annotation") and cat == "kernel":
            b = bucket(name)
            kern[b]["calls"] += 1
            kern[b]["device_us"] += dur
            per_kernel[name[:110]]["calls"] += 1
            per_kernel[name[:110]]["device_us"] += dur
        elif cat in ("gpu_memcpy", "gpu_memset"):
            args = e.get("args") or {}
            d = name  # e.g. "Memcpy HtoD (Pageable -> Device)"
            mem[d]["calls"] += 1
            mem[d]["bytes"] += int(args.get("bytes") or 0)
            mem[d]["device_us"] += dur
    total_dev = sum(v["device_us"] for v in kern.values()) + sum(v["device_us"] for v in mem.values())
    return {
        "kernel_buckets": {k: {**v, "device_pct": 100 * v["device_us"] / total_dev if total_dev else 0}
                           for k, v in sorted(kern.items(), key=lambda kv: -kv[1]["device_us"])},
        "memcpy": {k: dict(v) for k, v in sorted(mem.items(), key=lambda kv: -kv[1]["bytes"])},
        "total_device_us": total_dev,
        "kernel_launches": sum(v["calls"] for v in kern.values()),
        "top_kernels": [{"kernel": k, **v} for k, v in
                        sorted(per_kernel.items(), key=lambda kv: -kv[1]["device_us"])[:12]],
    }


# ---------------------------------------------------------------------------------------------
# Sync counting. A separate pass, so the counters never sit inside a timed region.
# ---------------------------------------------------------------------------------------------

def count_syncs(fetch_mod, fn) -> dict:
    import warp as wp

    import torch
    counts: dict = defaultdict(int)
    saved = []

    def wrap(owner, attr, label):
        orig = getattr(owner, attr)

        def w(*a, **k):
            counts[label] += 1
            return orig(*a, **k)
        try:
            setattr(owner, attr, w)
            saved.append((owner, attr, orig))
        except Exception:
            counts[label + ":unpatchable"] = -1

    for attr in ("cpu", "item", "tolist", "numpy"):
        wrap(torch.Tensor, attr, f"Tensor.{attr}")
    wrap(torch.cuda, "synchronize", "torch.cuda.synchronize")
    wrap(wp, "synchronize", "warp.synchronize")
    wrap(wp, "capture_launch", "warp.capture_launch")
    wrap(fetch_mod, "_context", "_context")
    wrap(fetch_mod.FetchBase, "_upload", "_upload")
    try:
        fn()
    finally:
        for owner, attr, orig in reversed(saved):
            setattr(owner, attr, orig)
    return dict(counts)


# ---------------------------------------------------------------------------------------------

def cuda_event_time(fn, repeat: int, warmup: int) -> dict:
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(repeat):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        s.append(a.elapsed_time(b))
    return {"ms_median": statistics.median(s), "ms_min": min(s), "ms_max": max(s),
            "ms_stdev": statistics.pstdev(s) if len(s) > 1 else 0.0, "samples": len(s)}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return "unknown"


def versions() -> dict:
    import importlib.metadata as md
    out = {}
    for p in ("torch", "triton", "numpy", "mujoco", "mujoco-warp", "warp-lang",
              "gymnasium", "gymnasium-robotics"):
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = "absent"
    return out


def upload_bytes(env, n: int) -> dict:
    """What ``_upload`` writes into the context per chunk, in bytes, split static vs dynamic.

    ``qpos`` and ``mocap_pos`` are the state -- genuinely dynamic. ``mat_rgba``, ``light_diffuse``,
    ``cam_pos`` and ``cam_quat`` are per-world MODEL fields: under the default style they carry the
    same numbers on every render, so every byte of them is a re-upload of something static.
    """
    b = env._b
    m = b.model
    nq = int(b.nq)
    f4 = 4
    dyn = n * nq * f4 + n * 3 * f4
    stat = (n * int(m.nmat) * 4 * f4          # mat_rgba (nworld, nmat, 4)
            + n * int(m.nlight) * 3 * f4      # light_diffuse
            + n * 3 * f4 + n * 4 * f4)        # cam_pos, cam_quat (one camera row each)
    return {"dynamic_state_bytes_per_chunk": int(dyn),
            "static_appearance_bytes_per_chunk": int(stat),
            "nq": nq, "nmat": int(m.nmat), "nlight": int(m.nlight), "ngeom": int(m.ngeom),
            "nbody": int(m.nbody)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 16, 64, 256, 1024, 4096])
    ap.add_argument("--sizes", type=int, nargs="*", default=[],
                    help="resolution sweep at --size-batch, the controlled occupancy experiment")
    ap.add_argument("--size-batch", type=int, nargs="+", default=[1, 256, 4096])
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--states-dir", type=Path,
                    default=ROOT / "examples" / "results" / "step4" / "states")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np
    import torch

    import stable_envs as se
    from stable_envs.envs import fetch as fetch_mod

    note: dict = {
        "workstream": "step6/fetch-gpu", "env": args.env, "device": "cuda",
        "what": "stage-resolved wall partition + device kernel partition of the public render path",
        "baseline_commit": git_commit(), "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0), "cpu": platform.processor() or platform.machine(),
        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "max_threads_per_sm": getattr(torch.cuda.get_device_properties(0),
                                      "max_threads_per_multi_processor", None),
        "versions": versions(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "mjw_chunk": int(fetch_mod.MJW_CHUNK),
        "batches": {}, "resolution_sweep": {},
    }

    npz = args.states_dir.expanduser().resolve() / f"{args.env}.npz"
    pool = np.load(npz)["states"]
    note["states_source"] = str(npz)

    probe = se.make(args.env, batch=1, device="cuda")
    note["size"] = int(probe.size)
    note["state_dim"] = int(probe.state_dim)
    note["model_dims"] = upload_bytes(probe, 1)
    del probe

    tmp = Path(tempfile.mkdtemp(prefix="se6-fetch-"))

    global ON
    install(fetch_mod)
    try:
        for B in args.batch:
            print(f"[stages] {args.env} B={B}", flush=True)
            idx = np.arange(B) % len(pool)
            states = np.asarray(pool[idx], np.float32)
            e = se.make(args.env, batch=B, device="cuda")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            st = torch.as_tensor(states, device="cuda")
            rec: dict = {"batch": B}
            rec["set_state_ms"] = cuda_event_time(lambda: e.set_state(states),
                                                  args.repeat, args.warmup)
            e.set_state(states)
            rec["render_ms"] = cuda_event_time(lambda: e.render(st), args.repeat, args.warmup)
            rec["fps"] = B / (rec["render_ms"]["ms_median"] / 1e3)

            # exhaustive wall partition, median over `repeat` renders
            parts = []
            for _ in range(args.repeat):
                EVENTS.clear()
                torch.cuda.synchronize()
                ON = True
                t0 = time.perf_counter()
                e.render(st)
                t1 = time.perf_counter()
                ON = False
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                p = partition(t0, t1)
                p["ms"]["tail_drain(after return)"] = (t2 - t1) * 1e3
                p["total_ms"] = (t2 - t0) * 1e3
                parts.append(p)
            keys = sorted({k for p in parts for k in p["ms"]})
            rec["stage_ms"] = {k: round(statistics.median([p["ms"].get(k, 0.0) for p in parts]), 4)
                               for k in keys}
            rec["stage_ms"] = dict(sorted(rec["stage_ms"].items(), key=lambda kv: -kv[1]))
            rec["stage_calls"] = parts[len(parts) // 2]["calls"]
            rec["stage_total_ms"] = round(statistics.median([p["total_ms"] for p in parts]), 4)
            rec["stage_ms_per_frame"] = {k: round(v / B, 5) for k, v in rec["stage_ms"].items()}

            rec["device"] = kineto(lambda: e.render(st), tmp)
            rec["syncs_per_render"] = count_syncs(fetch_mod, lambda: e.render(st))
            rec["upload_bytes"] = upload_bytes(e, min(B, fetch_mod.MJW_CHUNK))
            rec["chunks"] = -(-B // fetch_mod.MJW_CHUNK)
            rec["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
            rec["peak_vram_reserved_mib"] = torch.cuda.max_memory_reserved() / 2 ** 20
            note["batches"][str(B)] = rec
            del e, st
            torch.cuda.empty_cache()

        # ---- the controlled experiment: one code path, several resolutions ----------------
        for S in args.sizes:
            for B in args.size_batch:
                print(f"[res] {args.env} size={S} B={B}", flush=True)
                idx = np.arange(B) % len(pool)
                st = torch.as_tensor(np.asarray(pool[idx], np.float32), device="cuda")
                e = se.make(args.env, batch=B, device="cuda")
                torch.cuda.reset_peak_memory_stats()
                try:
                    r = cuda_event_time(lambda: e.render(st, size=S), args.repeat, args.warmup)
                except RuntimeError as exc:                       # OOM at the largest cell
                    note["resolution_sweep"].setdefault(str(S), {})[str(B)] = {"error": str(exc)[:200]}
                    torch.cuda.empty_cache()
                    continue
                note["resolution_sweep"].setdefault(str(S), {})[str(B)] = {
                    "ms_median": r["ms_median"], "fps": B / (r["ms_median"] / 1e3),
                    "us_per_frame": r["ms_median"] * 1e3 / B,
                    "peak_vram_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                    "rays_per_frame": S * S}
                del e, st
                torch.cuda.empty_cache()
    finally:
        restore()

    for S, cells in note["resolution_sweep"].items():
        f = {b: c.get("fps") for b, c in cells.items() if "fps" in c}
        if 1 in [int(k) for k in f] and len(f) > 1:
            hi = max(int(k) for k in f)
            note["resolution_sweep"][S]["batch_scales"] = f[str(hi)] / f["1"]

    args.out.mkdir(parents=True, exist_ok=True)
    tag = socket.gethostname().split(".")[0]
    jp = args.out / f"{args.env}_stages_{tag}.json"
    jp.write_text(json.dumps(note, indent=1, sort_keys=False))
    print("wrote", jp)
    write_md(note, args.out / f"{args.env}_stages_{tag}.md")


def write_md(n: dict, path: Path) -> None:
    L = [f"# Stage profile — {n['env']} on cuda ({n['gpu']})", "",
         f"- workstream: `{n['workstream']}`  ·  baseline `{n['baseline_commit'][:12]}`",
         f"- host {n['host']}  ·  slurm {n['slurm_job']}  ·  {n['sm_count']} SMs  ·  "
         f"{n['gpu_total_gb']} GB",
         f"- frame {n['size']}px  ·  MJW_CHUNK {n['mjw_chunk']}  ·  "
         f"mujoco {n['versions']['mujoco']} / mujoco-warp {n['versions']['mujoco-warp']} / "
         f"warp {n['versions']['warp-lang']}",
         f"- model: {n['model_dims']}", "",
         "## headline", "",
         "| B | render ms | fps | us/frame | set_state ms | chunks | kernels | peak VRAM MiB |",
         "|---|---|---|---|---|---|---|---|"]
    for b, r in n["batches"].items():
        L.append(f"| {b} | {r['render_ms']['ms_median']:.3f} | {r['fps']:.0f} | "
                 f"{r['render_ms']['ms_median']*1e3/int(b):.1f} | "
                 f"{r['set_state_ms']['ms_median']:.3f} | {r['chunks']} | "
                 f"{r['device']['kernel_launches']} | {r['peak_vram_mib']:.0f} |")
    L += ["", "## wall partition (ms, median; sums to the whole render)", ""]
    keys = sorted({k for r in n["batches"].values() for k in r["stage_ms"]})
    L.append("| stage | " + " | ".join(f"B={b}" for b in n["batches"]) + " |")
    L.append("|---" * (len(n["batches"]) + 1) + "|")
    for k in sorted(keys, key=lambda k: -max(r["stage_ms"].get(k, 0) for r in n["batches"].values())):
        L.append(f"| `{k}` | " + " | ".join(f"{r['stage_ms'].get(k, 0):.3f}"
                                            for r in n["batches"].values()) + " |")
    L.append("| **total** | " + " | ".join(f"{r['stage_total_ms']:.3f}"
                                           for r in n["batches"].values()) + " |")
    L += ["", "## device partition (us, one profiled render)", ""]
    dk = sorted({k for r in n["batches"].values() for k in r["device"]["kernel_buckets"]})
    L.append("| bucket | " + " | ".join(f"B={b}" for b in n["batches"]) + " |")
    L.append("|---" * (len(n["batches"]) + 1) + "|")
    for k in dk:
        L.append(f"| {k} | " + " | ".join(
            f"{r['device']['kernel_buckets'].get(k, {}).get('device_us', 0):.0f} "
            f"({r['device']['kernel_buckets'].get(k, {}).get('calls', 0)})"
            for r in n["batches"].values()) + " |")
    L += ["", "## memcpy per render (count, bytes)", ""]
    mk = sorted({k for r in n["batches"].values() for k in r["device"]["memcpy"]})
    L.append("| direction | " + " | ".join(f"B={b}" for b in n["batches"]) + " |")
    L.append("|---" * (len(n["batches"]) + 1) + "|")
    for k in mk:
        L.append(f"| {k} | " + " | ".join(
            f"{r['device']['memcpy'].get(k, {}).get('calls', 0)} / "
            f"{r['device']['memcpy'].get(k, {}).get('bytes', 0)}B"
            for r in n["batches"].values()) + " |")
    L += ["", "## host<->device sync points per render", ""]
    sk = sorted({k for r in n["batches"].values() for k in r["syncs_per_render"]})
    L.append("| call | " + " | ".join(f"B={b}" for b in n["batches"]) + " |")
    L.append("|---" * (len(n["batches"]) + 1) + "|")
    for k in sk:
        L.append(f"| `{k}` | " + " | ".join(str(r["syncs_per_render"].get(k, 0))
                                            for r in n["batches"].values()) + " |")
    if n["resolution_sweep"]:
        L += ["", "## resolution sweep — the same code path at several frame sizes", "",
              "| size | " + " | ".join(f"fps(B={b})" for b in
                                       sorted({b for c in n['resolution_sweep'].values()
                                               for b in c if b.isdigit()}, key=int))
              + " | fps(max)/fps(1) |", "|---" * 5 + "|"]
        bs = sorted({b for c in n["resolution_sweep"].values() for b in c if b.isdigit()}, key=int)
        for S, cells in sorted(n["resolution_sweep"].items(), key=lambda kv: int(kv[0])):
            row = " | ".join(f"{cells[b]['fps']:.0f}" if b in cells and "fps" in cells[b] else "-"
                             for b in bs)
            L.append(f"| {S}px | {row} | {cells.get('batch_scales', float('nan')):.1f}x |")
    L += ["", "## largest bottleneck", "", "_filled in findings.md_", ""]
    path.write_text("\n".join(L) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
