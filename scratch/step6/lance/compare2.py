"""The Step-6 deliverable table: our renderer against the production 8-worker JPEG-in-Lance pipeline.

Two break-even questions are kept SEPARATE and both are printed, because they are different
questions and Step 2 froze the first one:

1. **Conservative, in-memory, per-core** — `cores` / `cores_cpu` as `certification.md` §5.1/§6.3
   define them: our fps divided by ONE core of JPEG-**q90** decode of this env's own frame, taken
   from the `decode` block of the committed Step-3/Step-6 shards. Frozen; reproduced here unchanged.
2. **Realistic, stored-dataset, training-pipeline** — our fps against the 8-worker Lance pipeline
   measured on Oscar at each env's certified resolution, at q95, with the GPU side compared
   end-to-end (frames on the device on both sides, so the stored path is charged for its H2D copy).

Neither is substituted for the other anywhere in the output.

Inputs (all optional except the first two; every cell records the file it came from):
  --lance-cpu / --lance-gpu          run_lance_oscar.py output (3 arms, worker sweep)
  --lance-arms2-cpu / --gpu          lance_arms2.py output (CPU accounting, read bandwidth)
  --jpeg-to-gpu                      bench_jpeg_to_gpu.py output (loader8+H2D, nvJPEG+H2D, H2D alone)
  --cpu-8way                         bench_cpu_parallel.py output (1thr / 8thr / 8proc)
  --cpu1-shard                       bench_step3 --device cpu --threads 1 shard
  --gpu-shard                        bench_step3 --device cuda shard (renderer-only, post-Step-6)
  --pipeline / --pipeline-fetch      bench_step3_pipeline.py output (state -> device, end-to-end)
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

MANDATORY = [1, 16, 64, 256, 1024, 4096]


def load(p):
    return json.loads(Path(p).read_text()) if p and Path(p).exists() else None


def shard_rows(d, path):
    """{env: {fps, batch, cells, decode_q90_fps_per_core, host, ...}} from a bench_step3 shard."""
    out = {}
    if not d:
        return out
    prov = d.get("provenance", {})
    for e in d.get("envs", []):
        cells = [c for c in e.get("cells", []) if c.get("fps") and c.get("batch") in MANDATORY]
        if not cells:
            continue
        best = max(cells, key=lambda c: c["fps"])
        out[e["env"]] = {
            "fps": best["fps"], "batch": best["batch"],
            "cells": {c["batch"]: c["fps"] for c in e["cells"] if c.get("fps")},
            "shape": e.get("shape"), "size": e.get("size"), "supersample": e.get("supersample"),
            "backend": e.get("backend"), "tier": e.get("fidelity_tier"),
            "decode_q90_fps_per_core": (e.get("decode") or {}).get("decode_fps_per_core"),
            "decode_q90_jpeg_bytes": (e.get("decode") or {}).get("jpeg_bytes"),
            "official_cpu_fps": (e.get("official_cpu") or {}).get("fps"),
            "host": prov.get("host"), "device_name": prov.get("device_name"),
            "threads": prov.get("torch_threads"), "slurm_job_id": prov.get("slurm_job_id"),
            "git_commit": prov.get("git_commit"), "source": str(path),
        }
    return out


def pipeline_rows(d, path):
    """{env: {batch: {arm: fps}}} from bench_step3_pipeline."""
    out = {}
    if not d:
        return out
    prov = d.get("provenance", {})
    for e in d.get("envs", []):
        arms = e.get("arms") or {}
        cells = {}
        for b, a in arms.items():
            cells[int(b)] = {k: v.get("fps") for k, v in a.items() if isinstance(v, dict)}
        if cells:
            out[e["env"]] = {"cells": cells, "host": prov.get("host"),
                             "device_name": prov.get("device_name"),
                             "slurm_job_id": prov.get("slurm_job_id"), "source": str(path)}
    return out


def fmt(x, w=9):
    return f"{x:>{w},.0f}" if isinstance(x, (int, float)) else " " * (w - 1) + "-"


def fmtr(x, w=6):
    return f"{x:>{w},.2f}" if isinstance(x, (int, float)) else " " * (w - 1) + "-"


def pct(x):
    return f"{x*100:.2f}%" if isinstance(x, (int, float)) else "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--lance-cpu", required=True)
    ap.add_argument("--lance-gpu", default=None)
    ap.add_argument("--lance-arms2-cpu", default=None)
    ap.add_argument("--lance-arms2-gpu", default=None)
    ap.add_argument("--jpeg-to-gpu", default=None)
    ap.add_argument("--cpu-8way", default=None)
    ap.add_argument("--cpu1-shard", action="append", default=[])
    ap.add_argument("--gpu-shard", action="append", default=[])
    ap.add_argument("--pipeline", action="append", default=[])
    ap.add_argument("--randall", default="examples/results/storage/lance.json")
    ap.add_argument("--quality-probe", default=None)
    ap.add_argument("--fallback-rows", default="examples/results/step6/lance/renderer_rows.json")
    ap.add_argument("--out-json", default="examples/results/step6/lance/comparison2.json")
    ap.add_argument("--out-md", default="examples/results/step6/lance/comparison2.md")
    a = ap.parse_args()
    root = Path(a.root).resolve()

    lcpu = load(a.lance_cpu)
    lgpu = load(a.lance_gpu)
    a2c = load(a.lance_arms2_cpu)
    a2g = load(a.lance_arms2_gpu)
    j2g = load(a.jpeg_to_gpu)
    c8 = load(a.cpu_8way)
    randall = (load(root / a.randall) or {}).get("envs", {})
    qp = load(a.quality_probe)

    cpu1 = {}
    for p in a.cpu1_shard:
        cpu1.update(shard_rows(load(p), p))
    gpuR = {}
    for p in a.gpu_shard:
        gpuR.update(shard_rows(load(p), p))
    pipe = {}
    for p in a.pipeline:
        pipe.update(pipeline_rows(load(p), p))

    # fallback for cells no fresh shard covers (Fetch CPU has none by construction)
    fb = load(root / a.fallback_rows) or {"rows": []}
    fb_by = {}
    for r in fb["rows"]:
        fb_by.setdefault((r["env"], r["device"], r.get("threads")), []).append(r)

    def fallback(env, device, threads):
        cands = [r for r in fb_by.get((env, device, threads), [])
                 if r.get("host") and "oscar" in r["host"]
                 and not any(x in r["source"] for x in ("before", "superseded", "rejected",
                                                        "control", "m5_", "order_check",
                                                        "mjwarp_1thread"))]
        return max(cands, key=lambda r: r["fps"]) if cands else None

    rows = []
    for env in sorted(lcpu["envs"]):
        L = lcpu["envs"][env]
        G = ((lgpu or {}).get("envs", {}) or {}).get(env, {})
        A2 = ((a2c or {}).get("envs", {}) or {}).get(env, {})
        A2G = ((a2g or {}).get("envs", {}) or {}).get(env, {})
        J = ((j2g or {}).get("envs", {}) or {}).get(env, {})
        C8 = ((c8 or {}).get("envs", {}) or {}).get(env, {})
        R = randall.get(env, {})

        def f(d, k):
            v = (d or {}).get(k)
            return v.get("fps") if isinstance(v, dict) else None

        gr = gpuR.get(env) or fallback(env, "cuda", 1) or fallback(env, "cuda", None)
        c1 = cpu1.get(env) or fallback(env, "cpu", 1)
        # end-to-end GPU at the renderer's certified batch, batched-indexing arm
        e2e = None
        P = pipe.get(env)
        if P and gr:
            b = gr["batch"] if gr["batch"] in P["cells"] else max(P["cells"])
            cell = P["cells"].get(b, {})
            e2e = {"batch": b, "batch_sampler": cell.get("batch_sampler"),
                   "collate_batched": cell.get("collate_batched"),
                   "no_loader": cell.get("no_loader"),
                   "default_collate": cell.get("default_collate"),
                   "source": P["source"], "host": P["host"]}
            e2e["fps"] = e2e["batch_sampler"] or e2e["collate_batched"] or e2e["no_loader"]

        row = {
            "env": env,
            "frame": f"{L['meta']['h']}x{L['meta']['w']}",
            "size": L["meta"].get("size"),
            "jpeg_bytes_per_frame": L["jpeg_bytes_per_frame"],
            "raw_bytes_per_frame": L["raw_bytes_per_frame"],
            "randall_frame": (f"{R['meta']['h']}x{R['meta']['w']}" if R else None),
            "lance_cpu_node": {
                "node": lcpu["node"], "cpu": lcpu["cpu"], "job": lcpu.get("slurm", {}).get("SLURM_JOB_ID"),
                "local_1proc": f(L, "local_1"), "local_n1": f(L, "local_n1"),
                "local_n4": f(L, "local_n4"), "local_n8": f(L, "local_n"),
                "shared_1proc": f(L, "shared_1"), "shared_n8": f(L, "shared_n"),
                "jobtmp_1proc": f(L, "jobtmp_1"), "jobtmp_n8": f(L, "jobtmp_n"),
                "local_read_frac": (L.get("local_1") or {}).get("read_frac"),
                "shared_read_frac": (L.get("shared_1") or {}).get("read_frac"),
                "jobtmp_read_frac": (L.get("jobtmp_1") or {}).get("read_frac"),
                "local_read_fps": (L.get("local_1") or {}).get("read_fps"),
                "output_location": "HOST memory, uint8 CHW",
            },
            "lance_instrumented": ({
                "node": a2c["node"], "job": a2c.get("slurm", {}).get("SLURM_JOB_ID"),
                **{k: {kk: vv for kk, vv in v.items()
                       if kk in ("fps", "cores_used_total", "cpu_efficiency_vs_workers",
                                 "decode_MBps", "read_MBps", "read_frac", "read_fps",
                                 "raw_out_MBps_host", "per_worker_cores_used", "output_location")}
                   for k, v in A2.items() if isinstance(v, dict) and v.get("fps")},
            } if A2 else None),
            "lance_gpu_node": ({
                "node": lgpu["node"], "gpu": lgpu.get("gpu"),
                "job": lgpu.get("slurm", {}).get("SLURM_JOB_ID"),
                "local_1proc": f(G, "local_1"), "local_n8": f(G, "local_n"),
                "nvjpeg": f(G, "local_gpu"),
            } if G else None),
            "jpeg_to_device": ({
                "node": j2g["node"], "gpu": j2g.get("gpu"),
                "job": j2g.get("slurm", {}).get("SLURM_JOB_ID"),
                "loader8_h2d": f(J, "loader8_h2d"),
                "loader8_h2d_unpinned": f(J, "loader8_h2d_unpinned"),
                "nvjpeg_h2d": f(J, "nvjpeg_h2d"),
                "h2d_only_pinned": f(J, "h2d_only_pinned"),
                "h2d_only_unpinned": f(J, "h2d_only_unpinned"),
                "source": a.jpeg_to_gpu,
            } if J else None),
            "ours_cpu": {
                "threads1": c1 and {"fps": c1["fps"], "batch": c1["batch"], "host": c1.get("host"),
                                    "source": c1["source"]},
                "threads8": (C8.get("threads8") or {}).get("fps"),
                "procs8": (C8.get("procs8") or {}).get("fps"),
                "best_8way": C8.get("best_8way"), "best_8way_arm": C8.get("best_8way_arm"),
                "threads1_same_job": (C8.get("threads1") or {}).get("fps"),
                "scaling_efficiency_best": C8.get("scaling_efficiency_best"),
                "scaling_efficiency_threads8": C8.get("scaling_efficiency_threads8"),
                "scaling_efficiency_procs8": C8.get("scaling_efficiency_procs8"),
                "cores_used_8way": (C8.get("procs8") or {}).get("cores_used_total"),
                "source_8way": a.cpu_8way, "node_8way": (c8 or {}).get("node"),
                "job_8way": (c8 or {}).get("slurm", {}).get("SLURM_JOB_ID"),
            },
            "ours_gpu": {
                "renderer_only": gr and {"fps": gr["fps"], "batch": gr["batch"], "host": gr.get("host"),
                                         "device_name": gr.get("device_name"), "source": gr["source"]},
                "end_to_end": e2e,
            },
            "frozen_one_core_q90": {
                "decode_q90_fps_per_core": (gr or {}).get("decode_q90_fps_per_core")
                                           or (c1 or {}).get("decode_q90_fps_per_core"),
                "decode_q90_jpeg_bytes": (gr or {}).get("decode_q90_jpeg_bytes")
                                         or (c1 or {}).get("decode_q90_jpeg_bytes"),
            },
            "quality_q90_over_q95": (((qp or {}).get("envs", {}) or {}).get(env, {})
                                     .get("q90", {}).get("rel_to_q95")),
        }

        ln = row["lance_cpu_node"]["local_n8"]
        l1 = row["lance_cpu_node"]["local_1proc"]
        percore = ln / 8 if ln else None
        b8 = row["ours_cpu"]["best_8way"]
        t1 = (row["ours_cpu"]["threads1"] or {}).get("fps")
        grf = (row["ours_gpu"]["renderer_only"] or {}).get("fps")
        e2ef = (e2e or {}).get("fps")
        jl8 = (row["jpeg_to_device"] or {}).get("loader8_h2d")
        jnv = (row["jpeg_to_device"] or {}).get("nvjpeg_h2d")
        dq90 = row["frozen_one_core_q90"]["decode_q90_fps_per_core"]

        row["ratios"] = {
            # (2) realistic stored-dataset training-pipeline break-even
            "cpu_pipeline_ratio": (b8 / ln) if b8 and ln else None,
            "cpu1_vs_lance_1proc": (t1 / l1) if t1 and l1 else None,
            "cpu1_vs_lance_percore": (t1 / percore) if t1 and percore else None,
            "gpu_renderer_only_vs_lance_n8_host": (grf / ln) if grf and ln else None,
            "gpu_pipeline_ratio": (e2ef / jl8) if e2ef and jl8 else None,
            "gpu_pipeline_ratio_nvjpeg": (e2ef / jnv) if e2ef and jnv else None,
            "gpu_rendereronly_vs_loader8_h2d": (grf / jl8) if grf and jl8 else None,
            # (1) frozen conservative one-core q90 break-even -- untouched
            "cores_cpu_frozen": (t1 / dq90) if t1 and dq90 else None,
            "cores_gpu_frozen": (grf / dq90) if grf and dq90 else None,
        }
        row["verdict"] = {
            "cpu_8way_beats_lance_8worker": bool(b8 and ln and b8 > ln),
            "cpu1_beats_lance_1proc": bool(t1 and l1 and t1 > l1),
            "cpu1_beats_lance_percore": bool(t1 and percore and t1 > percore),
            "gpu_e2e_beats_jpeg_loader8": bool(e2ef and jl8 and e2ef > jl8),
            "gpu_e2e_beats_nvjpeg": bool(e2ef and jnv and e2ef > jnv),
            "gpu_rendereronly_beats_lance_8worker_host": bool(grf and ln and grf > ln),
            "cores_cpu_frozen_ge_1": bool(row["ratios"]["cores_cpu_frozen"]
                                          and row["ratios"]["cores_cpu_frozen"] >= 1),
        }
        rows.append(row)

    out = {"what": "Step 6 deliverable: our renderer vs the production 8-worker JPEG-in-Lance "
                   "pipeline, at each config's certified resolution, on Oscar",
           "two_break_evens": {
               "frozen_conservative": "cores / cores_cpu: our fps / one core of q90 in-memory decode "
                                      "(certification.md 5.1, 6.3). Reproduced unchanged.",
               "realistic_pipeline": "our fps / the 8-worker Lance pipeline measured here at q95 and "
                                     "at the certified resolution; on GPU both sides end with frames "
                                     "on the device."},
           "runs": {"lance_cpu": {k: v for k, v in lcpu.items() if k != "envs"},
                    "lance_gpu": ({k: v for k, v in lgpu.items() if k != "envs"} if lgpu else None),
                    "lance_arms2_cpu": ({k: v for k, v in a2c.items() if k != "envs"} if a2c else None),
                    "lance_arms2_gpu": ({k: v for k, v in a2g.items() if k != "envs"} if a2g else None),
                    "jpeg_to_gpu": ({k: v for k, v in j2g.items() if k != "envs"} if j2g else None),
                    "cpu_8way": ({k: v for k, v in c8.items() if k != "envs"} if c8 else None)},
           "rows": rows}
    (root / a.out_json).parent.mkdir(parents=True, exist_ok=True)
    (root / a.out_json).write_text(json.dumps(out, indent=2))

    md = []
    md.append("# Step 6 — our renderer vs the production 8-worker JPEG-in-Lance pipeline (Oscar)\n")
    md.append("Every config at **its own certified resolution**. JPEG **q95**, chunk **256**, "
              "`examples/bench_lance.py` semantics (its timing functions imported verbatim).\n")
    md.append("Two break-even columns are kept separate throughout and neither replaces the other:\n"
              "**(1) frozen conservative** = our fps / one core of *in-memory q90* decode "
              "(`certification.md` §5.1/§6.3, taken from the committed shards' `decode` block); "
              "**(2) realistic pipeline** = our fps / the 8-worker stored-JPEG pipeline measured here.\n")

    md.append("\n## T1 — the stored baseline, per storage arm\n")
    md.append("Output of every CPU arm is **uint8 CHW tensors in HOST memory**. `cores used` is "
              "`sum(worker CPU seconds) / slowest-worker elapsed`, i.e. how many cores the 8 workers "
              "actually burned.\n")
    md.append("| config | frame | kB/f | tmpfs 1w | tmpfs 8w | NFS 1w | NFS 8w | GPFS 1w | GPFS 8w | "
              "cores used (8w) | read MB/s | read frac | 1-proc best-of-8 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        Lc = r["lance_cpu_node"]
        I = r["lance_instrumented"] or {}
        loc8 = I.get("local_n8") or {}
        md.append(f"| {r['env']} | {r['frame']} | {r['jpeg_bytes_per_frame']/1000:.1f} | "
                  f"{fmt(Lc['local_n1'])} | {fmt(Lc['local_n8'])} | {fmt(I.get('shared_n1',{}).get('fps'))} | "
                  f"{fmt(Lc['shared_n8'])} | {fmt(I.get('jobtmp_n1',{}).get('fps'))} | "
                  f"{fmt(Lc['jobtmp_n8'])} | {fmtr(loc8.get('cores_used_total'))} | "
                  f"{fmtr((I.get('local_1proc') or {}).get('read_MBps'))} | "
                  f"{pct(Lc['local_read_frac'])} | {fmt(Lc['local_1proc'])} |")

    md.append("\n## T2 — CPU: our renderer vs the stored pipeline, same node, same job\n")
    md.append("| config | lance 1 proc | lance 8w | lance 8w / 8 | ours 1 thread | ours 8 threads | "
              "ours 8 procs | best 8-way | scaling eff. | **cpu_pipeline_ratio** (8-way / lance 8w) | "
              "1thr / lance 1proc | 1thr / per-core |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        Lc, O, Ra = r["lance_cpu_node"], r["ours_cpu"], r["ratios"]
        md.append(f"| {r['env']} | {fmt(Lc['local_1proc'])} | {fmt(Lc['local_n8'])} | "
                  f"{fmt(Lc['local_n8']/8 if Lc['local_n8'] else None)} | "
                  f"{fmt((O['threads1'] or {}).get('fps'))} | {fmt(O['threads8'])} | "
                  f"{fmt(O['procs8'])} | {fmt(O['best_8way'])} | "
                  f"{fmtr(O['scaling_efficiency_best'])} | **{fmtr(Ra['cpu_pipeline_ratio'])}** | "
                  f"{fmtr(Ra['cpu1_vs_lance_1proc'])} | {fmtr(Ra['cpu1_vs_lance_percore'])} |")

    md.append("\n## T3 — GPU: both sides end with frames ON THE DEVICE\n")
    md.append("| config | JPEG→device, 8-worker loader + H2D | JPEG→device, nvJPEG | H2D alone (pinned) | "
              "ours renderer-only | ours end-to-end (states→device) | **gpu_pipeline_ratio** | "
              "vs nvJPEG | renderer-only / loader8+H2D |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        J = r["jpeg_to_device"] or {}
        Og, Ra = r["ours_gpu"], r["ratios"]
        md.append(f"| {r['env']} | {fmt(J.get('loader8_h2d'))} | {fmt(J.get('nvjpeg_h2d'))} | "
                  f"{fmt(J.get('h2d_only_pinned'))} | "
                  f"{fmt((Og['renderer_only'] or {}).get('fps'))} | "
                  f"{fmt((Og['end_to_end'] or {}).get('fps'))} | "
                  f"**{fmtr(Ra['gpu_pipeline_ratio'])}** | {fmtr(Ra['gpu_pipeline_ratio_nvjpeg'])} | "
                  f"{fmtr(Ra['gpu_rendereronly_vs_loader8_h2d'])} |")

    md.append("\n## T4 — the FROZEN conservative column, unchanged (one core of in-memory q90 decode)\n")
    md.append("This is `certification.md` §5.1/§6.3's `cores_cpu` / `cores`, reproduced from the "
              "committed shards' own `decode` block. It is **not** the pipeline ratio and must not be "
              "read as one.\n")
    md.append("| config | one-core q90 decode f/s | q90/q95 decode ratio (measured) | ours CPU 1 thread | "
              "**cores_cpu** (frozen) | ours GPU renderer-only | **cores** (frozen) |")
    md.append("|---|---|---|---|---|---|---|")
    for r in rows:
        F, O, Og, Ra = r["frozen_one_core_q90"], r["ours_cpu"], r["ours_gpu"], r["ratios"]
        md.append(f"| {r['env']} | {fmt(F['decode_q90_fps_per_core'])} | "
                  f"{fmtr(r['quality_q90_over_q95'])} | {fmt((O['threads1'] or {}).get('fps'))} | "
                  f"**{fmtr(Ra['cores_cpu_frozen'])}** | {fmt((Og['renderer_only'] or {}).get('fps'))} | "
                  f"**{fmtr(Ra['cores_gpu_frozen'])}** |")

    md.append("\n## T5 — per-cell provenance\n")
    md.append("| config | our CPU 1-thread cell | our CPU 8-way cell | our GPU renderer-only cell | "
              "our GPU end-to-end cell |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        t1 = r["ours_cpu"]["threads1"]
        gr = r["ours_gpu"]["renderer_only"]
        e2 = r["ours_gpu"]["end_to_end"]
        def s(x, extra=""):
            if not x:
                return "-"
            return (str(x.get("source", "")).split("examples/results/")[-1]
                    + f" (B={x.get('batch')}, {x.get('host')})" + extra)
        md.append(f"| {r['env']} | {s(t1)} | "
                  f"{(r['ours_cpu']['source_8way'] or '-').split('examples/results/')[-1]}"
                  f" ({r['ours_cpu']['node_8way']}, arm={r['ours_cpu']['best_8way_arm']}) | "
                  f"{s(gr)} | {s(e2)} |")

    def agg(path):
        v = []
        for r in rows:
            cur = r
            for k in path:
                cur = (cur or {}).get(k) if isinstance(cur, dict) else None
            if isinstance(cur, (int, float)):
                v.append(cur)
        return v

    md.append("\n## T6 — aggregates over the 36 rows\n")
    md.append("| quantity | median | min | max | n |")
    md.append("|---|---|---|---|---|")
    for label, path in [
            ("lance 1 proc (tmpfs)", ("lance_cpu_node", "local_1proc")),
            ("lance 8 workers (tmpfs)", ("lance_cpu_node", "local_n8")),
            ("lance 8 workers (NFS)", ("lance_cpu_node", "shared_n8")),
            ("lance 8 workers (GPFS)", ("lance_cpu_node", "jobtmp_n8")),
            ("JPEG→device, 8w loader + H2D", ("jpeg_to_device", "loader8_h2d")),
            ("JPEG→device, nvJPEG", ("jpeg_to_device", "nvjpeg_h2d")),
            ("ours CPU 1 thread", ("ours_cpu", "threads1", "fps")),
            ("ours CPU best 8-way", ("ours_cpu", "best_8way")),
            ("ours CPU 8-way scaling efficiency", ("ours_cpu", "scaling_efficiency_best")),
            ("ours GPU renderer-only", ("ours_gpu", "renderer_only", "fps")),
            ("ours GPU end-to-end", ("ours_gpu", "end_to_end", "fps")),
            ("cpu_pipeline_ratio", ("ratios", "cpu_pipeline_ratio")),
            ("gpu_pipeline_ratio", ("ratios", "gpu_pipeline_ratio")),
            ("gpu_pipeline_ratio (nvJPEG)", ("ratios", "gpu_pipeline_ratio_nvjpeg")),
            ("cores_cpu (frozen, one core q90)", ("ratios", "cores_cpu_frozen")),
            ("cores (frozen, GPU vs one core q90)", ("ratios", "cores_gpu_frozen"))]:
        v = agg(path)
        if v:
            md.append(f"| {label} | {st.median(v):,.2f} | {min(v):,.2f} | {max(v):,.2f} | {len(v)} |")

    md.append("\n## T7 — who wins against the production pipeline\n")
    for key, title in [
            ("cpu_8way_beats_lance_8worker", "CPU: our 8-way renderer beats the 8-worker Lance pipeline"),
            ("cpu1_beats_lance_percore", "CPU: our 1 thread beats ONE CORE's share of the 8-worker pipeline"),
            ("cpu1_beats_lance_1proc", "CPU: our 1 thread beats 1-process Lance"),
            ("gpu_e2e_beats_jpeg_loader8", "GPU: our end-to-end state→device beats 8-worker JPEG→device"),
            ("gpu_e2e_beats_nvjpeg", "GPU: our end-to-end beats nvJPEG→device"),
            ("gpu_rendereronly_beats_lance_8worker_host",
             "GPU (renderer-only vs host-memory Lance — the ASYMMETRIC comparison, shown for continuity)"),
            ("cores_cpu_frozen_ge_1", "FROZEN column: cores_cpu ≥ 1 (the Step-2 G4-on-CPU bar)")]:
        win = [r["env"] for r in rows if r["verdict"][key]]
        lose = [r["env"] for r in rows if not r["verdict"][key]]
        md.append(f"\n### {title}\n")
        md.append(f"**{len(win)}/{len(rows)} win:** " + (", ".join(win) or "none") + "\n")
        md.append(f"**lose:** " + (", ".join(lose) or "none") + "\n")

    (root / a.out_md).write_text("\n".join(md) + "\n")
    print(f"wrote {a.out_json} and {a.out_md} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
