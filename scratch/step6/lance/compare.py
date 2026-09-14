"""Join our renderer's certified fps to the Oscar JPEG-in-Lance baseline, per registered_default row.

Selection policy, applied per config and recorded per cell:

* **GPU cell** -- Oscar RTX 3090 only. Prefer a Step-6 *after* shard; fall back to Step 3's
  ``3090_full.json`` / ``3090_fetch311.json`` for rows Step 6 did not re-measure. Best batch from the
  mandatory certification list (certification.md s6.2/s6.3).
* **CPU 1-thread cell** -- Oscar x86 ``cascade`` batch node only (never the M5 laptop, which is not a
  certification host, and never ``oscar_x86_mjwarp_1thread.json``: that shard was taken on a GPU node
  and its 70-80 kf/s "cpu" rows are MJWarp on the device, not a CPU backend). Prefer the
  same-job supplemental run, then a Step-6 after shard, then ``oscar_x86_1thread.json``.
* **CPU 8-thread cell** -- the same-job supplemental run only. Reported beside the 1-thread number
  and never substituted for it (certification.md s5.3).

Ratios produced per row:
    gpu_vs_local_n      our GPU fps / the 8-worker Lance fps       -> "GPUs worth of an 8-worker loader"
    gpu_cores_of_8      our GPU fps / (local_n fps / 8)            -> cores of 8-worker Lance replaced
    cpu1_vs_local_1     our 1-thread CPU fps / 1-process Lance fps
    cpu1_vs_percore     our 1-thread CPU fps / (local_n fps / 8)   -> the honest per-core comparison
    cpu8_vs_local_n     our 8-thread CPU fps / the 8-worker Lance fps
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

MANDATORY = [1, 16, 64, 256, 1024, 4096]
BAD_CPU_SHARDS = ("oscar_x86_mjwarp_1thread",)


def load_lance(paths):
    out = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        out[str(p)] = d
    return out


def pick(rows, device, prefer_substrings, exclude_substrings=(), require_host=None, threads=None):
    cands = [r for r in rows if r["device"] == device]
    if threads is not None:
        cands = [r for r in cands if r.get("threads") == threads]
    if require_host:
        cands = [r for r in cands if r.get("host") and any(h in r["host"] for h in require_host)]
    cands = [r for r in cands if not any(x in r["source"] for x in exclude_substrings)]
    if not cands:
        return None
    for sub in prefer_substrings:
        hit = [r for r in cands if sub in r["source"]]
        if hit:
            return max(hit, key=lambda r: r["fps"])
    return max(cands, key=lambda r: r["fps"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--renderer-rows", default="examples/results/step6/lance/renderer_rows.json")
    ap.add_argument("--lance-cpu", required=True)
    ap.add_argument("--lance-gpu", default=None)
    ap.add_argument("--randall", default="examples/results/storage/lance.json")
    ap.add_argument("--cpu1-supplemental", default=None)
    ap.add_argument("--cpu8-supplemental", default=None)
    ap.add_argument("--out-json", default="examples/results/step6/lance/comparison.json")
    ap.add_argument("--out-md", default="examples/results/step6/lance/comparison.md")
    a = ap.parse_args()
    root = Path(a.root).resolve()

    rr = json.loads((root / a.renderer_rows).read_text())["rows"]
    by_env: dict[str, list] = {}
    for r in rr:
        by_env.setdefault(r["env"], []).append(r)

    # the same-job supplemental CPU shards, folded into the same row shape
    def fold(path, threads):
        if not path:
            return
        d = json.loads(Path(path).read_text())
        prov = d.get("provenance", {})
        for e in d.get("envs", []):
            cells = [c for c in e.get("cells", []) if c.get("fps") and c.get("batch") in MANDATORY]
            if not cells:
                continue
            best = max(cells, key=lambda c: c["fps"])
            by_env.setdefault(e["env"], []).append({
                "env": e["env"], "device": "cpu", "size": e.get("size"), "shape": e.get("shape"),
                "supersample": e.get("supersample"), "backend": e.get("backend"),
                "tier": e.get("fidelity_tier"), "fps": best["fps"], "batch": best["batch"],
                "all_cells": {c["batch"]: c["fps"] for c in e["cells"] if c.get("fps")},
                "threads": threads, "host": prov.get("host"), "device_name": None,
                "cpu": prov.get("cpu"), "git_commit": prov.get("git_commit"),
                "slurm_job_id": prov.get("slurm_job_id"),
                "decode_q90_fps_per_core": (e.get("decode") or {}).get("decode_fps_per_core"),
                "decode_jpeg_bytes_q90": (e.get("decode") or {}).get("jpeg_bytes"),
                "official_cpu_fps": (e.get("official_cpu") or {}).get("fps"),
                "source": str(path) + "#same_job_supplemental",
            })

    fold(a.cpu1_supplemental, 1)
    fold(a.cpu8_supplemental, 8)

    lcpu = json.loads(Path(a.lance_cpu).read_text())
    lgpu = json.loads(Path(a.lance_gpu).read_text()) if a.lance_gpu else None
    randall = json.loads((root / a.randall).read_text())["envs"]

    rows = []
    for env in sorted(lcpu["envs"]):
        L = lcpu["envs"][env]
        G = (lgpu or {}).get("envs", {}).get(env, {})
        R = randall.get(env, {})
        cands = by_env.get(env, [])
        gpu = pick(cands, "cuda", ["/step6/", "3090_fetch311", "3090_full"],
                   exclude_substrings=("before", "superseded", "rejected", "control"),
                   require_host=("gpu2", "oscar"))
        cpu1 = pick(cands, "cpu", ["#same_job_supplemental", "/step6/", "oscar_x86_1thread"],
                    exclude_substrings=("before", "superseded", "rejected", "control",
                                        "m5_", *BAD_CPU_SHARDS),
                    require_host=("node", "oscar"), threads=1)
        cpu8 = pick(cands, "cpu", ["#same_job_supplemental"],
                    exclude_substrings=("before", "superseded", "rejected", "control", "m5_",
                                        *BAD_CPU_SHARDS),
                    require_host=("node", "oscar"), threads=8)

        def f(d, k):
            v = (d or {}).get(k)
            return v.get("fps") if isinstance(v, dict) else None

        row = {
            "env": env,
            "frame": f"{L['meta']['h']}x{L['meta']['w']}",
            "jpeg_bytes_per_frame": L["jpeg_bytes_per_frame"],
            "randall_frame": (f"{R['meta']['h']}x{R['meta']['w']}" if R else None),
            "randall_jpeg_bytes_per_frame": R.get("jpeg_bytes_per_frame") if R else None,
            "lance": {
                "local_1": f(L, "local_1"), "local_n8": f(L, "local_n"),
                "local_n1": f(L, "local_n1"), "local_n4": f(L, "local_n4"),
                "shared_1": f(L, "shared_1"), "shared_n8": f(L, "shared_n"),
                "jobtmp_1": f(L, "jobtmp_1"), "jobtmp_n8": f(L, "jobtmp_n"),
                "local_read_frac": (L.get("local_1") or {}).get("read_frac"),
                "shared_read_frac": (L.get("shared_1") or {}).get("read_frac"),
                "jobtmp_read_frac": (L.get("jobtmp_1") or {}).get("read_frac"),
                "gpu_nvjpeg": f(G, "local_gpu"),
                "gpu_node_local_1": f(G, "local_1"), "gpu_node_local_n8": f(G, "local_n"),
            },
            "randall_lance": {
                "local_1": f(R, "local_1"), "local_n8": f(R, "local_n"),
                "shared_1": f(R, "shared_1"), "local_gpu": f(R, "local_gpu"),
            } if R else None,
            "ours": {
                "gpu": gpu and {"fps": gpu["fps"], "batch": gpu["batch"], "host": gpu["host"],
                                "shape": gpu["shape"], "size": gpu["size"], "source": gpu["source"]},
                "cpu1": cpu1 and {"fps": cpu1["fps"], "batch": cpu1["batch"], "host": cpu1["host"],
                                  "shape": cpu1["shape"], "source": cpu1["source"]},
                "cpu8": cpu8 and {"fps": cpu8["fps"], "batch": cpu8["batch"], "host": cpu8["host"],
                                  "shape": cpu8["shape"], "source": cpu8["source"]},
                "decode_q90_fps_per_core": next((r["decode_q90_fps_per_core"] for r in cands
                                                 if r.get("decode_q90_fps_per_core")), None),
            },
        }
        ln = row["lance"]["local_n8"]
        l1 = row["lance"]["local_1"]
        percore = (ln / 8) if ln else None
        g = row["ours"]["gpu"]["fps"] if row["ours"]["gpu"] else None
        c1 = row["ours"]["cpu1"]["fps"] if row["ours"]["cpu1"] else None
        c8 = row["ours"]["cpu8"]["fps"] if row["ours"]["cpu8"] else None
        row["ratios"] = {
            "gpu_vs_local_n": (g / ln) if g and ln else None,
            "gpu_cores_of_8": (g / percore) if g and percore else None,
            "cpu1_vs_local_1": (c1 / l1) if c1 and l1 else None,
            "cpu1_vs_percore": (c1 / percore) if c1 and percore else None,
            "cpu8_vs_local_n": (c8 / ln) if c8 and ln else None,
            "gpu_vs_randall_local_n": (g / R["local_n"]["fps"]) if g and R and R.get("local_n", {}).get("fps") else None,
            "nvjpeg_vs_our_gpu": (row["lance"]["gpu_nvjpeg"] / g) if g and row["lance"]["gpu_nvjpeg"] else None,
        }
        row["verdict"] = {
            "gpu_beats_8worker_lance": bool(g and ln and g > ln),
            "cpu1_beats_1proc_lance": bool(c1 and l1 and c1 > l1),
            "cpu1_beats_percore_lance": bool(c1 and percore and c1 > percore),
            "cpu8_beats_8worker_lance": bool(c8 and ln and c8 > ln),
        }
        rows.append(row)

    out = {"what": "Step 6: our renderer vs the JPEG-in-Lance 8-worker baseline, on Oscar",
           "lance_cpu_run": {k: v for k, v in lcpu.items() if k != "envs"},
           "lance_gpu_run": ({k: v for k, v in lgpu.items() if k != "envs"} if lgpu else None),
           "rows": rows}
    (root / a.out_json).parent.mkdir(parents=True, exist_ok=True)
    (root / a.out_json).write_text(json.dumps(out, indent=2))

    def fmt(x, w=9, c=","):
        return f"{x:>{w},.0f}" if isinstance(x, (int, float)) else " " * (w - 1) + "-"

    def fmtr(x, w=7):
        return f"{x:>{w},.2f}" if isinstance(x, (int, float)) else " " * (w - 1) + "-"

    md = ["# Our renderer vs the 8-worker JPEG-in-Lance baseline (Oscar)", "",
          "All Lance columns are this repo's `examples/bench_lance.py`, unmodified, on Oscar.",
          "`local_n` is 8 spawn workers with 1 torch thread each, aggregated from the workers' own",
          "elapsed times. `/8` is that divided by eight, i.e. one core's share.", "",
          "| config | frame | kB/f | lance local_1 | lance local_n(8) | lance/core | our GPU | GPU/local_n | GPU cores-of-8 | our CPU 1thr | CPU1/local_1 | CPU1/core | our CPU 8thr | CPU8/local_n |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ln = r["lance"]["local_n8"]
        md.append("| {e} | {fr} | {kb} | {l1} | {ln} | {pc} | {g} | {gr} | {gc} | {c1} | {c1r} | {c1p} | {c8} | {c8r} |".format(
            e=r["env"], fr=r["frame"], kb=f"{r['jpeg_bytes_per_frame']/1000:.1f}",
            l1=fmt(r["lance"]["local_1"]), ln=fmt(ln), pc=fmt(ln / 8 if ln else None),
            g=fmt(r["ours"]["gpu"]["fps"] if r["ours"]["gpu"] else None),
            gr=fmtr(r["ratios"]["gpu_vs_local_n"]), gc=fmtr(r["ratios"]["gpu_cores_of_8"]),
            c1=fmt(r["ours"]["cpu1"]["fps"] if r["ours"]["cpu1"] else None),
            c1r=fmtr(r["ratios"]["cpu1_vs_local_1"]), c1p=fmtr(r["ratios"]["cpu1_vs_percore"]),
            c8=fmt(r["ours"]["cpu8"]["fps"] if r["ours"]["cpu8"] else None),
            c8r=fmtr(r["ratios"]["cpu8_vs_local_n"])))
    md += ["", "## Per-cell provenance", "",
           "| config | GPU cell source | CPU 1-thread cell source | CPU 8-thread cell source |",
           "|---|---|---|---|"]
    for r in rows:
        def src(k):
            v = r["ours"][k]
            return (v["source"].split("examples/results/")[-1] + f" (B={v['batch']}, {v['host']})") if v else "-"
        md.append(f"| {r['env']} | {src('gpu')} | {src('cpu1')} | {src('cpu8')} |")

    def med(key, path):
        vals = []
        for r in rows:
            cur = r
            for p in path:
                cur = (cur or {}).get(p)
            if isinstance(cur, (int, float)):
                vals.append(cur)
        return (st.median(vals), min(vals), max(vals), len(vals)) if vals else None

    md += ["", "## Aggregates", "", "| quantity | median | min | max | n |", "|---|---|---|---|---|"]
    for label, path in [("lance local_1", ("lance", "local_1")),
                        ("lance local_n (8 workers)", ("lance", "local_n8")),
                        ("lance shared_1 (NFS)", ("lance", "shared_1")),
                        ("lance jobtmp_1 (GPFS)", ("lance", "jobtmp_1")),
                        ("lance nvJPEG (3090)", ("lance", "gpu_nvjpeg")),
                        ("our GPU fps", ("ours", "gpu", "fps")),
                        ("our CPU 1-thread fps", ("ours", "cpu1", "fps")),
                        ("our CPU 8-thread fps", ("ours", "cpu8", "fps")),
                        ("GPU / local_n", ("ratios", "gpu_vs_local_n")),
                        ("GPU cores-of-8 replaced", ("ratios", "gpu_cores_of_8")),
                        ("CPU1 / local_1", ("ratios", "cpu1_vs_local_1")),
                        ("CPU1 / per-core", ("ratios", "cpu1_vs_percore")),
                        ("CPU8 / local_n", ("ratios", "cpu8_vs_local_n"))]:
        m = med(label, path)
        if m:
            md.append(f"| {label} | {m[0]:,.2f} | {m[1]:,.2f} | {m[2]:,.2f} | {m[3]} |")

    for key, title in [("gpu_beats_8worker_lance", "Rows whose GPU renderer beats the 8-worker Lance baseline"),
                       ("cpu1_beats_1proc_lance", "Rows whose 1-thread CPU renderer beats 1-process Lance"),
                       ("cpu1_beats_percore_lance", "Rows whose 1-thread CPU renderer beats one core of the 8-worker Lance baseline"),
                       ("cpu8_beats_8worker_lance", "Rows whose 8-thread CPU renderer beats the 8-worker Lance baseline")]:
        win = [r["env"] for r in rows if r["verdict"][key]]
        lose = [r["env"] for r in rows if not r["verdict"][key]]
        md += ["", f"### {title}", "", f"**{len(win)}/{len(rows)}**: " + (", ".join(win) or "none"),
               "", f"Not: " + (", ".join(lose) or "none")]

    (root / a.out_md).write_text("\n".join(md) + "\n")
    print(f"wrote {a.out_json} and {a.out_md} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
