"""Headline table of a compare.sh results folder: per workload, the geomean over envs of each env's best multiple.

usage: python3 benchmarks/paper/headline.py OUT [--batches 1024:2048] [--write]
OUT holds one folder per env (run.sh output: offline/, online/). The multiple of a cell is ours / source on the same
machine: the source at the same lane count, else the best source over the lane counts that were timed. Offline cells use
the best source over 1 and N processes. GPU frames use the source's GPU renderer where it passed the frame check, else
the CPU source. --batches keeps only the cells of ours at those batches and lane counts. Works on a partial OUT.
"""

import argparse
import glob
import json
import math
import os
import re
import statistics
from collections import defaultdict

GROUPS = [
    ("steps", "cpu"),
    ("steps", "gpu"),
    ("framesteps", "cpu"),
    ("framesteps", "gpu"),
    ("offline", "cpu"),
    ("offline", "gpu"),
]
NAMES = {"steps": "steps", "framesteps": "frames+steps", "offline": "offline"}


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def offline(folder, env, name):
    e = (load(f"{folder}/offline/{env}/{name}.json") or {}).get("envs", {}).get(env, {})
    pts = e.get("points") or ([e] if e.get("status") == "ok" else [])
    return {p.get("batch"): p.get("fps") for p in pts if p.get("status") == "ok" and p.get("fps")}


def online(folder, env, prefix, max_l=None):
    """{(L, T): rate} of the ok arms in online/ENV/PREFIX-L*.json (whole episodes only)."""
    out = {}
    for f in glob.glob(f"{folder}/online/{env}/{prefix}-L*.json"):
        m = re.search(r"-L(\d+)\.json$", f)
        d = load(f)
        if f.endswith("-s8.json") or not m or not d or (max_l and int(m.group(1)) > max_l):
            continue
        for k, a in d.get("arms", {}).items():
            if a.get("status", "ok") == "ok" and a.get("steps_per_s") and a.get("steps_per_episode") != 8:
                t = re.search(r"-T(\d+)$", k)
                out[(int(m.group(1)), int(t.group(1)) if t else 1)] = a["steps_per_s"]
    return out


def cells(folder, env):
    """[(regime, device, B, T, ours, source)] of one env folder."""
    out = []
    for dev in ("cpu", "gpu"):
        ours = offline(folder, env, f"snv-{dev}")
        srcs = list(offline(folder, env, f"src-{dev}-1p").values()) + list(
            offline(folder, env, f"src-{dev}-8p").values()
        )
        if not srcs and dev == "gpu":
            srcs = list(offline(folder, env, "src-cpu-1p").values()) + list(
                offline(folder, env, "src-cpu-8p").values()
            )
        if srcs:
            out += [("offline", dev, b, "", v, max(srcs)) for b, v in ours.items()]
    step_src = {L: v for (L, _), v in online(folder, env, "src-step").items()}
    rcpu = {L: v for (L, _), v in online(folder, env, "src-render-cpu").items()}
    rgpu = {L: v for (L, _), v in online(folder, env, "src-render-gpu", max_l=256).items()}
    for reg, dev, prefix, src in (
        ("steps", "cpu", "snv-step-cpu", step_src),
        ("steps", "gpu", "snv-step-gpu", step_src),
        ("framesteps", "cpu", "snv-render-cpu", rcpu),
        ("framesteps", "gpu", "snv-render-gpu", rgpu or rcpu),
    ):
        if not src:
            continue
        for (L, T), v in online(folder, env, prefix).items():
            out.append((reg, dev, L, T, v, src.get(L, max(src.values()))))
    return out


def gm(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def hardware(folder):
    """(key, line) of the machine an env folder ran on, from run.json (hardware) or hardware.json."""
    run = load(f"{folder}/run.json") or {}
    h = run.get("hardware") or load(f"{folder}/hardware.json") or {}
    gpu = h.get("gpu") or (run.get("gpu") or "?").split(",")[0]
    cpu = h.get("cpu") or "?"
    line = (
        f"{gpu} ({h.get('gpu_count', '?')} on the machine), driver {h.get('driver', '?')}, CUDA {h.get('cuda', '?')}; "
        f"{cpu}, 8 pinned physical cores per pass, CPU quota {h.get('cpu_quota') or 'none'}, RAM {h.get('ram_gib', '?')} GiB; "
        f"{h.get('os', '?')}; Python {h.get('python', '?')}, torch {h.get('torch', '?')}, "
        f"stable-envs {h.get('stable_envs', '?')} ({(h.get('commit') or run.get('commit') or '?')[:8]})"
    )
    return (gpu, cpu), line


def table(out, envs, keep, label):
    best, by_b, n_cells, below = {}, defaultdict(dict), 0, 0
    for e in envs:
        for reg, dev, B, T, ours, src in cells(f"{out}/{e}", e):
            if keep and B not in keep:
                continue
            x = ours / src
            n_cells += 1
            below += x < 1
            g = (reg, dev)
            if x > best.get((e, g), (0,))[0]:
                best[(e, g)] = (x, B, T)
            by_b[g][(e, B)] = max(by_b[g].get((e, B), 0), x)
    done = [e for e in envs if os.path.exists(f"{out}/{e}/done.json")]
    L = [
        f"Hardware: {label}",
        "",
        f"Envs: {len(envs)} with results, {len(done)} finished; cells: {n_cells}, below 1x: {below}"
        + (f"; batches and lane counts {':'.join(map(str, sorted(keep)))}" if keep else ""),
        "",
        "| Workload | Geomean of best cell | N envs | Median | Min env (x) | Envs best < 1x |",
        "|---|---|---|---|---|---|",
    ]
    for g in GROUPS:
        v = {e: x for (e, gg), (x, _, _) in best.items() if gg == g}
        if not v:
            continue
        mn = min(v, key=v.get)
        L.append(
            f"| {NAMES[g[0]]} {g[1].upper()} | {gm(list(v.values())):.2f} | {len(v)} | {statistics.median(v.values()):.2f} "
            f"| {mn} ({v[mn]:.2f}) | {sum(x < 1 for x in v.values())} |"
        )
    bs = sorted({B for g in by_b for (_, B) in by_b[g]})
    L += [
        "",
        "Geomean over envs of the best cell at each batch or lane count (N envs):",
        "",
        "| Workload | " + " | ".join(f"B={b}" for b in bs) + " |",
        "|---|" + "---|" * len(bs),
    ]
    for g in GROUPS:
        if not by_b[g]:
            continue
        row = []
        for b in bs:
            xs = [x for (e, B), x in by_b[g].items() if B == b]
            row.append(f"{gm(xs):.2f} ({len(xs)})" if xs else "-")
        L.append(f"| {NAMES[g[0]]} {g[1].upper()} | " + " | ".join(row) + " |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--batches", default="")
    ap.add_argument("--write", action="store_true", help="also write OUT/HEADLINE.md")
    a = ap.parse_args()
    keep = {int(b) for b in a.batches.split(":") if b}
    envs = sorted(
        d for d in os.listdir(a.out) if os.path.isdir(f"{a.out}/{d}/online") or os.path.isdir(f"{a.out}/{d}/offline")
    )
    groups = defaultdict(list)
    lines = {}
    for e in envs:
        k, line = hardware(f"{a.out}/{e}")
        groups[k].append(e)
        lines.setdefault(k, line)
    L = [
        "# stable-envs against the source libraries",
        "",
        "Multiple = ours / source on the same machine; best cell = the largest multiple of an env over batches and T.",
        "One table per GPU and CPU model." if len(groups) > 1 else "",
    ]
    for k in sorted(groups):
        L += ["", f"## {k[0]}, {k[1]}" if len(groups) > 1 else ""] + table(a.out, groups[k], keep, lines[k])
    text = "\n".join(L) + "\n"
    print(text)
    if a.write:
        with open(f"{a.out}/HEADLINE.md", "w") as f:
            f.write(text)
        print(f"{a.out}/HEADLINE.md")


if __name__ == "__main__":
    main()
