"""The cells of a compare.sh run and an estimate of its time, from the rt4 L40S rates (rt4_l40s.tar.gz).

usage: python3 benchmarks/paper/plan.py --envs E E ... --sizes 1:16:256:1024 [--missing-only] [--out OUT] [--jobs N]
Lists every cell file that the run times (with --missing-only: only those that rt4 does not have as an ok cell, and
with --out: not those that OUT already has), with an estimate in seconds, then the total. The estimate uses the rt4
rate of the same cell, or of the nearest lane count that rt4 timed, and the harness's timing rules: process start,
2 s warm-up, three 3 s windows that each hold whole 64-step episodes, the two T values of ours, the source frame
check, and the 20-minute limit per cell. Rates on another GPU or CPU differ (the H100 source GPU frames are slower).
"""

import argparse
import glob
import json
import os
import re
import tarfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LIMIT = 20 * 60
LENGTH = 64


def rt4_rates():
    """{env: {prefix: {L: rate}}} and {env: set of ok cell files} from the rt4 archive."""
    rates, have = defaultdict(lambda: defaultdict(dict)), defaultdict(set)
    with tarfile.open(f"{HERE}/rt4_l40s.tar.gz") as t:
        for m in t.getmembers():
            p = m.name.split("/")
            if len(p) != 5 or p[2] != "online" or not p[4].endswith(".json"):
                continue
            env, name = p[1], p[4]
            mm = re.match(r"(.*)-L(\d+)\.json$", name)
            if not mm:
                continue
            have[env].add(name)
            d = json.load(t.extractfile(m))
            rs = [a.get("steps_per_s") for a in d.get("arms", {}).values() if a.get("steps_per_s")]
            if rs:
                rates[env][mm.group(1)][int(mm.group(2))] = max(rs)
    return rates, have


def online_s(rate, lanes, ours, gpu, check):
    if not rate:
        return LIMIT
    unit = lanes * LENGTH / rate
    window = max(3.0, unit)
    windows = window if window >= 60 else min(3 * window, max(60.0, window))
    t = max(2.0, unit) + windows
    t = (2 * t if ours else t) + (40 if gpu else 20)
    if check:
        t += 8 * LENGTH / max(rate / max(lanes, 1), 1.0) + 20
    return min(t, LIMIT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", required=True)
    ap.add_argument("--sizes", default="1:16:256:1024")
    ap.add_argument("--missing-only", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(":")]
    gs = json.load(open(f"{HERE}/fleet/costs.json")).get("gpu_source", {})
    rates, have = rt4_rates()
    total, n = 0.0, 0
    for env in a.envs:
        g = gs.get(env, "")
        r = rates[env]

        def near(prefix, L, fallback=None):
            d = r.get(prefix) or r.get(fallback or "", {})
            return d[min(d, key=lambda x: abs(x - L))] if d else None

        cells = []
        # Offline: ours over the batches and the frame MSE on each device; the source in 1 and 8 processes.
        for dev in ("cpu", "gpu"):
            cells += [(f"offline/snv-{dev}.json", 40 + 13 * len(sizes)), (f"offline/mse-{dev}.json", 60)]
        cells += [("offline/src-cpu-1p.json", 35), ("offline/src-cpu-8p.json", 35)]
        if g == "mujoco-egl":
            cells += [("offline/src-gpu-1p.json", 45), ("offline/src-gpu-8p.json", 45)]
        for L in sizes:
            cells += [
                (f"src-step-L{L}.json", online_s(near("src-step", L), L, False, False, False)),
                (f"snv-step-cpu-L{L}.json", online_s(near("snv-step-cpu", L), L, True, False, False)),
                (f"snv-step-gpu-L{L}.json", online_s(near("snv-step-gpu", L), L, True, True, False)),
                (f"src-render-cpu-L{L}.json", online_s(near("src-render-cpu", L), L, False, False, True)),
                (f"snv-render-cpu-L{L}.json", online_s(near("snv-render-cpu", L), L, True, False, False)),
                (f"snv-render-gpu-L{L}.json", online_s(near("snv-render-gpu", L), L, True, True, False)),
            ]
            if g:
                rate = near("src-render-gpu", L, "src-render-cpu")
                cells.append((f"src-render-gpu-L{L}.json", online_s(rate, L, False, True, True)))
        sub = 0.0
        for name, s in cells:
            if a.missing_only and name in have[env]:
                continue
            path = f"offline/{env}/{name.split('/')[1]}" if name.startswith("offline") else f"online/{env}/{name}"
            if a.out and (os.path.exists(f"{a.out}/{env}/{path}") or glob.glob(f"{a.out}/{env}/pass-*/{path}")):
                continue
            side = "ours" if os.path.basename(name).startswith(("snv-", "mse-")) else "source"
            flag = "  (may reach the 20-minute limit)" if s >= LIMIT else ""
            print(f"  {env:22s} {side:6s} {path:58s} ~{s / 60:5.1f} min{flag}")
            sub += s
            n += 1
        total += sub
        print(f"  {env}: ~{sub / 3600:.2f} GPU-hours")
    jobs = max(1, min(a.jobs, len(a.envs)))
    print(
        f"estimate: {n} cells, ~{total / 3600:.1f} GPU-hours; with {jobs} jobs at a time ~{total / 3600 / jobs:.1f} h of wall time"
        " (one env per job, so the longest env sets a lower bound); from rt4 L40S rates"
    )


if __name__ == "__main__":
    main()
