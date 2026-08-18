"""One table per question, from whatever has finished.

``--what parity``   the 41-draw survey: does each cell still pass the three assertions at ss=1, and
                    with how much margin on the one that binds.
``--what fps``      the single-process throughput column, ss=1 against ss=2 against the bar.
``--what corr``     the correctors: every metric the parity test reads, held out, plus the tail.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

LOGS = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/msaanet"
BASELINE = 16889.0


def parity(size_kind: str = "default"):
    rows = []
    for p in sorted(glob.glob(os.path.join(LOGS, "parity_draws", "*.json"))):
        d = json.load(open(p))
        size = str(d["default_size"]) if size_kind == "default" else size_kind
        if size not in d["sizes"]:
            continue
        f = d["floors"]
        r = {"cell": d["cell"], "size": int(size), "floor_interior": f["interior"],
             "floor_exact": f["all_exact"]}
        for ss in ("1", "2"):
            s = d["sizes"][size]["per_ss"].get(ss)
            if s is None:
                continue
            r[f"ss{ss}"] = s
        rows.append(r)
    hdr = (f"{'cell':30s} {'px':>4s} | {'ss=1 fail/41':>12s} {'exact':>7s} {'mean':>6s} "
           f"{'maxabs':>6s} {'int-exact':>12s} | {'ss=2 fail/41':>12s} {'exact':>7s} {'mean':>6s} "
           f"{'maxabs':>6s} {'int-exact':>12s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cells = []
        for ss in ("1", "2"):
            s = r.get(f"ss{ss}")
            if s is None:
                cells.append(f"{'-':>12s} {'-':>7s} {'-':>6s} {'-':>6s} {'-':>12s}")
                continue
            ie = s["worst_interior_exact"]
            cells.append(f"{s['n_fail_any']:>12d} {s['worst_exact_frac']:>7.4f} "
                         f"{s['worst_mean_abs']:>6.3f} {s['worst_max_abs']:>6d} "
                         f"{('n/a' if ie is None else f'{ie:.7f}'):>12s}")
        print(f"{r['cell']:30s} {r['size']:>4d} | {cells[0]} | {cells[1]}")
    print("\nfloors: interior >= parity_interior_min (per class), exact >= 0.90, mean <= 4.0.  "
          "'worst' is over 41 seeded draws of 64 states.")


def fps(path=os.path.join(LOGS, "fps.json")):
    d = json.load(open(path))
    print(f"host {d['host']} {d['device']} batch {d['batch']} reps {d['reps']} "
          f"loadavg {d.get('loadavg_start')} -> {d.get('loadavg_end')}\n")
    print(f"{'corrector':24s} {'84px us':>8s} {'240px us':>9s} {'256px us':>9s} {'params':>8s}")
    names = sorted({k.split("@")[0] for k in d["corrector"]})
    for n in names:
        g = lambda s: d["corrector"].get(f"{n}@{s}", {}).get("us_per_frame")  # noqa: E731
        f8, f240, f256 = g(84), g(240), g(256)
        p = d["corrector"].get(f"{n}@256", {}).get("params", 0)
        print(f"{n:24s} {('-' if f8 is None else f'{f8:.2f}'):>8s} "
              f"{('-' if f240 is None else f'{f240:.2f}'):>9s} "
              f"{('-' if f256 is None else f'{f256:.2f}'):>9s} {p/1e3:>7.0f}k")
    for size_kind in ("default", "256"):
        print(f"\n--- render throughput, {size_kind} size ---")
        print(f"{'cell':30s} {'px':>4s} {'ss=2 f/s':>10s} {'ss=1 f/s':>10s} {'x':>5s} "
              f"{'ss2 xbar':>9s} {'ss1 xbar':>9s} {'dev share ss1':>13s} {'dev share ss2':>13s} "
              f"{'ss1+corr f/s':>13s}")
        for cell, row in d["cells"].items():
            size = str(row["default_size"]) if size_kind == "default" else size_kind
            e = row["sizes"].get(size)
            if e is None:
                continue
            a, b = e["render_ss1"], e["render_ss2"]
            ee = e.get("end_to_end_ss1_plus_corrector", {})
            # The cheapest corrector that was timed on this cell: the row a consumer would ship.
            best = min(ee.values(), key=lambda v: v["us_per_frame"]) if ee else {"fps": 0.0}
            ds = lambda x: "-" if x.get("device_kernel_share") is None else f"{x['device_kernel_share']:.3f}"  # noqa: E731
            print(f"{cell:30s} {size:>4s} {b['fps']:>10.0f} {a['fps']:>10.0f} "
                  f"{a['fps']/b['fps']:>5.2f} {b['x_baseline']:>9.2f} {a['x_baseline']:>9.2f} "
                  f"{ds(a):>13s} {ds(b):>13s} {best['fps']:>13.0f}")


def corr():
    rows = []
    for p in sorted(glob.glob(os.path.join(LOGS, "ckpt", "*.json"))):
        d = json.load(open(p))
        a = d["args"]
        h = d["history"][-1] if d["history"] else {}
        best = min(d["history"], key=lambda m: m["mean_abs"]) if d["history"] else {}
        rows.append({"name": os.path.basename(p)[:-5], "cell": a["cell"], "size": a["size"],
                     "net": a["net"], "shrink": a["shrink"], "loss": a["loss"],
                     "target": a["target"], "params": d["params"],
                     "gflops": d["gflops_per_frame"], "base": d["baseline"], "best": best,
                     "last": h})
    hdr = (f"{'cell':22s} {'px':>4s} {'net':16s} {'sh':>4s} {'ls':>4s} | "
           f"{'row':10s} {'mean':>7s} {'exact':>7s} {'w8':>6s} {'max':>5s} {'p99.9':>6s} "
           f"{'int-exact':>11s} {'int bad px':>11s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def line(tag, m):
            ie = m.get("interior_exact_frac")
            return (f"{'':22s} {'':>4s} {'':16s} {'':>4s} {'':>4s} | {tag:10s} "
                    f"{m.get('mean_abs', float('nan')):>7.4f} {m.get('exact_frac', 0):>7.4f} "
                    f"{m.get('within8_frac', 0):>6.3f} {m.get('max_abs', 0):>5d} "
                    f"{m.get('p999', 0):>6.0f} "
                    f"{('n/a' if ie is None else f'{ie:.8f}'):>11s} "
                    f"{m.get('interior_bad_px', 0):>5d}/{m.get('interior_px', 0):<5d}")
        print(f"{r['cell']:22s} {r['size']:>4d} {r['net']:16s} {r['shrink']:>4g} {r['loss']:>4s} | "
              f"{'params':10s} {r['params']/1e3:>6.0f}k {r['gflops']:>6.2f} GF")
        print(line("ss1 (in)", r["base"]["ss1"]))
        print(line("ss2 (ship)", r["base"]["ss2"]))
        print(line("corrected", r["best"]))
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="parity")
    ap.add_argument("--size", default="default")
    a = ap.parse_args()
    if a.what == "parity":
        parity(a.size)
    elif a.what == "fps":
        fps()
    else:
        corr()
