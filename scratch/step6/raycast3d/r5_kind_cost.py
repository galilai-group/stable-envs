"""Measure, per primitive KIND, whether the bound can ever pay for itself.

THE PREDICATE THIS PRODUCES IS SCENE-INDEPENDENT, WHICH IS THE WHOLE POINT. RAYCAST-4 shipped a bound
that is on for every kind, and it cost ``PointmazeLargeNavigate`` 8 % and ``PointmazeMediumNavigate``
25 % while gaining 3.08x on ``ball_in_cup``. The obvious repair -- a threshold on ``K``, on ``N``, or
on the selectivity -- would be a constant fitted to the eight rows that happened to be swept, which is
exactly the move that put a far-field precision bug past two test suites earlier in this workstream.

The mechanism is not about the scene at all. For one primitive over a bundle of ``N`` rays:

    unbounded:  cost_intersect(kind) * N
    bounded:    cost_bound * N  +  cost_intersect(kind) * N * p        (p = fraction of rays kept)

so the bound pays iff ``cost_bound < cost_intersect(kind) * (1 - p)``. ``p`` depends on the scene and
is not knowable here, but ``p >= 0``, so the NECESSARY condition -- the one that holds for every
scene, every batch, every resolution -- is

    cost_bound < cost_intersect(kind).

A kind whose intersector is cheaper than the bound's own mask + nonzero + gather + scatter can never
pay, no matter how well the bound prunes: there is nothing to save. A sphere is a handful of ops; a
capsule is an order of magnitude more. That ratio is a property of the intersector's arithmetic, so
measuring it once on a batch node gives a table that is a fact about the code rather than a fit to a
sweep.

What this script measures, per kind and per bundle width:

  * ``ns_bound``      -- ``_rays_near`` + ``.any(0)`` + ``nonzero`` + the gather of ``o``/``d`` + the
                         scatter back, i.e. everything the bounded path does that the unbounded one
                         does not, at the WORST case for the bound (nothing pruned, so the gather is
                         full width). Reported per ray.
  * ``ns_intersect``  -- ``intersect(kind, ...)`` at full width. Reported per ray.

and the margin, which is also measured rather than chosen: the decision has to survive this
benchmark's own spread, so the rule is ``ns_intersect > (1 + 3*rsd) * ns_bound`` with ``rsd`` the
largest relative standard deviation observed across all the timings taken here.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))                    # APPEND: $PYTHONPATH decides which stable_envs is timed


def _stats(fn, repeat, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e9)          # ns
    return {"ns_median": statistics.median(ts), "ns_min": min(ts),
            "ns_stdev": statistics.stdev(ts) if len(ts) > 1 else 0.0, "samples": repeat}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--rays", type=int, nargs="+", default=[7056, 50176, 200704],
                    help="84^2 (DM Control), 224^2 (maze, ss=1), 224^2*4 (maze, ss=2)")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    import stable_envs as se
    from stable_envs.render import raycast3d as R3

    torch.set_num_threads(a.threads)
    g = torch.Generator().manual_seed(0)

    note = {"tool": "r5_kind_cost", "workstream": "step6/raycast3d",
            "host": socket.gethostname(), "cpu": platform.processor() or platform.machine(),
            "threads": torch.get_num_threads(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "torch": torch.__version__, "numpy": __import__("numpy").__version__,
            "stable_envs_from": se.__file__, "pythonpath": os.environ.get("PYTHONPATH", ""),
            "rows": []}

    KINDS = [("PLANE", R3.PLANE), ("SPHERE", R3.SPHERE), ("CAPSULE", R3.CAPSULE),
             ("CYLINDER", R3.CYLINDER), ("BOX", R3.BOX)]

    for B in a.batch:
        for N in a.rays:
            o = torch.rand(B, N, 3, generator=g) * 2.0 - 1.0
            d = torch.nn.functional.normalize(torch.rand(B, N, 3, generator=g) * 2.0 - 1.0, dim=-1)
            o, d = o.contiguous(), d.contiguous()
            dd = R3._dot(d, d)
            pos = torch.rand(B, 1, 3, generator=g)
            Rm = torch.eye(3).view(1, 1, 3, 3).expand(B, 1, 3, 3).contiguous()
            size = torch.rand(B, 1, 3, generator=g) * 0.3 + 0.05
            # THE BOUND'S COST AT ITS WORST: a radius large enough that nothing is pruned, so the
            # nonzero returns every index and the gather and scatter run at full width. Timing it with
            # a selective radius would measure the bound at its best and hide the cost this table is
            # about.
            big = torch.full((B, 1), 1e3)
            blocked = torch.zeros((B, N), dtype=torch.bool)

            def bound_path():
                js = torch.nonzero(R3._rays_near(pos, big, o, d, dd).any(0),
                                   as_tuple=False).reshape(-1)
                oo, ddd = o[:, js], d[:, js]
                blocked[:, js] |= (oo[..., 0] > ddd[..., 0])
                return js

            row = {"batch": B, "rays": N,
                   "bound": _stats(bound_path, a.repeat, a.warmup), "kinds": {}}
            for name, k in KINDS:
                fn = (lambda k=k: R3.intersect(k, size, pos, Rm, o, d))
                row["kinds"][name] = _stats(fn, a.repeat, a.warmup)
            for v in [row["bound"]] + list(row["kinds"].values()):
                v["ns_per_ray"] = v["ns_median"] / (B * N)
                v["rsd"] = (v["ns_stdev"] / v["ns_median"]) if v["ns_median"] else 0.0
            note["rows"].append(row)
            print(f"B={B:5d} N={N:7d}  bound {row['bound']['ns_per_ray']:.3f} ns/ray  " +
                  "  ".join(f"{n} {row['kinds'][n]['ns_per_ray']:.3f}" for n, _ in KINDS), flush=True)

    # THE MARGIN IS MEASURED, NOT CHOSEN: the decision must survive this benchmark's own spread.
    rsd = max(max([r["bound"]["rsd"]] + [v["rsd"] for v in r["kinds"].values()]) for r in note["rows"])
    margin = 1.0 + 3.0 * rsd
    table, detail = {}, {}
    for name, _ in KINDS:
        if name == "PLANE":
            table[name] = False                       # unbounded by construction: no bounding sphere
            detail[name] = {"reason": "no finite bounding sphere; always tested"}
            continue
        # WORST CASE OVER EVERY WIDTH MEASURED. A kind is bounded only if its intersector beats the
        # bound at every (B, N) here, so the table cannot be carried by one favourable width.
        ratios = [r["kinds"][name]["ns_per_ray"] / r["bound"]["ns_per_ray"] for r in note["rows"]]
        table[name] = min(ratios) > margin
        detail[name] = {"ratio_min": min(ratios), "ratio_max": max(ratios),
                        "ratio_median": statistics.median(ratios)}
    note["margin"] = {"value": margin, "worst_rsd": rsd,
                      "rule": "bounded(kind) iff min over (B,N) of "
                              "ns_intersect/ns_bound > 1 + 3*worst_rsd"}
    note["predicate"] = table
    note["predicate_detail"] = detail
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(note, indent=1))
    print("\nmargin", f"{margin:.3f}", "(worst rsd", f"{rsd:.4f})")
    for name, _ in KINDS:
        d_ = detail[name]
        r_ = f"ratio min {d_['ratio_min']:6.2f} max {d_['ratio_max']:6.2f}" if "ratio_min" in d_ \
            else d_["reason"]
        print(f"  {name:9s} bounded={str(table[name]):5s}  {r_}")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
