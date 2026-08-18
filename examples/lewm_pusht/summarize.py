"""Aggregate the per-run JSONs into the table that goes in the README.

The two arms are *paired*: for a given seed the episode indices, start steps, goal offset and CEM seed
are identical, so episode i in the official run and episode i in the swapped run are the same task. That
makes McNemar's exact test the right one — it looks only at the episodes where the two arms disagree and
asks whether the split between them is lopsided. An unpaired comparison of two percentages would throw
away exactly the structure that makes this a controlled experiment.

    python summarize.py
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p for b vs c discordant pairs (binomial, p=0.5)."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def main() -> None:
    here = Path(__file__).resolve().parent
    runs = defaultdict(dict)  # renderer -> seed -> record
    for f in sorted(glob.glob(str(here / "results" / "*.json"))):
        try:
            d = json.loads(Path(f).read_text())
        except json.JSONDecodeError:
            continue
        if "renderer" not in d or "episode_successes" not in d:
            continue
        runs[d["renderer"]][d["seed"]] = d

    order = ["official", "stable_envs", "stable_envs_resize512", "stable_envs_live",
             "stable_envs_dataset", "blank"]
    names = [r for r in order if r in runs] + [r for r in runs if r not in order]

    print(f"{'renderer':24s} {'seeds':>22s} {'succ/n':>10s} {'SR %':>7s} {'95% CI':>16s}")
    agg = {}
    for r in names:
        seeds = sorted(runs[r])
        succ = np.concatenate([np.asarray(runs[r][s]["episode_successes"]) for s in seeds])
        agg[r] = {s: np.asarray(runs[r][s]["episode_successes"]) for s in seeds}
        k, n = int(succ.sum()), len(succ)
        lo, hi = wilson(k, n)
        print(f"{r:24s} {','.join(map(str, seeds)):>22s} {f'{k}/{n}':>10s} "
              f"{100 * k / n:7.1f} {f'[{lo:.1f}, {hi:.1f}]':>16s}")

    print()
    if "official" not in agg:
        return
    for r in names:
        if r == "official":
            continue
        shared = sorted(set(agg["official"]) & set(agg[r]))
        if not shared:
            continue
        a = np.concatenate([agg["official"][s] for s in shared]).astype(bool)
        b = np.concatenate([agg[r][s] for s in shared]).astype(bool)
        only_off = int((a & ~b).sum())
        only_new = int((~a & b).sum())
        p = mcnemar_exact(only_off, only_new)
        print(f"official vs {r:24s} paired n={len(a):4d}  "
              f"official-only wins={only_off:3d}  {r}-only wins={only_new:3d}  "
              f"delta={100 * (b.mean() - a.mean()):+5.1f} pts  McNemar p={p:.3f}")


if __name__ == "__main__":
    main()
