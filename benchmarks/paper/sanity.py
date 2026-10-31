"""Plausibility checks of a throughput run before its numbers enter a table.

A flagged row is not reported until the flag is explained. Checks per env:
  - every arm finished with status "ok";
  - the timed windows of one arm agree within SPREAD (max/min rate);
  - a Gymnasium SyncVectorEnv is not much slower than one env (SYNC_FLOOR x single);
  - an AsyncVectorEnv with N workers is at least ASYNC_FLOOR x single (CPU-bound sources scale with workers);
  - stable-envs T=16 is not much slower than T=1 (CHUNK_FLOOR);
  - optionally, no arm moves more than DRIFT x from a reference run (--ref, same layout).

usage: python benchmarks/paper/sanity.py RUN_DIR [--ref REF_RUN_DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPREAD = 1.3
SYNC_FLOOR = 0.7
ASYNC_FLOOR = 1.5
CHUNK_FLOOR = 0.7
DRIFT = 2.0


def arms(env_dir: Path) -> dict:
    out = {}
    for f in sorted(env_dir.glob("*.json")):
        out.update(json.loads(f.read_text())["arms"])
    return out


def check(env: str, a: dict, ref: dict | None) -> list[str]:
    flags = []
    rate = {k: v.get("steps_per_s") for k, v in a.items() if v.get("status") == "ok"}
    flags += [
        f"{k}: status {v.get('status')} {v.get('error', '')[:120]}" for k, v in a.items() if v.get("status") != "ok"
    ]
    for k, v in a.items():
        rates = [r for r in v.get("rates") or [] if r]
        if len(rates) > 1 and max(rates) / min(rates) > SPREAD:
            flags.append(f"{k}: windows spread {max(rates) / min(rates):.2f}x")
    single = rate.get("src-single-render")
    if single:
        if rate.get("src-sync-render", single) < SYNC_FLOOR * single:
            flags.append(f"sync {rate['src-sync-render']:.0f} < {SYNC_FLOOR} x single {single:.0f}")
        if "src-async-render" in rate and rate["src-async-render"] < ASYNC_FLOOR * single:
            flags.append(f"async {rate['src-async-render']:.0f} < {ASYNC_FLOOR} x single {single:.0f}")
    for base in ("snv-render-cpu", "snv-render-gpu"):
        t1, t16 = rate.get(f"{base}-T1"), rate.get(f"{base}-T16")
        if t1 and t16 and t16 < CHUNK_FLOOR * t1:
            flags.append(f"{base}: T16 {t16:.0f} < {CHUNK_FLOOR} x T1 {t1:.0f}")
    if ref:
        for k, r in rate.items():
            old = (ref.get(k) or {}).get("steps_per_s")
            if old and not (1 / DRIFT <= r / old <= DRIFT):
                flags.append(f"{k}: {r:.0f} vs reference {old:.0f} ({r / old:.2f}x)")
    return flags


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--ref")
    args = p.parse_args()
    root = Path(args.run) / "throughput"
    bad = 0
    for env_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        ref_dir = Path(args.ref) / "throughput" / env_dir.name if args.ref else None
        ref = arms(ref_dir) if ref_dir and ref_dir.is_dir() else None
        flags = check(env_dir.name, arms(env_dir), ref)
        bad += bool(flags)
        print(f"{env_dir.name}: {'FLAG' if flags else 'ok'}")
        for f in flags:
            print(f"  - {f}")
    print(f"{bad} env(s) flagged")


if __name__ == "__main__":
    main()
