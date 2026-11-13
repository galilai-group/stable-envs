"""Plausibility checks of a throughput run before its numbers enter a table.

A flagged row is not reported until the flag is explained. Checks per env:
  - every arm finished with status "ok";
  - the timed windows of one arm agree within SPREAD (max/min rate);
  - a Gymnasium SyncVectorEnv is not much slower than one env (SYNC_FLOOR x single);
  - an AsyncVectorEnv with N workers is at least ASYNC_FLOOR x single (CPU-bound sources scale with workers);
  - stable-envs T=16 is not much slower than T=1 (CHUNK_FLOOR);
  - optionally, no arm moves more than DRIFT x from a reference run (--ref, same layout).

Offline regime (RUN_DIR/offline), per env: every file present and every arm "ok"; stable-envs GPU frames/s at its
best batch at least the source's best arm (speedup >= 1); GPU frame MSE mean at most MSE_MAX. It also writes
RUN_DIR/offline_gpu.csv with one row per env.

usage: python benchmarks/paper/sanity.py RUN_DIR [--ref REF_RUN_DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SPREAD = 1.3
SYNC_FLOOR = 0.7
ASYNC_FLOOR = 1.5
CHUNK_FLOOR = 0.7
DRIFT = 2.0
MSE_MAX = 1.0


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


def _env_row(path: Path, env: str) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())["envs"].get(env)


def check_offline(env_dir: Path) -> tuple[dict, list[str]]:
    env, flags = env_dir.name, []
    row = {"env": env}
    ours = _env_row(env_dir / "snv-gpu.json", env)
    points = (ours or {}).get("points", [])
    ok = [p for p in points if p.get("status") == "ok"]
    if not ok:
        flags.append(
            "snv-gpu: no batch finished" + (f" ({points[0].get('error', '')[:100]})" if points else " (missing)")
        )
    flags += [f"snv-gpu b={p['batch']}: {p['status']}" for p in points if p.get("status") != "ok"]
    best = max(ok, key=lambda p: p["fps"]) if ok else None
    sources = {}
    for f in sorted(env_dir.glob("src-*.json")):
        r = _env_row(f, env)
        if r and r.get("status") == "ok":
            sources[f.stem] = r
        elif r and r.get("status") != "skipped":
            flags.append(f"{f.stem}: {r.get('status')} {r.get('error', '')[:100]}")
    src = max(sources.items(), key=lambda kv: kv[1]["fps"]) if sources else None
    if not src:
        flags.append("no source arm finished")
    for name, r in [("snv-gpu", best)] + ([src] if src else []):
        rates = [x for x in (r or {}).get("rates") or [] if x]
        if len(rates) > 1 and max(rates) / min(rates) > SPREAD:
            flags.append(f"{name}: windows spread {max(rates) / min(rates):.2f}x")
    mse = _env_row(env_dir / "mse-gpu.json", env) or {}
    if mse.get("status") != "ok":
        flags.append(f"mse-gpu: {mse.get('status', 'missing')} {mse.get('reason', mse.get('error', ''))[:100]}")
    elif mse["mse_mean"] > MSE_MAX:
        flags.append(f"frame MSE mean {mse['mse_mean']:.2f} > {MSE_MAX} (max {mse['mse_max']:.2f})")
    speedup = best["fps"] / src[1]["fps"] if best and src else None
    if speedup is not None and speedup < 1:
        flags.append(f"below source: {best['fps']:.0f} vs {src[0]} {src[1]['fps']:.0f} ({speedup:.2f}x)")
    row.update(
        snv_gpu_fps=best and round(best["fps"], 1),
        snv_gpu_batch=best and best["batch"],
        source_fps=src and round(src[1]["fps"], 1),
        source_arm=src and src[0],
        speedup=speedup and round(speedup, 2),
        mse_mean=mse.get("mse_mean"),
        mse_max=mse.get("mse_max"),
        flags="; ".join(flags),
    )
    return row, flags


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--ref")
    args = p.parse_args()
    off = Path(args.run) / "offline"
    if off.is_dir():
        rows, bad = [], 0
        for env_dir in sorted(d for d in off.iterdir() if d.is_dir()):
            row, flags = check_offline(env_dir)
            rows.append(row)
            bad += bool(flags)
            print(f"{env_dir.name}: {'FLAG' if flags else 'ok'} speedup {row['speedup']} mse {row['mse_mean']}")
            for f in flags:
                print(f"  - {f}")
        with open(Path(args.run) / "offline_gpu.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"offline: {bad}/{len(rows)} env(s) flagged")
    root = Path(args.run) / "throughput"
    if not root.is_dir():
        return
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
