"""Private PushT get/set State cost; host-local timings, not an online speedup.

Setup trajectories are excluded. Restore timing includes validation, complete
engine reconstruction and atomic replacement. Disk serialization is measured
as bytes only; there is no source API with equivalent exact restore semantics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch

from stable_envs._adapters.pusht.replay_batch import PushTReplayAdapter


def stats(samples):
    return {
        "median_ms": float(np.median(samples) * 1000),
        "p95_ms": float(np.percentile(samples, 95) * 1000),
        "mean_ms": float(np.mean(samples) * 1000),
        "sd_ms": float(np.std(samples, ddof=1) * 1000),
        "operations_per_second": 1 / float(np.mean(samples)),
        "samples_seconds": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    if args.samples < 3:
        parser.error("At least three samples required")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    report = dict(
        status="RUNNING",
        scope="PRIVATE_REPLAY_CPU_COST_ONLY",
        host=platform.node(),
        platform=platform.platform(),
        machine=platform.machine(),
        python=sys.version,
        public_acceptance=False,
        source_speedup_claim=False,
        device="cpu",
        resolution=64,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        production_files={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "stable_envs/_adapters").glob("pusht_*.py"))
        },
        cases=[],
    )

    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for discrete in (False, True):
        for batch in (1, 16):
            for length in (0, 100, 300, 1025, 10240):
                start = time.perf_counter()
                env = PushTReplayAdapter(batch, seed=17, discrete=discrete, config={"resolution": 64})
                construction = time.perf_counter() - start
                try:
                    for lane, engine in enumerate(env.envs):
                        for step in range(max(0, length - lane % 4)):
                            control = (
                                np.array([step % 16, step // 8 % 16], np.int64)
                                if discrete
                                else np.array([(-1.0) ** (step // 32), (-1.0) ** (step // 64)], np.float32)
                            )
                            engine.step(control)
                    snapshot = env.get_state()
                    state_bytes = sum(v.numel() * v.element_size() for _, v in snapshot.items(True, True))
                    buf = io.BytesIO()
                    np.savez_compressed(buf, **{"/".join(k): v.numpy() for k, v in snapshot.items(True, True)})
                    disk_bytes = buf.tell()
                    env.set_state(snapshot)  # explicit warmup
                    gets, sets = [], []
                    for _ in range(args.samples):
                        start = time.perf_counter()
                        owned = env.get_state()
                        gets.append(time.perf_counter() - start)
                        del owned
                        start = time.perf_counter()
                        env.set_state(snapshot)
                        sets.append(time.perf_counter() - start)
                    record = dict(
                        variant="discrete" if discrete else "continuous",
                        batch=batch,
                        max_history=length,
                        replay_steps=int(snapshot["history", "lengths"].sum()),
                        state_bytes=state_bytes,
                        compressed_npz_bytes=disk_bytes,
                        construction_seconds=construction,
                        get_state=stats(gets),
                        set_state=stats(sets),
                    )
                    report["cases"].append(record)
                    save()
                    print(record["variant"], batch, length, "PASS", flush=True)
                finally:
                    env.close()
    report["status"] = "PASS_PRIVATE_CPU_REPLAY_COST"
    save()
    rows = []
    for record in report["cases"]:
        row = {
            k: record[k]
            for k in ("variant", "batch", "max_history", "replay_steps", "state_bytes", "compressed_npz_bytes")
        }
        for operation in ("get_state", "set_state"):
            for key, value in record[operation].items():
                if key != "samples_seconds":
                    row[operation + "_" + key] = value
        rows.append(row)
    with (args.output / "cost.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    main()
