"""Replay the five frozen gfxdraw configurations and diff their digests against Step 4's oracle.

The authoritative replay is ``examples/fidelity_step4.py``; this is the tight loop that runs in
seconds so a renderer edit is checked at every step rather than once at the end. It computes the
SAME three digests that ``fidelity_step4.oracle`` records -- default style, the corpus style, and a
per-batch-size repeat -- over the SAME frozen states, with the repository's own
``conformance.frames_digest`` and ``conformance.uniform_style``, and compares them to the frozen
numbers in ``examples/results/step4/conformance/raster.json``. It never writes into step4/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ("Acrobot", "CartPole", "MountainCar", "MountainCarContinuous", "Pendulum")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batches", default="64", help="extra batch sizes to check invariance over")
    ap.add_argument("--out", default="")
    ap.add_argument("--snapshot-out", default="", help="write these digests as a HOST baseline")
    ap.add_argument("--snapshot-in", default="",
                    help="compare against a host baseline as well as the frozen Step 4 oracle. "
                         "Needed because the frozen digests were taken on Oscar under its pygame, "
                         "and Pendulum's torque sprite is rasterised by pygame itself -- so a Mac "
                         "disagrees with the frozen number before any edit, while still matching "
                         "the official renderer ON THIS HOST exactly.")
    a = ap.parse_args()

    import stable_envs as se
    from stable_envs.testing import conformance as C

    frozen = json.loads((ROOT / "examples/results/step4/conformance/raster.json").read_text())
    snap = json.loads(Path(a.snapshot_in).read_text()) if a.snapshot_in else None
    rows, bad, snapshot = {}, 0, {}
    for name in a.configs.split(","):
        ref = frozen["configs"][name]["oracle"]
        states = np.load(ROOT / f"examples/results/step4/states/{name}.npz")["states"]
        env = se.make(name, batch=len(states), device=a.device)
        got = {"digest_ours": C.frames_digest(env.render(states).cpu().numpy())}
        got["digest_ours_default_style"] = C.frames_digest(env.render(
            states, style=np.tile(np.asarray(env.default_style(), np.float32), (len(states), 1))))
        style = C.uniform_style(env, len(states), groups=2, seed=0)
        got["digest_ours_corpus_style"] = C.frames_digest(env.render(states, style=style))
        snapshot[name] = got
        row = {k: {"frozen": ref[k], "got": v, "match": ref[k] == v} for k, v in got.items()}
        if snap is not None:
            for k, v in got.items():
                row[k]["host_baseline"] = snap[name][k]
                row[k]["match"] = snap[name][k] == v

        # batch-size invariance: the same states rendered in a different batch must give the same
        # pixels, which is where a window or a cached index that quietly depends on B shows up.
        full = env.render(states).cpu().numpy()
        for B in [int(b) for b in a.batches.split(",") if b]:
            e2 = se.make(name, batch=B, device=a.device)
            chunks = [e2.render(states[i:i + B]).cpu().numpy() for i in range(0, len(states), B)
                      if len(states[i:i + B]) == B]
            n = len(chunks) * B
            row[f"batch_{B}_identical"] = bool(n and np.array_equal(
                np.concatenate(chunks), full[:n]))
        rows[name] = row
        ok = all(v["match"] if isinstance(v, dict) else v for v in row.values())
        bad += not ok
        print(f"[{name:22s}] {'OK' if ok else 'MISMATCH'}")
        if not ok:
            for k, v in row.items():
                print(f"    {k}: {v}")
    if a.snapshot_out:
        Path(a.snapshot_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.snapshot_out).write_text(json.dumps(snapshot, indent=1))
        print(f"wrote host baseline {a.snapshot_out}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"device": a.device, "torch": torch.__version__, "configs": rows}, indent=1))
        print(f"wrote {a.out}")
    what = "the host baseline" if snap is not None else "the Step 4 oracle"
    print(f"\n{len(rows) - bad}/{len(rows)} configurations bit-identical to {what}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
