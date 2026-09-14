"""RAYCAST-6's interleaved CPU sweep: signal against the harness's own noise, and the tier.

Same protocol as ``r5_noise_summary.py`` and for the same reason: on a shared one-thread cluster node
this workstream measured an 11.7 % off-vs-off spread on IDENTICAL code, so a single before/after pair
cannot be read. Four legs, off/on/off/on, and ``off2/off1`` is the noise floor that the signal has to
clear. ``cores_cpu`` is reported against the decode rate the paired shard measured on the SAME node,
not against a rate frozen on another one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / "examples/results/step6/raycast3d/profile_after_r6"
SW = ROOT / "examples/results/step6/raycast3d/sweep/r6"
NEVER_CERTIFY = {1}


def decode_rates() -> dict:
    """env -> one-core JPEG-q90 decode fps, from the paired shard taken on this node."""
    out = {}
    for f in sorted(SW.glob("cpu_1thread_*.json")):
        d = json.loads(f.read_text())
        for rec in d.get("envs", []):
            dec = (rec.get("decode") or {}).get("decode_fps_per_core")
            if dec:
                out.setdefault(rec["env"], dec)
    return out


def main() -> int:
    legs: dict = {}
    for L in ("off1", "on1", "off2", "on2"):
        d = A / f"interleaved_{L}"
        if not d.is_dir():
            print(f"missing leg {L} ({d})")
            return 1
        for f in sorted(d.glob("*.json")):
            n = json.loads(f.read_text())
            for B, r in n["batches"].items():
                if r.get("fps"):
                    legs.setdefault((n["env"], int(B)), {})[L] = r["fps"]

    dec = decode_rates()
    print(f"{'env':28s} {'B':>5s} {'off1':>9s} {'off2':>9s} {'noise':>6s} | "
          f"{'on1':>9s} {'on2':>9s} {'signal':>7s} | {'cores off':>9s} {'cores on':>9s}")
    worst_noise = 0.0
    for (env, B) in sorted(legs):
        v = legs[(env, B)]
        if len(v) < 4:
            print(f"{env:28s} {B:5d}  incomplete: {sorted(v)}")
            continue
        noise = v["off2"] / v["off1"]
        off = (v["off1"] + v["off2"]) / 2
        on = (v["on1"] + v["on2"]) / 2
        if B not in NEVER_CERTIFY:
            worst_noise = max(worst_noise, max(noise, 1 / noise))
        d = dec.get(env)
        c0 = f"{off / d:9.3f}" if d else f"{'-':>9s}"
        c1 = f"{on / d:9.3f}" if d else f"{'-':>9s}"
        print(f"{env:28s} {B:5d} {v['off1']:9.1f} {v['off2']:9.1f} {noise:6.2f} | "
              f"{v['on1']:9.1f} {v['on2']:9.1f} {on / off:7.3f} | {c0} {c1}"
              + ("   (B=1 is not a certification batch)" if B in NEVER_CERTIFY else ""))
    print(f"\nworst off-vs-off spread on identical code, certification batches only: "
          f"{worst_noise:.3f}x")
    if not dec:
        print("NO DECODE RATE: the paired shard is missing or was rejected, so cores_cpu cannot be "
              "computed here and must not be quoted from another node's frozen row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
