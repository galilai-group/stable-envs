"""Signal against noise for the RAYCAST-5 acceptance bar.

The bar is "no row below 0.97x". Whether a row is below it is only a meaningful question if the
harness can resolve 3 %, and on a shared one-thread cluster node it cannot: this reads two legs of
IDENTICAL code (bound off, twice) and reports their ratio as the noise floor, next to the signal
averaged over two legs each way. A row whose signal sits inside the noise is not a regression; it is
an unanswered question, and saying so is the point of running the off leg twice.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

A = Path(__file__).resolve().parents[3] / "examples/results/step6/raycast3d/profile_after_r5"
BAR = 0.97
#: The frozen spec forbids certifying on B=1 (``gpu.never_certify_on_batch``): the corpus median
#: there is 0.6x the official renderer, so it measures launch overhead. A row's acceptance is
#: therefore reported at the CERTIFICATION batches, with B=1 shown but not gating.
NEVER_CERTIFY = {1}


def main() -> int:
    legs: dict = {}
    for L in ("off1", "on1", "off2", "on2"):
        d = A / f"noise_{L}"
        if not d.is_dir():
            print(f"missing leg {L}")
            return 1
        for f in sorted(d.glob("*.json")):
            n = json.loads(f.read_text())
            for B, r in n["batches"].items():
                legs.setdefault((n["env"], int(B)), {})[L] = r["fps"]

    print(f"{'env':26s} {'B':>5s} {'off1':>8s} {'off2':>8s} {'noise':>6s} | "
          f"{'on1':>8s} {'on2':>8s} {'signal':>7s}  verdict")
    worst_noise, cert_fail, b1_fail = 0.0, [], []
    for (env, B) in sorted(legs):
        v = legs[(env, B)]
        if len(v) < 4:
            continue
        noise = v["off2"] / v["off1"]
        sig = statistics.mean([v["on1"], v["on2"]]) / statistics.mean([v["off1"], v["off2"]])
        worst_noise = max(worst_noise, abs(1 - noise))
        ok = sig >= BAR
        if not ok:
            (b1_fail if B in NEVER_CERTIFY else cert_fail).append((env, B, sig, noise))
        tag = "ok" if ok else ("FAIL(B=1, not a certification batch)" if B in NEVER_CERTIFY
                               else "FAIL")
        print(f"{env:26s} {B:5d} {v['off1']:8.1f} {v['off2']:8.1f} {noise:6.3f} | "
              f"{v['on1']:8.1f} {v['on2']:8.1f} {sig:7.3f}  {tag}")

    print(f"\nNOISE FLOOR: worst off-vs-off spread on identical code {worst_noise * 100:.1f} %, "
          f"against a bar that asks about {100 * (1 - BAR):.0f} %.")
    print(f"ACCEPTANCE at the certification batches (B != 1): "
          f"{'PASS' if not cert_fail else 'FAIL'} ({len(cert_fail)} below {BAR})")
    for e, B, s, n in cert_fail:
        print(f"  {e} B={B} signal {s:.3f} (noise {n:.3f})")
    print(f"B=1, reported not gating: {len(b1_fail)} below {BAR}")
    for e, B, s, n in b1_fail:
        print(f"  {e} B={B} signal {s:.3f} (noise on identical code {n:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
