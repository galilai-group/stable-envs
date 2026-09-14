"""Tier and its two numerators, read off the RAYCAST-6 shard pair.

``scratch/step3/certify.py`` owns the classification and is what the sweep job runs; this prints the
same rows in one table so the before and after can be read against each other, with the denominators
each leg measured on its own host beside them. ``cores_cpu`` divides by the decode rate taken in the
SAME shard, never by a rate frozen on another node -- that substitution is what
``examples/results/step6/gfxdraw/characterize/cpu_shard_attempts.json`` records going wrong.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scratch" / "step3"))
import argparse

_AP = argparse.ArgumentParser(description=__doc__)
_AP.add_argument("--sweep", default="sweep/r6",
                 help="directory under examples/results/step6/raycast3d/ holding the pair")
SW = ROOT / "examples/results/step6/raycast3d" / _AP.parse_args().sweep


def rows(path: Path) -> dict:
    d = json.loads(path.read_text())
    out = {}
    for r in d.get("envs", []):
        cells = [c["fps"] for c in r.get("cells", []) if c.get("fps")]
        best = max(cells) if cells else None
        oc = (r.get("official_cpu") or {})
        ref = (oc.get("set_state_render") or {}).get("fps")
        dec = (r.get("decode") or {}).get("decode_fps_per_core")
        out[r["env"]] = {"fps": best, "official": ref, "decode": dec,
                         "S_cpu": best / ref if best and ref else None,
                         "cores": best / dec if best and dec else None,
                         "spread": (oc.get("paired") or {}).get("spread")}
    return out


def main() -> int:
    b = SW / "cpu_1thread_before_integration.json"
    a = SW / "cpu_1thread_after_branch.json"
    for p in (b, a):
        if not p.is_file():
            print(f"missing {p}")
            return 1
    B, A = rows(b), rows(a)
    print(f"{'env':28s}{'fps before':>11s}{'fps after':>10s}{'x':>7s}"
          f"{'S before':>10s}{'S after':>9s}{'cores b':>9s}{'cores a':>9s}{'spread':>8s}")
    for e in A:
        x, y = B.get(e, {}), A[e]
        r = (y["fps"] / x["fps"]) if x.get("fps") and y.get("fps") else None
        f = lambda v, n=1: ("-" if v is None else f"{v:,.{n}f}")            # noqa: E731
        print(f"{e:28s}{f(x.get('fps'),0):>11s}{f(y['fps'],0):>10s}{f(r,2):>7s}"
              f"{f(x.get('S_cpu'),2):>10s}{f(y['S_cpu'],2):>9s}"
              f"{f(x.get('cores'),3):>9s}{f(y['cores'],3):>9s}{f(y['spread'],2):>8s}"
              + ("   <-- cores_cpu >= 1" if (y["cores"] or 0) >= 1 else ""))
    worst = max((r["spread"] or 0) for r in list(A.values()) + list(B.values()))
    print(f"\nworst paired reference spread in either leg: {worst:.2f}x "
          f"(bound 1.25x; over it the shard is written marked and NOT published as a tier)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
