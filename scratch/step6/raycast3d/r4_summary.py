"""Read the RAYCAST-4 artifacts and print the table the findings need.

Kept as a script rather than done by hand because the numbers it joins come from six files and the
join -- which leg, which batch, tests per pixel against fps -- is exactly where a transcription error
would be invisible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / "examples/results/step6/raycast3d"


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:                                                   # noqa: BLE001
        return None


def sweep(leg):
    out = {}
    d = A / "profile_after_r4" / leg
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.json")):
        n = json.loads(f.read_text())
        rows = {}
        for B, r in n["batches"].items():
            tests = (r.get("facts", {}).get("ray_primitive_tests") or {}).get("sum_per_render")
            px = int(n["size"]) ** 2
            rows[int(B)] = {
                "fps": r["fps"], "ms": r["render_ms"]["ms_median"],
                "tests": tests,
                "tests_per_px": (tests / (int(B) * px)) if tests else None,
            }
        out[n["env"]] = {"size": n["size"], "from": n.get("stable_envs_from"), "rows": rows}
    return out


def main():
    print("=== identity and replay ===")
    for name, f in (("cpu: branch vs pristine baseline, 21 configs", "replay/r4_cpu_branch_vs_baseline.json"),
                    ("cpu: PRIMITIVE_BOUNDS on vs off, 21 configs", "replay/r4_cpu_bounds_identity.json"),
                    ("cuda: branch vs pristine baseline, 21 configs", "replay/r4_cuda_pixel_diff.json")):
        d = load(A / f)
        if d is None:
            print(f"  {name:46s} (missing)")
            continue
        bad = [k for k, v in d["configs"].items() if not v.get("identical")]
        print(f"  {name:46s} all_identical={d['all_identical']} n={d.get('n_compared')}"
              + (f"  DIFFERS: {bad}" if bad else ""))

    before, after = sweep("before_baseline"), sweep("after")
    if not before:
        print("\n(no sweep yet)")
        return
    print("\n=== 1-thread CPU sweep, fps and ray-primitive tests per pixel ===")
    print(f"{'env':26s} {'px':>5s} {'B':>5s} {'before fps':>11s} {'after fps':>10s} {'x':>6s} "
          f"{'tests/px before':>16s} {'after':>8s} {'x':>6s}")
    for env in sorted(before):
        b, a = before[env], after.get(env, {"rows": {}})
        for B in sorted(b["rows"]):
            rb, ra = b["rows"][B], a["rows"].get(B)
            if ra is None:
                continue
            sp = f"{rb['fps'] and ra['fps'] / rb['fps']:.2f}" if rb["fps"] else "-"
            tb, ta = rb["tests_per_px"], ra["tests_per_px"]
            tx = f"{tb / ta:.1f}" if (tb and ta) else "-"
            print(f"{env:26s} {b['size']:5d} {B:5d} {rb['fps']:11.1f} {ra['fps']:10.1f} {sp:>6s} "
                  f"{(tb if tb else 0):16.2f} {(ta if ta else 0):8.2f} {tx:>6s}")
    print("\nlegs:", {k: v["from"] for k, v in list(before.items())[:1]},
          {k: v["from"] for k, v in list(after.items())[:1]})


if __name__ == "__main__":
    sys.exit(main())
