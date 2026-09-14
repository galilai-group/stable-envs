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
                "device_us": (r.get("kernels") or {}).get("device_us_total"),
                "tests_per_px": (tests / (int(B) * px)) if tests else None,
            }
        out[n["env"]] = {"size": n["size"], "from": n.get("stable_envs_from"),
                         "host": n.get("host"), "job": n.get("slurm_job"),
                         "threads": n.get("threads"), "rows": rows}
    return out


def step3_rows():
    """The frozen Step-3 CPU rows on ``node2349``: se fps, official fps and the decode rate.

    The denominators must come from the frozen certification record, not be re-derived here: the bar
    is ``cores_cpu = se_fps / decode_fps_per_core`` and ``S_cpu = se_fps / official_cpu_fps``, and a
    tier computed against a locally invented denominator is not a tier.
    """
    d = json.loads((ROOT / "examples/results/step3/certification.json").read_text())
    out = {}
    for r in d["labels"]:
        if r.get("device") != "cpu" or "raycast3d" not in r.get("backends", []):
            continue
        if not str(r.get("host", "")).startswith("node2349"):
            continue                      # the Oscar 1-thread rows; the laptop ones are a different host
        e = r["evidence"]
        out[r["env"]] = {"se_fps": e.get("se_fps"), "official": e.get("official_cpu_fps"),
                         "decode": e.get("decode_fps_per_core"), "cores": e.get("cores"),
                         "ratio": e.get("ratio"), "tier": r["tier"], "q": r["qualifiers"],
                         "best_batch": e.get("best_batch")}
    return out


def tiers(before, after):
    s3 = step3_rows()
    print("\n=== tier, projected by scaling the frozen Step-3 row by the measured speed-up ===")
    print("   (S_cpu bar 5, cores_cpu bar 1; CPU_FAST needs BOTH)")
    print(f"{'env':26s} {'step3 fps':>10s} {'best x':>7s} {'proj fps':>9s} "
          f"{'S_cpu':>12s} {'cores_cpu':>16s} {'tier':>20s}  speed-up from")
    for env in sorted(after):
        b, a = before.get(env), after[env]
        if not b or env not in s3:
            print(f"{env:26s} {'-':>10s} {'-':>7s} {'-':>9s} {'no frozen Step-3 node2349 row':>30s}")
            continue
        r = s3[env]
        # The speed-up AT THE BATCH STEP 3 CERTIFIED THE ROW AT, not the best one across batches.
        # Step 3's se_fps is that row's number at its own best_batch, so scaling it by a ratio
        # measured somewhere else compares two different operating points and flatters whichever
        # batch happened to gain most. Falls back to the max only when that batch was not swept, and
        # says so.
        bb = r.get("best_batch")
        at_bb = (bb in a["rows"] and bb in b["rows"] and b["rows"][bb]["fps"])
        if at_bb:
            x, how = a["rows"][bb]["fps"] / b["rows"][bb]["fps"], f"@B={bb}"
        else:
            xs = [(a["rows"][B]["fps"] / b["rows"][B]["fps"])
                  for B in a["rows"] if B in b["rows"] and b["rows"][B]["fps"]]
            x, how = (max(xs) if xs else 1.0), f"max (B={bb} not swept)"
        proj = r["se_fps"] * x
        S, C = proj / r["official"], proj / r["decode"]
        tier = ("CPU_FAST" if (S >= 5 and C >= 1)
                else "CPU_FASTER+BELOW_G4" if S >= 2 else "CPU_PARITY+BELOW_G4")
        print(f"{env:26s} {r['se_fps']:10.1f} {x:7.2f} {proj:9.1f} "
              f"{r['ratio']:6.2f}->{S:5.2f} {r['cores']:7.4f}->{C:7.4f} {tier:>20s}  {how}")


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
    # THE UNION IS OVER THE BATCH, and that shows up here rather than in any single number: the ray
    # set for a primitive is the rays near it in ANY row, so it grows with B while the per-row set
    # does not. Printed explicitly because it is the honest limit of this design and the lever for
    # whatever comes next.
    print("\n=== the batch-union effect: tests/px at B=1 against B=64 (after) ===")
    for env in sorted(after):
        r = after[env]["rows"]
        one, many = r.get(1, {}).get("tests_per_px"), r.get(64, {}).get("tests_per_px")
        if one and many:
            print(f"  {env:26s} B=1 {one:6.2f}   B=64 {many:6.2f}   the union costs {many / one:5.1f}x")
    tiers(before, after)
    # PROVENANCE PER ROW, not per run: the first sweep of this workstream compared the branch against
    # ITSELF because sys.path.insert shadowed $PYTHONPATH, and nothing in the output said so. Every
    # row now carries the interpreter's own answer for where stable_envs came from.
    # ATTRIBUTION. The sweep above compares the branch against the PRISTINE pre-Step-6 baseline, so
    # its ratios carry 2a and 2b as well. These two legs are the same branch with the switch pinned,
    # one thread, one node, one job -- the only numbers that belong to the bound alone.
    off, on = sweep("bounds_off"), sweep("bounds_on")
    if off and on:
        print("\n=== R4 alone: same branch, PRIMITIVE_BOUNDS off vs on ===")
        print(f"{'env':26s} {'B':>5s} {'off fps':>9s} {'on fps':>9s} {'x':>6s} "
              f"{'tests/px off':>13s} {'on':>8s} {'x':>6s}")
        for env in sorted(off):
            for B in sorted(off[env]["rows"]):
                rb, ra = off[env]["rows"][B], on.get(env, {}).get("rows", {}).get(B)
                if ra is None:
                    continue
                tb, ta = rb["tests_per_px"], ra["tests_per_px"]
                tx = f"{tb / ta:.1f}" if (tb and ta) else "-"
                print(f"{env:26s} {B:5d} {rb['fps']:9.1f} {ra['fps']:9.1f} "
                      f"{ra['fps'] / rb['fps']:6.2f} {(tb or 0):13.2f} {(ta or 0):8.2f} {tx:>6s}")

    # THE GPU CLAIM IS "NOTHING MOVED", so the number that has to be quoted is DEVICE time, not fps.
    # fps on this leg also carries RAYCAST-2a, which is already merged; device_us is what PRIMITIVE_BOUNDS
    # could have touched and did not.
    gb, ga = sweep("gpu_before_baseline"), sweep("gpu_after")
    if gb and ga:
        print("\n=== cuda no-regression: device time is the claim, fps is 2a's already-merged win ===")
        print(f"{'env':22s} {'B':>5s} {'dev us before':>14s} {'dev us after':>13s} {'x':>6s} "
              f"{'fps before':>11s} {'fps after':>10s} {'x':>6s}")
        for env in sorted(gb):
            for B in sorted(gb[env]["rows"]):
                rb, ra = gb[env]["rows"][B], ga.get(env, {}).get("rows", {}).get(B)
                if ra is None:
                    continue
                db, da = rb.get("device_us"), ra.get("device_us")
                dx = f"{da / db:.3f}" if (db and da) else "-"
                print(f"{env:22s} {B:5d} {(db or 0):14.0f} {(da or 0):13.0f} {dx:>6s} "
                      f"{rb['fps']:11.1f} {ra['fps']:10.1f} {ra['fps'] / rb['fps']:6.2f}")

    print("\n=== provenance, per row ===")
    print(f"{'env':26s} {'leg':6s} {'host':12s} {'job':9s} {'thr':>3s}  stable_envs.__file__")
    for leg, d in (("before", before), ("after", after),
                   ("gpu-b", gb), ("gpu-a", ga),
                   ("bnd-off", off), ("bnd-on", on)):
        for env in sorted(d):
            v = d[env]
            print(f"{env:26s} {leg:6s} {str(v['host']).split('.')[0]:12s} "
                  f"{str(v['job']):9s} {str(v['threads']):>3s}  {v['from']}")


if __name__ == "__main__":
    sys.exit(main())
