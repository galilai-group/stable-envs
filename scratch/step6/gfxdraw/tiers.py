"""Tiers for the gfxdraw before/after sweeps, computed by the FROZEN Step 2/3 evaluator.

No bound is written here. ``scratch/step3/certify.py`` reads every threshold out of
``scratch/certification.json`` and a unit test fails if a constant appears in it; this just points
it at the two Step 6 shards and prints the before/after pair per configuration, so a tier change is
a re-run of the same evaluator over new measurements rather than a claim.

    python scratch/step6/gfxdraw/tiers.py --before <shard> --after <shard>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scratch" / "step3"))
import certify as CT  # noqa: E402


def label_map(shard: Path, census: dict) -> dict:
    m = CT.merge([shard])
    return {l["env"]: l for l in CT.certify(m, census)["labels"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    census = CT.read_censuses(sorted((ROOT / "scratch" / "census").glob("census_*.json")))
    before = label_map(Path(a.before), census)
    after = label_map(Path(a.after), census)

    rows = []
    hdr = (f"| {'config':22s} | {'B':>5s} | {'f/s before':>10s} | {'f/s after':>9s} | {'x':>5s} | "
           f"{'ratio before':>12s} | {'ratio after':>11s} | {'cores before':>12s} | "
           f"{'cores after':>11s} | tier before -> tier after |")
    print(hdr)
    print("|" + "|".join(["-" * 24, "-" * 7, "-" * 12, "-" * 11, "-" * 7, "-" * 14, "-" * 13,
                          "-" * 14, "-" * 13, "-" * 27]) + "|")
    for env in sorted(after):
        b, f = before.get(env, {}), after[env]
        eb, ef = b.get("evidence", {}), f.get("evidence", {})
        sb, sf = eb.get("se_fps"), ef.get("se_fps")
        rb = (sb / eb["official_cpu_fps"]) if sb and eb.get("official_cpu_fps") else None
        rf = (sf / ef["official_cpu_fps"]) if sf and ef.get("official_cpu_fps") else None
        cb = (sb / eb["decode_fps_per_core"]) if sb and eb.get("decode_fps_per_core") else None
        cf = (sf / ef["decode_fps_per_core"]) if sf and ef.get("decode_fps_per_core") else None
        row = {"env": env, "device": f["device"], "best_batch": ef.get("best_batch"),
               "fps_before": sb, "fps_after": sf,
               "ratio_before": rb, "ratio_after": rf, "cores_before": cb, "cores_after": cf,
               "tier_before": b.get("tier"), "tier_after": f["tier"],
               "qualifiers_before": b.get("qualifiers"), "qualifiers_after": f["qualifiers"],
               "host": f.get("host"), "device_name": f.get("device_name"),
               "threads": f.get("threads"), "backend_class": f.get("backend_class")}
        rows.append(row)
        n = lambda x, d=2: ("-" if x is None else f"{x:,.{d}f}")                  # noqa: E731
        print(f"| {env:22s} | {ef.get('best_batch', '-'):>5} | {n(sb, 0):>10s} | {n(sf, 0):>9s} | "
              f"{(sf / sb if sb and sf else 0):>4.1f}x | {n(rb):>12s} | {n(rf):>11s} | "
              f"{n(cb):>12s} | {n(cf):>11s} | {b.get('tier')} -> {f['tier']} "
              f"{'/'.join(f['qualifiers']) or ''} |")

    # The two qualifiers the GPU ladder requires, from the cells rather than from the best batch.
    for arm, shard in (("before", a.before), ("after", a.after)):
        doc = json.loads(Path(shard).read_text())
        for r in doc.get("envs", doc.get("runs", [])) or []:
            cells = {c["batch"]: c.get("fps") for c in (r.get("cells") or []) if c.get("fps")}
            if 4096 in cells and 1 in cells:
                bs = cells[4096] >= 20 * cells[1]
                nl = (4096 in cells and 256 in cells and cells[4096] >= 0.9 * cells[256])
                print(f"  {arm:6s} {r['env']:22s} BATCH_SCALES {cells[4096] / cells[1]:7.1f}x "
                      f"{'PASS' if bs else 'FAIL'}   NO_LARGE_BATCH_REGRESSION "
                      f"{'PASS' if nl else 'FAIL'}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"before_shard": a.before, "after_shard": a.after,
                                           "rows": rows}, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
