"""One row per cell: shipped against corrected, on all five declared metrics, with the ship verdict.

The verdict is not a judgement call, it is three gates and a cell has to pass all of them:

1. **Held out** -- on a DISJOINT set of recorded states, the corrector beats the shipped 256 px render
   on every one of the five quantities a ``visual_fidelity`` entry declares. This is the gate that
   cannot be won by memorising the training states.
2. **Declared** -- on the env's own ``sample_states`` at four seeds, worst of the four, the same five.
   This is the number a conformance test would compute, and it overlaps training by construction, so
   it is reported beside the held-out one rather than instead of it.
3. **Throughput** -- render plus corrector, end to end, faster than the shipped render. A corrector
   that improves fidelity and costs speed is a different trade and is not what was approved; a
   corrector that is merely faster is not either, which is why gate 1 comes first.

``mean_abs`` and ``interior_mean_abs`` must go DOWN, ``within8_frac``, ``ssim`` and
``interior_exact_frac`` must go UP. A tie is a failure: the point of the exercise is to be better.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

LOWER = ("mean_abs", "interior_mean_abs")
HIGHER = ("within8_frac", "ssim", "interior_exact_frac")
FIVE = LOWER + HIGHER


def beats(cor: dict, ship: dict, tol: float = 0.0) -> dict:
    """Per-metric pass/fail of corrector against shipped, in the direction each metric improves."""
    out = {}
    for k in LOWER:
        out[k] = cor[k] < ship[k] - tol
    for k in HIGHER:
        out[k] = cor[k] > ship[k] + tol
    return out


def row_of(res: dict) -> dict:
    """Everything the table needs out of one ``eval_corrector.py`` JSON."""
    by = {r["split"]: r for r in res["splits"]}
    out = {"ckpt": os.path.basename(res["ckpt"])[:-3], "env": res["env"], "net": res["net"],
           "tag": res["tag"], "aux": res.get("aux", "none"), "params": res.get("params"),
           "render_size": res.get("render_size"),
           "fps_shipped": res["throughput"]["shipped_256_fps"],
           "fps_corrected": res["throughput"]["render_plus_corrector"]["fps"],
           "fps_render_only": res["throughput"]["render_only_fps"]}
    out["speedup"] = out["fps_corrected"] / out["fps_shipped"]
    if "heldout" in by:
        v = by["heldout"]["variants"]
        out["heldout"] = {k: {m: v[k][m] for m in FIVE + ("max_abs", "p999", "p99",
                                                          "mean_abs_dynamic", "mean_abs_static",
                                                          "frac_over_64")}
                          for k in v}
        out["heldout_gate"] = beats(v["corrector"], v["shipped_256"])
    if "decl" in by and "worst_of_draws" in by["decl"]:
        w = by["decl"]["worst_of_draws"]
        out["decl_worst"] = w
        out["decl_gate"] = beats(w["corrector"], w["shipped_256"])
    out["speed_gate"] = out["fps_corrected"] > out["fps_shipped"]
    gates = [all(out.get("heldout_gate", {}).values()) if "heldout_gate" in out else False,
             all(out.get("decl_gate", {}).values()) if "decl_gate" in out else False,
             out["speed_gate"]]
    out["ship"] = all(gates)
    out["gates"] = {"heldout_all_five": gates[0], "decl_all_five": gates[1], "faster": gates[2]}
    return out


def as_decl_rows(rows):
    """The same rows re-keyed so ``fmt`` prints the DECLARED draw instead of the held-out split.

    ``worst_of_draws`` carries the five declared quantities and not the tails, so the two missing
    columns are filled with NaN rather than with a plausible-looking number.
    """
    out = []
    for r in rows:
        w = r.get("decl_worst")
        if not w:
            continue
        blk = {k: dict(v, p999=float("nan"), max_abs=float("nan")) for k, v in w.items()}
        blk["shipped_256"] = blk.pop("shipped_256")
        out.append(dict(r, heldout=blk, heldout_gate=r.get("decl_gate", {})))
    return out


def fmt(rows, key="heldout"):
    hdr = (f"{'cell / checkpoint':58s} {'f/s':>8s} {'x':>5s} "
           f"{'mean':>7s} {'w8':>6s} {'ssim':>7s} {'imean':>6s} {'iexact':>6s} "
           f"{'p99.9':>6s} {'max':>4s}  ship")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        blk = r.get(key)
        if not blk:
            continue
        c, s = blk["corrector"], blk.get("shipped_256")
        if s is not None:
            lines.append(f"{r['env'] + ' @ ' + r['tag']:58s} {r['fps_shipped']:8.0f} {1.0:5.2f} "
                         f"{s['mean_abs']:7.2f} {s['within8_frac']:6.3f} {s['ssim']:7.4f} "
                         f"{s['interior_mean_abs']:6.2f} {s['interior_exact_frac']:6.3f} "
                         f"{s['p999']:6.1f} {s['max_abs']:4.0f}  shipped")
        g = r.get(f"{key}_gate", {})
        mark = "".join("." if g.get(k) else "X" for k in FIVE)
        lines.append(f"  {r['ckpt']:56s} {r['fps_corrected']:8.0f} {r['speedup']:5.2f} "
                     f"{c['mean_abs']:7.2f} {c['within8_frac']:6.3f} {c['ssim']:7.4f} "
                     f"{c['interior_mean_abs']:6.2f} {c['interior_exact_frac']:6.3f} "
                     f"{c['p999']:6.1f} {c['max_abs']:4.0f}  "
                     f"{'SHIP' if r['ship'] else 'no'} [{mark}]")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+", help="eval_*.json paths or globs")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    paths = sorted({p for g in a.globs for p in glob.glob(g)})
    rows = []
    for p in paths:
        try:
            rows.append(row_of(json.load(open(p))))
            rows[-1]["json"] = p
        except Exception as e:                                        # noqa: BLE001
            print(f"skip {p}: {type(e).__name__}: {e}")
    rows.sort(key=lambda r: (r["env"], r["tag"], r["ckpt"]))
    print("HELD OUT (disjoint recorded states)\n")
    print(fmt(rows, "heldout"))
    print("\n\nDECLARED (sample_states, 4 seeds, worst of the four)\n")
    print(fmt(as_decl_rows(rows), "heldout"))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
