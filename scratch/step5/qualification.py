"""Is the Reacher instrument alive before it is asked to judge a renderer?

Step 5 section 5: reproduce the official-renderer baseline first. A renderer-equivalence test at
``official = 0%`` is not a null result, it is no result -- two broken arms agree perfectly.

This script collects the official-renderer runs and puts them beside the baseline
``stable_worldmodel`` records for the same 50-step budget, so "in the expected neighbourhood and
clearly non-degenerate" is a comparison a reader can check rather than a claim. It also records the
blank-frame arm, because a competent instrument that does not depend on pixels would qualify by
this test and still be useless for the question Step 5 asks.

    python scratch/step5/qualification.py --runs examples/results/step5/reacher/runs_lewm \
        --out examples/results/step5/reacher/qualification.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: What stable_worldmodel's own docs/baselines.md records for Reacher under the fixed 50-step
#: evaluation budget. Quoted, not recomputed: it is the external number this run is checked against.
RECORDED = {
    "quentinll/lewm-reacher": ("86 %", "LeWM, swm docs/baselines.md"),
    "lewm": ("86 %", "LeWM, swm docs/baselines.md"),
    "prejepa": ("79 %", "DINO-WM, swm docs/baselines.md"),
    # Not a baseline to reproduce -- the floor a trained instrument has to clear. Measured here
    # rather than assumed to be zero, because reaching a target qpos within 0.05 rad on a two-joint
    # arm is something random torques manage sometimes, and "sometimes" is the number that decides
    # how much of a trained arm's success rate is the model.
    "random": ("", "random-action floor, measured here"),
}


def recorded_for(policy: str) -> tuple[str, str]:
    for k, v in RECORDED.items():
        if k in policy:
            return v
    return ("", "no recorded baseline")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    rows = []
    for d in args.runs:
        for p in sorted(d.rglob("*.json")):
            r = json.loads(p.read_text())
            if "episode_successes" not in r:
                continue
            pol = (r.get("provenance", {}).get("checkpoint", {}) or {}).get("policy", "?")
            base, why = recorded_for(pol)
            # The arithmetic, spelled out. Two official rows that differ only in precision or in
            # how many envs the solver plans for at once are eight points apart, and a table that
            # cannot tell them apart is the table that makes somebody trust the faster one.
            cfg = r.get("config") or {}
            bs = ((cfg.get("solver") or {}).get("batch_size"))
            bits = [f"tf32" if r.get("tf32") else "fp32"]
            if bs is not None:
                bits.append(f"bs{bs}")
            precision = " ".join(bits) if pol != "random" else ""
            rows.append({
                "policy": pol,
                "precision": precision,
                "arm": r["arm"] + (f" ({r['renderer_target_mode']})"
                                   if r.get("renderer_target_mode") else ""),
                "seed": r["seed"],
                "successes": int(sum(r["episode_successes"])),
                "n": len(r["episode_successes"]),
                "rate": float(r["success_rate"]),
                "seconds": round(r.get("eval_seconds", 0), 1),
                "reference": f"{base} ({why})" if base else why,
                "source": str(p),
            })
    rows.sort(key=lambda r: (r["policy"], r["arm"], r["seed"]))
    out = {"note": args.note, "runs": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    for r in rows:
        print(f"{r['policy']:28s} {r.get('precision',''):9s} {r['arm']:24s} seed {r['seed']}  "
              f"{r['successes']:3d}/{r['n']:<3d} = {r['rate']:5.1f} %  "
              f"({r['seconds']:.0f}s)  recorded: {r['reference']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
