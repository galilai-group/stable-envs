"""Turn a directory of renderer-arm runs into episode-level records and the frozen Step 2 verdict.

Two jobs, and the first is the one that makes the second mean anything.

**Prove the arms are comparable.** Section 18 of the brief asks that the official and stable-envs
arms be demonstrably identical on every field except the renderer. Each run records its resolved
config, the checkpoint's file hashes, both git commits and its dependency versions, so "identical"
is a diff this script performs rather than a claim somebody makes. Anything that differs and is not
the renderer is reported as a discrepancy, and a discrepancy in the checkpoint or the planner is
fatal: the paired test would then be attributing a model difference to the pixels.

**Then compute the gate.** ``step5_stats.gate_h`` applies the frozen ``downstream`` policy from
``scratch/certification.json`` -- margin, CI formula, control thresholds and power guard all come
from there.

    python examples/downstream_step5/paired_gate.py --runs examples/results/step5/reacher/runs \
        --out examples/results/step5/reacher
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Config keys that are ALLOWED to differ between arms, because they are the experiment's own
#: bookkeeping rather than part of what is being held fixed.
# NOTE: "tf32" is deliberately NOT here. Precision is part of the experiment: if one arm ran with
# TF32 matmul and another without, they are not the same computation and the paired test is void.
ALLOWED_DIFF = (
    "arm",
    "renderer_target_mode",
    "renderer_dataset_target_mode",
    "renderer_surfaces",
    "eval_seconds",
    "render_calls",
    "success_rate",
    "episode_successes",
    "provenance.host",
    "provenance.gpu",
    "provenance.slurm_job",
)


def flat(d, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        p = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, p + "."))
        else:
            out[p] = json.dumps(v, sort_keys=True) if isinstance(v, list) else v
    return out


def load(runs_dir: Path) -> list[dict]:
    out = []
    for p in sorted(runs_dir.glob("*.json")):
        d = json.loads(p.read_text())
        if "episode_successes" not in d:
            continue
        d["_path"] = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        out.append(d)
    return out


def compare_arms(runs: list[dict]) -> dict:
    """Every arm at a seed must agree on episodes, starts, config, checkpoint and versions."""
    by_seed: dict[int, list[dict]] = {}
    for r in runs:
        by_seed.setdefault(int(r["seed"]), []).append(r)

    problems, notes, checked = [], [], 0
    for seed, group in sorted(by_seed.items()):
        ref = next((g for g in group if g["arm"] == "official"), None)
        if ref is None:
            problems.append(f"seed {seed}: no official arm")
            continue
        rf = flat(
            {
                "config": ref.get("config"),
                "provenance": ref.get("provenance"),
                "goal_offset": ref.get("goal_offset"),
                "eval_budget": ref.get("eval_budget"),
                "task": ref.get("task"),
                "num_eval": ref.get("num_eval"),
            }
        )
        for g in group:
            if g["arm"] == "official":
                continue
            checked += 1
            if g["episodes_idx"] != ref["episodes_idx"]:
                problems.append(f"seed {seed} {g['arm']}: different episode indices")
            if g["start_steps"] != ref["start_steps"]:
                problems.append(f"seed {seed} {g['arm']}: different start steps")
            gf = flat(
                {
                    "config": g.get("config"),
                    "provenance": g.get("provenance"),
                    "goal_offset": g.get("goal_offset"),
                    "eval_budget": g.get("eval_budget"),
                    "task": g.get("task"),
                    "num_eval": g.get("num_eval"),
                }
            )
            for k in sorted(set(rf) | set(gf)):
                if any(k.endswith(a) or k == a for a in ALLOWED_DIFF):
                    continue
                if rf.get(k) != gf.get(k):
                    msg = f"seed {seed} {g['arm']}: {k} = {gf.get(k)!r} vs official {rf.get(k)!r}"
                    # A checkpoint or planner difference is not a note, it is the experiment failing.
                    (
                        problems
                        if any(t in k for t in ("checkpoint", "solver", "plan_config", "policy", "dataset", "eval."))
                        else notes
                    ).append(msg)
    return {"arm_runs_checked": checked, "problems": problems, "benign_differences": notes, "comparable": not problems}


def arm_name(r: dict) -> str:
    """``stable_envs`` swaps every pixel surface; a partial swap is a DIFFERENT arm.

    Keeping the name distinct is what stops a surface-split diagnostic run from being silently
    averaged into the arm that carries the gate.
    """
    sfc = r.get("renderer_surfaces")
    return r["arm"] if sfc in (None, "all") else f"{r['arm']}_{sfc}"


def episodes(runs: list[dict]) -> list[dict]:
    rows = []
    for r in runs:
        for i, (ep, st, ok) in enumerate(zip(r["episodes_idx"], r["start_steps"], r["episode_successes"])):
            rows.append(
                {
                    "env": r["env"],
                    "task": r.get("task"),
                    "model_family": r.get("model_family"),
                    "checkpoint": (r.get("provenance", {}).get("checkpoint", {}) or {}).get("policy"),
                    "planner": "cem",
                    "arm": arm_name(r),
                    "seed": int(r["seed"]),
                    "episode_id": int(ep),
                    "start_step": int(st),
                    "goal_offset": r.get("goal_offset"),
                    "eval_budget": r.get("eval_budget"),
                    "slot": i,
                    "success": int(ok),
                    "source": r["_path"],
                }
            )
    return rows


def series(rows, arm, seeds=None):
    sel = [r for r in rows if r["arm"] == arm and (seeds is None or r["seed"] in seeds)]
    sel.sort(key=lambda r: (r["seed"], r["slot"]))
    return ([r["success"] for r in sel], [r["seed"] for r in sel], [r["episode_id"] for r in sel])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--control", default="blank")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import step5_stats as S

    runs = load(args.runs)
    if not runs:
        raise SystemExit(f"no run JSONs under {args.runs}")
    comp = compare_arms(runs)
    rows = episodes(runs)

    off, seeds, eps = series(rows, "official")
    ours, oseeds, oeps = series(rows, "stable_envs")
    if (seeds, eps) != (oseeds, oeps):
        raise SystemExit("official and stable_envs episodes are not aligned; refusing to pair them")

    cseeds = sorted({r["seed"] for r in rows if r["arm"] == args.control})
    ctrl, cs, ce = series(rows, args.control)
    offc, ocs, oce = series(rows, "official", seeds=cseeds)
    if ctrl and (cs, ce) != (ocs, oce):
        raise SystemExit("control arm is not aligned with official")

    per_seed_n = [sum(1 for s in seeds if s == sd) for sd in sorted(set(seeds))]
    gate = S.gate_h(off, ours, ctrl or None, seeds=seeds, per_seed_n=per_seed_n)
    if ctrl:
        gate["control"]["measured_on_seeds"] = cseeds
        gate["control"]["official_baseline_on_those_seeds"] = S.arm(offc)

    # Secondary arms: a partial pixel swap, or anything else that ran beside the gate. Reported
    # with its own paired statistics and never folded into the verdict.
    secondary = {}
    for a in sorted({r["arm"] for r in rows} - {"official", "stable_envs", args.control}):
        sa, ss, se = series(rows, a)
        oa, os_, oe = series(rows, "official", seeds=sorted(set(ss)))
        secondary[a] = (
            {"error": "not aligned with official"}
            if (ss, se) != (os_, oe)
            else {"arm": S.arm(sa), "paired": S.paired(oa, sa), "seeds": sorted(set(ss))}
        )

    # THE RENDERER ARM, gated in its own right.
    #
    # `stable_envs` swaps all three pixel surfaces, and on this environment that changes more than
    # render(): two of the three are STORED corpus frames, so swapping them also replaces
    # months-old bytes with freshly rendered ones. The official_regen control measures that
    # substitution with MuJoCo doing the rendering and finds it costs about the same, so the
    # all-surface arm is not a clean test of the rasteriser. The live-observation arm is, and it
    # gets the same frozen rule applied to it rather than being demoted to a footnote.
    gate_live = None
    xl, sl, el = series(rows, "stable_envs_live")
    if xl:
        seeds_l = sorted({s for s in sl})
        ol, osl, oel = series(rows, "official", seeds=seeds_l)
        if (sl, el) == (osl, oel):
            cl, cs2, ce2 = series(rows, args.control, seeds=seeds_l)
            oc2, ocs2, oce2 = series(rows, "official", seeds=sorted({s for s in cs2}))
            gate_live = S.gate_h(
                ol,
                xl,
                cl if (cs2, ce2) == (ocs2, oce2) else None,
                seeds=sl,
                per_seed_n=[sl.count(sd) for sd in seeds_l],
            )

    out = {
        "env": runs[0]["env"],
        "task": runs[0].get("task"),
        "gate_h_renderer_arm": gate_live,
        "model_family": runs[0].get("model_family"),
        "arms": {a: sum(1 for r in rows if r["arm"] == a) for a in sorted({r["arm"] for r in rows})},
        "comparability": comp,
        "gate_h": gate,
        "secondary_arms": secondary,
        "runs": [
            {
                "path": r["_path"],
                "arm": r["arm"],
                "seed": r["seed"],
                "success_rate": r["success_rate"],
                "seconds": round(r.get("eval_seconds", 0), 1),
                "render_calls": r.get("render_calls"),
            }
            for r in runs
        ],
        "provenance": runs[0].get("provenance"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    jl = args.out / "episodes.jsonl"
    jl.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out["episodes_file"] = str(jl.relative_to(ROOT)) if jl.is_relative_to(ROOT) else str(jl)
    out["episodes_sha256"] = hashlib.sha256(jl.read_bytes()).hexdigest()[:12]
    (args.out / "gate_h.json").write_text(json.dumps(out, indent=2))

    print(f"comparability: {'OK' if comp['comparable'] else 'PROBLEMS'} ({comp['arm_runs_checked']} arm-runs checked)")
    for p in comp["problems"]:
        print("  !", p)
    for n in comp["benign_differences"][:10]:
        print("  ~", n)
    g = gate
    print(f"official    {g['official']['successes']}/{g['official']['n']} = {g['official']['success_rate']:.1f}%")
    print(
        f"stable-envs {g['stable_envs']['successes']}/{g['stable_envs']['n']} = {g['stable_envs']['success_rate']:.1f}%"
    )
    e = g["equivalence"]
    print(
        f"paired      b={e['b']} c={e['c']} delta={e['delta_points']:+.2f} pts "
        f"CI [{e['ci'][0]:.2f}, {e['ci'][1]:.2f}] McNemar p={e['mcnemar_p']:.3f}"
    )
    if g["control"]["present"]:
        c = g["control"]
        print(
            f"control     {c['arm']['successes']}/{c['arm']['n']} "
            f"delta={c['paired']['delta_points']:+.1f} pts p={c['paired']['mcnemar_p']:.2e} "
            f"degraded={c['degraded']}"
        )
    print(f"VERDICT     {g['verdict']} -- {g['verdict_reason']}")


if __name__ == "__main__":
    main()
