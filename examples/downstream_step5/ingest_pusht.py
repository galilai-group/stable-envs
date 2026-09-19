"""Audit the existing PushT + LeWM renderer swap and put it in the Step 5 format.

Step 5 section 2 says to trace the existing 300-pair result to its checkpoint, config, seeds and
episode selection and, if it holds up, ingest it rather than burn 300 episodes reproducing a table
we already have. This script is that audit. It reads only the RAW per-run JSONs under
``examples/lewm_pusht/results/`` -- never the prose -- and recomputes every number from the
per-episode outcomes, so a disagreement between the artifact and the write-up surfaces as a
finding instead of being inherited.

What "paired" is checked to mean here: for each seed, every arm evaluated the SAME episode indices
at the SAME start steps with the same goal offset, budget, checkpoint and planner seed. That is the
whole basis for using McNemar at all; if it does not hold, the discordant counts are comparing
different tasks and the interval around them means nothing.

    python examples/downstream_step5/ingest_pusht.py --out examples/results/step5/pusht
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "examples" / "lewm_pusht" / "results"

#: The arm that carries the claim, and the two that split it by pixel surface. ``stable_envs``
#: replaces every image the checkpoint ever sees; the other two exist to say WHERE any movement
#: came from, and are reported but do not decide the gate.
PRIMARY = "stable_envs"
CONTROL = "blank"

#: Fields that must agree across arms for a pair to be a pair. ``policy`` is here because a
#: comparison between two checkpoints is not a renderer experiment.
PAIRING_FIELDS = ("seed", "goal_offset", "eval_budget", "policy", "num_eval")

RUN_RE = re.compile(r"^seed(?P<seed>\d+)_n(?P<n>\d+)_off(?P<off>\d+)_(?P<arm>.+)\.json$")


def load_runs(results: Path) -> list[dict]:
    """One record per run file, with the filename's own claims checked against the contents."""
    runs = []
    for p in sorted(results.glob("seed*_n*_off*_*.json")):
        m = RUN_RE.match(p.name)
        if not m:
            continue
        d = json.loads(p.read_text())
        seed, off, arm = int(m["seed"]), int(m["off"]), m["arm"]
        # The filename is metadata too, and metadata that disagrees with the payload is a reason to
        # trust neither. Cheap to check, and it is the sort of thing a hand-renamed file gets wrong.
        assert d["seed"] == seed, f"{p.name}: filename says seed {seed}, payload says {d['seed']}"
        assert d["goal_offset"] == off, f"{p.name}: filename says offset {off}, payload {d['goal_offset']}"
        assert d["renderer"] == arm, f"{p.name}: filename says {arm}, payload says {d['renderer']}"
        assert len(d["episode_successes"]) == d["num_eval"] == len(d["episodes_idx"]) == len(d["start_steps"])
        runs.append({"path": p, "arm": arm, **d})
    return runs


def check_pairing(runs: list[dict]) -> dict:
    """Every arm at a given seed must have drawn the same episodes at the same start steps."""
    by_seed: dict[int, list[dict]] = {}
    for r in runs:
        by_seed.setdefault(r["seed"], []).append(r)

    problems, checked = [], []
    for seed, group in sorted(by_seed.items()):
        ref = next((g for g in group if g["arm"] == "official"), None)
        if ref is None:
            problems.append(f"seed {seed}: no official arm to pair against")
            continue
        for g in group:
            if g["arm"] == "official":
                continue
            if g["episodes_idx"] != ref["episodes_idx"]:
                problems.append(f"seed {seed} {g['arm']}: different episode indices from official")
            if g["start_steps"] != ref["start_steps"]:
                problems.append(f"seed {seed} {g['arm']}: different start steps from official")
            for f in PAIRING_FIELDS:
                if g[f] != ref[f]:
                    problems.append(f"seed {seed} {g['arm']}: {f}={g[f]!r} vs official {ref[f]!r}")
            checked.append((seed, g["arm"]))
        # A repeated (episode, start) inside one run would make two rows of the table the same task.
        keys = list(zip(ref["episodes_idx"], ref["start_steps"]))
        if len(set(keys)) != len(keys):
            problems.append(f"seed {seed}: {len(keys) - len(set(keys))} duplicated (episode, start) pairs")
    return {"pairs_checked": len(checked), "problems": problems, "paired": not problems}


def episodes(runs: list[dict]) -> list[dict]:
    """The Step 5 episode-level record: one row per (arm, seed, episode), nothing aggregated."""
    rows = []
    for r in runs:
        for i, (ep, st, ok) in enumerate(zip(r["episodes_idx"], r["start_steps"], r["episode_successes"])):
            rows.append(
                {
                    "env": "swm/PushT-v1",
                    "model_family": "lewm",
                    "checkpoint": r["policy"],
                    "planner": "cem",
                    "arm": r["arm"],
                    "seed": r["seed"],
                    "episode_id": int(ep),
                    "start_step": int(st),
                    "goal_offset": r["goal_offset"],
                    "eval_budget": r["eval_budget"],
                    "slot": i,
                    "success": int(ok),
                    "source": str(r["path"].relative_to(ROOT)),
                }
            )
    return rows


def series(rows: list[dict], arm: str, seeds: list[int] | None = None) -> tuple[list[int], list[int], list[int]]:
    """Outcomes for one arm in a fixed order, plus the seed of each -- the order IS the pairing."""
    sel = [r for r in rows if r["arm"] == arm and (seeds is None or r["seed"] in seeds)]
    sel.sort(key=lambda r: (r["seed"], r["slot"]))
    return ([r["success"] for r in sel], [r["seed"] for r in sel], [r["episode_id"] for r in sel])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RESULTS, help="raw lewm_pusht run JSONs")
    ap.add_argument("--out", type=Path, default=ROOT / "examples" / "results" / "step5" / "pusht")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import step5_stats as S

    runs = load_runs(args.results)
    if not runs:
        raise SystemExit(f"no run files under {args.results}")
    pairing = check_pairing(runs)
    rows = episodes(runs)

    arms = sorted({r["arm"] for r in rows})
    off_all, seeds_all, eps_all = series(rows, "official")
    ours_all, ours_seeds, ours_eps = series(rows, PRIMARY)
    # Pair on identity, not on position alone: same seed and same episode id, row for row.
    assert (seeds_all, eps_all) == (ours_seeds, ours_eps), "official and stable_envs rows are not aligned"

    ctrl_seeds = sorted({r["seed"] for r in rows if r["arm"] == CONTROL})
    ctrl, cseeds, ceps = series(rows, CONTROL)
    off_c, ocseeds, oceps = series(rows, "official", seeds=ctrl_seeds)
    assert (cseeds, ceps) == (ocseeds, oceps), "control arm is not aligned with official"

    per_seed_n = [sum(1 for s in seeds_all if s == sd) for sd in sorted(set(seeds_all))]
    gate = S.gate_h(off_all, ours_all, ctrl, seeds=seeds_all, per_seed_n=per_seed_n)
    # The control is measured on its own seed only; say so rather than letting a reader assume 300.
    gate["control"]["measured_on_seeds"] = ctrl_seeds
    gate["control"]["official_baseline_on_those_seeds"] = S.arm(off_c)

    secondary = {}
    for a in arms:
        if a in {"official", PRIMARY, CONTROL}:
            continue
        sa, ss, se = series(rows, a)
        oa, os_, oe = series(rows, "official", seeds=sorted(set(ss)))
        if (ss, se) != (os_, oe):
            secondary[a] = {"error": "not aligned with official"}
            continue
        secondary[a] = {"arm": S.arm(sa), "paired": S.paired(oa, sa), "seeds": sorted(set(ss))}

    out = {
        "env": "swm/PushT-v1",
        "model_family": "lewm",
        "checkpoint": runs[0]["policy"],
        "arms": {a: len([r for r in rows if r["arm"] == a]) for a in arms},
        "pairing": pairing,
        "gate_h": gate,
        "secondary_arms": secondary,
        "provenance": provenance(runs),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    jl = args.out / "episodes.jsonl"
    jl.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out["episodes_file"] = str(jl.relative_to(ROOT))
    out["episodes_sha256"] = hashlib.sha256(jl.read_bytes()).hexdigest()[:12]
    (args.out / "gate_h.json").write_text(json.dumps(out, indent=2))

    g = gate
    print(f"pairing: {'OK' if pairing['paired'] else 'PROBLEMS'} ({pairing['pairs_checked']} arm-seeds checked)")
    for p in pairing["problems"]:
        print("  !", p)
    print(f"official    {g['official']['successes']}/{g['official']['n']} = {g['official']['success_rate']:.1f}%")
    print(
        f"stable-envs {g['stable_envs']['successes']}/{g['stable_envs']['n']} = {g['stable_envs']['success_rate']:.1f}%"
    )
    e = g["equivalence"]
    print(
        f"paired      b={e['b']} c={e['c']} delta={e['delta_points']:+.2f} pts "
        f"CI [{e['ci'][0]:.2f}, {e['ci'][1]:.2f}] McNemar p={e['mcnemar_p']:.3f}"
    )
    c = g["control"]
    print(
        f"control     {c['arm']['successes']}/{c['arm']['n']} delta={c['paired']['delta_points']:+.1f} pts "
        f"p={c['paired']['mcnemar_p']:.2e} degraded={c['degraded']}"
    )
    print(f"VERDICT     {g['verdict']} -- {g['verdict_reason']}")
    print(f"wrote {jl.relative_to(ROOT)} and {(args.out / 'gate_h.json').relative_to(ROOT)}")


def provenance(runs: list[dict]) -> dict:
    """What the artifact records, and -- as loudly -- what it does not.

    Step 5 section 18 asks for git commits, dependency versions and a GPU beside every downstream
    number. These runs predate that requirement and were executed on a cluster this checkout cannot
    reach, so those fields cannot be recovered by rerunning something. Listing them as missing is
    the honest option; silently omitting the requirement is not.
    """
    have = {
        "checkpoint": sorted({r["policy"] for r in runs}),
        "seeds": sorted({r["seed"] for r in runs}),
        "goal_offset": sorted({r["goal_offset"] for r in runs}),
        "eval_budget": sorted({r["eval_budget"] for r in runs}),
        "episodes_per_run": sorted({r["num_eval"] for r in runs}),
        "runtime_seconds": {r["path"].name: round(r["eval_seconds"], 1) for r in runs},
        "episode_ids_and_start_steps": "recorded per run",
    }
    return {
        "recorded_in_artifact": have,
        "recorded_in_readme_only": [
            "checkpoint source quentinll/lewm-pusht at ~/.stable_worldmodel/checkpoints/pusht/lewm/",
            "dataset pusht_expert_train.h5 (quentinll/lewm-pusht)",
            "planner CEM 300 samples x 30 iterations, top 30, horizon 5, receding 5, action block 5",
            "image size 224, ImageNet normalisation, StandardScaler fitted over 2.34M corpus rows",
            "success criterion PushT.eval_state: within 20 px and pi/9",
        ],
        "not_recorded": [
            "stable-envs git commit at run time",
            "stable-worldmodel git commit at run time",
            "checkpoint file hash",
            "dependency versions",
            "GPU model",
        ],
        "why_not_recovered": (
            "these runs were executed on the open_research cluster referenced by "
            "examples/lewm_pusht/run_eval.sbatch, which this checkout has no access to; the "
            "per-episode outcomes are committed and are what the audit reads"
        ),
    }


if __name__ == "__main__":
    main()
