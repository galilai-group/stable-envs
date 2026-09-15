"""Label this env three ways with Step 3's OWN evaluator, because the reference decides the label.

``reference_env`` returns craftax **un-jitted**, which parity requires (the jitted renderer disagrees
with itself in uint8). ``bench_step3`` then measures ``official_cpu`` on that un-jitted function and
the Step-3 ladder divides by it, so the speed ratio is taken against a renderer nobody runs.

How much that matters is not a matter of opinion. On one 3090, in one allocation, in one job:

    Step-6 tree   1,174,913 f/s   official 386.7 f/s (jitted)      R = 3,038    GPU_FAST
    this branch      46,187 f/s   official   1.6 f/s (un-jitted)   R = 29,214   GPU_EXCEPTIONAL

**The branch is 25x slower on the same card and labels one rung higher.** Nothing about the renderer
causes that; it is entirely which reference each leg divided by.

So all three labelings are produced and recorded together:

  as_measured    what the harness produces today -- ratio against the un-jitted reference
  vs_jitted      the honest speed comparison -- ratio against the renderer a user runs
  no_reference   cores only, ignoring the ratio entirely

Nothing here edits the shared harness or the frozen spec; it calls ``classify`` on the same records
three times with the reference varied, and prints what each says.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "examples/results/step7/craftax-classic-full/sweep"


def load_certify():
    spec = importlib.util.spec_from_file_location(
        "step3_certify", ROOT / "scratch" / "step3" / "certify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_for(path: Path, env: str) -> dict:
    d = json.loads(Path(path).read_text())
    for e in d.get("envs", []):
        if e.get("env") == env:
            e = dict(e)
            e["_provenance"] = d.get("provenance", {})
            return e
    raise SystemExit(f"{env} is not in {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="CraftaxClassic")
    ap.add_argument("--device", required=True, choices=["cpu", "cuda"])
    ap.add_argument("--branch", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    a = ap.parse_args()

    C = load_certify()
    census = C.read_censuses(sorted((ROOT / "scratch" / "census").glob("census_*.json")))
    branch, base = run_for(a.branch, a.env), run_for(a.baseline, a.env)
    jitted = (base.get("official_cpu") or {}).get("set_state_render", {}).get("fps")

    variants = {}
    # 1. as the harness measures it today
    variants["as_measured"] = copy.deepcopy(branch)
    # 2. against the reference a user actually runs, taken from the BASELINE leg of the same job --
    #    same card, same node, same interpreter, and jitted because that tree's reference_env jits
    v = copy.deepcopy(branch)
    v["official_cpu"] = copy.deepcopy(base["official_cpu"])
    variants["vs_jitted"] = v
    # 3. no ratio at all
    v = copy.deepcopy(branch)
    v.pop("official_cpu", None)
    variants["no_reference"] = v

    res = {"env": a.env, "device": a.device,
           "evaluator": "scratch/step3/certify.py:classify (frozen Step-2 ladder, not restated here)",
           "jitted_official_fps": jitted,
           "unjitted_official_fps": (branch.get("official_cpu") or {})
           .get("set_state_render", {}).get("fps"),
           "baseline_label": C.classify(base, a.device, census)["tier"],
           "baseline_fps": C.best_certified(base, a.device)[1],
           "labels": {}}
    for name, run in variants.items():
        c = C.classify(run, a.device, census)
        res["labels"][name] = {"tier": c["tier"], "qualifiers": c["qualifiers"],
                               "ratio": c["evidence"].get("ratio"),
                               "cores": c["evidence"].get("cores"),
                               "fps": c["evidence"].get("se_fps"),
                               "best_batch": c["evidence"].get("best_batch")}
    res["note"] = ("as_measured is what the harness produces and it is NOT a defensible performance "
                   "claim: the ratio is against a renderer nobody runs. Which of vs_jitted and "
                   "no_reference is correct is Sami's ruling; both are recorded so it can be made "
                   "on numbers.")
    p = OUT / f"tier_three_ways_{a.device}.json"
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
