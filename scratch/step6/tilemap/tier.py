"""Label one re-measured Step-3 row before and after, using **Step 3's own evaluator**.

``scratch/step3/certify.py`` already implements the frozen Step-2 ladder, including the two rules
that are easy to get wrong by hand: the ratio picks the rung and a failed ``cores`` conjunct walks
DOWN from it (so ``BELOW_G4`` caps rather than unlabels), and ``BATCH_SCALES`` caps a GPU row at
``GPU_PARITY`` no matter how large ``R`` is. Reimplementing that here would be a second evaluator
with a second set of bugs, so this imports it and calls ``classify`` on one run.

What is new is only the plumbing: ``certify.main`` writes into ``examples/results/step3/``, which
Step 6 must not touch, and it certifies a whole shard directory. This takes the two shards a
workstream produced in one job and prints the pair.

    python scratch/step6/tilemap/tier.py --device cpu --env CraftaxClassic \
        --before sweep/cpu_before.json --after sweep/cpu_after.json --out sweep/tier_cpu.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load_certify():
    """Import ``scratch/step3/certify.py`` as a module without putting the repo root on the path."""
    spec = importlib.util.spec_from_file_location(
        "step3_certify", ROOT / "scratch" / "step3" / "certify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_for(path: Path, env: str) -> dict:
    """One env's run out of a ``bench_step3`` shard, in the shape ``classify`` expects."""
    d = json.loads(Path(path).read_text())
    for e in d.get("envs", []):
        if e.get("env") == env:
            e = dict(e)
            e["_provenance"] = d.get("provenance", {})
            return e
    raise SystemExit(f"{env} is not in {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="CraftaxClassic")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    C = load_certify()
    census = C.read_censuses(sorted((ROOT / "scratch" / "census").glob("census_*.json")))
    res = {"env": a.env, "device": a.device,
           "evaluator": "scratch/step3/certify.py:classify",
           "spec": "scratch/certification.json (frozen in Step 2, not read by this file)"}
    for leg, path in (("before", a.before), ("after", a.after)):
        run = run_for(path, a.env)
        lab = C.classify(run, a.device, census)
        lab["shard"] = str(path)
        lab["per_batch_fps"] = {str(c["batch"]): c.get("fps")
                                for c in run.get("cells", []) if c.get("fps")}
        res[leg] = lab
    for k in ("tier", "qualifiers"):
        res.setdefault("delta", {})[k] = [res["before"].get(k), res["after"].get(k)]
    print(json.dumps(res, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
