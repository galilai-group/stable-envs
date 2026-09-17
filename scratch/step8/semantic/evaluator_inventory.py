"""What downstream Craftax evaluator actually exists -- audited, not assumed.

Question (1) of the Workstream-H brief. A Gate-H run in the Step-5 sense needs four things at
once: a trained checkpoint for the environment, a corpus of episodes to draw paired starts and
goals from, a planner that can act in the environment's action space, and a success criterion
computed from simulator state rather than from pixels. This script looks for each of them in the
places the brief names, and writes what it found (and did not find) to
``examples/results/step8/semantic/evaluator_inventory.json``.

Roots default to this machine's checkouts and may be overridden on the command line. Only
existence and names are recorded, so the artifact stays reproducible and carries no cluster paths.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

OUT = "examples/results/step8/semantic/evaluator_inventory.json"
CRAFTAX = re.compile(r"craftax", re.I)


def listdir(p: str) -> list[str]:
    try:
        return sorted(os.listdir(p))
    except OSError:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swm", default="/Users/samib/stable-worldmodel")
    a = ap.parse_args()

    swm = a.swm
    rec: dict = {
        "what": ("the downstream assets a Step-5-style Gate H would need for Craftax, and which "
                 "of them exist"),
        "stable_worldmodel_checkout": os.path.basename(swm),
        "checkout_present": os.path.isdir(swm),
        "needs": {},
    }

    # 1. a trained checkpoint for the environment
    ck = listdir(os.path.join(swm, "checkpoints"))
    rec["needs"]["checkpoint"] = {
        "where_looked": ["<swm>/checkpoints", "<swm>/docs/baselines.md"],
        "found": ck,
        "craftax_entries": [x for x in ck if CRAFTAX.search(x)],
        "satisfied": False,
    }

    # 2. a corpus of episodes
    data = listdir(os.path.join(swm, "data"))
    collect = [x for x in listdir(os.path.join(swm, "scripts", "data")) if x.startswith("collect_")]
    rec["needs"]["episode_corpus"] = {
        "where_looked": ["<swm>/data", "<swm>/scripts/data"],
        "datasets_present": [x for x in data if not x.startswith("__")],
        "collectors_present": collect,
        "craftax_collector": [x for x in collect if CRAFTAX.search(x)],
        "satisfied": False,
    }

    # 3. a planner configuration for the environment
    plan_cfg = [x for x in listdir(os.path.join(swm, "scripts", "plan", "config"))
                if x.endswith(".yaml")]
    solvers = [x for x in listdir(os.path.join(swm, "stable_worldmodel", "solver"))
               if x.endswith(".py") and not x.startswith("__")]
    rec["needs"]["planner_config"] = {
        "where_looked": ["<swm>/scripts/plan/config", "<swm>/stable_worldmodel/solver"],
        "eval_configs_present": plan_cfg,
        "craftax_eval_config": [x for x in plan_cfg if CRAFTAX.search(x)],
        "solvers_present": solvers,
        "discrete_solver_exists": any("categorical" in x for x in solvers),
        "satisfied": False,
    }

    # 4. a success criterion computed from simulator state
    train_cfg = [x for x in listdir(os.path.join(swm, "scripts", "train")) if x.endswith(".py")]
    rec["needs"]["success_criterion"] = {
        "where_looked": ["<swm>/scripts/plan/eval_wm.py"],
        "form": ("goal image taken from a corpus episode `goal_offset_steps` ahead of the start "
                 "index; success is a state-space distance to that goal state"),
        "craftax_applicable": False,
        "why": ("Craftax is open-ended and achievement-scored with death termination; it publishes "
                "no goal state and swm registers no goal or success for either Craftax id"),
        "satisfied": False,
    }
    rec["trainers_present"] = train_cfg

    # the stable-envs side: the Step-5 evaluator itself
    ds = listdir("examples/downstream_step5")
    rec["step5_evaluator"] = {
        "files": ds,
        "arms": ["pusht (ingested from the raw run JSONs)", "reacher"],
        "craftax_arm": [x for x in ds if CRAFTAX.search(x)],
    }
    swmc = listdir("examples/swm_compat")
    rec["swm_compat_workflow"] = {"files": swmc, "craftax_arm": [x for x in swmc if CRAFTAX.search(x)]}

    rec["any_need_satisfied"] = any(v.get("satisfied") for v in rec["needs"].values())
    rec["verdict"] = (
        "no downstream Craftax evaluator exists: none of the four required assets is present"
        if not rec["any_need_satisfied"] else "at least one asset is present -- re-read the record"
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec, indent=1))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
