"""Union the per-interpreter censuses into the one number the README is allowed to quote.

No interpreter registers every family (robocasa pins mujoco 3.3.1, mujoco_warp needs >= 3.11, the JAX
families bring their own CUDA), so "how many envs" has no single-process answer. This prints the union,
the per-interpreter counts that compose it, and — the part worth keeping — which envs only ONE
interpreter can reach, since those are the ones a reader will fail to import and report as missing.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict

per: dict[str, list[str]] = {}
# sorted(), not bare glob(): an earlier run's log holds the same interpreter names with no registry
# (that run had no PYTHONPATH, so every import failed), and reading the logs in filesystem order let
# those entries decide which names were considered at all. The union came out 191 instead of 196, and
# the five it dropped were whole families — Overcooked among them.
for path in sys.argv[1:] or sorted(glob.glob(
        "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/census/*.log")):
    name = None
    for line in open(path):
        line = line.strip()
        if line.startswith("###"):
            name = line[3:].strip()
        elif line.startswith("CENSUS ") and name:
            d = json.loads(line[7:])
            if "registry" in d:
                per[name] = d["registry"]
            name = None

union = sorted({e for r in per.values() for e in r})
owners = defaultdict(list)
for k, r in per.items():
    for e in r:
        owners[e].append(k)

print(f"UNION {len(union)} envs across {len(per)} interpreters\n")
for k in sorted(per, key=lambda k: -len(per[k])):
    print(f"  {len(per[k]):4d}  {k}")

solo = sorted(e for e in union if len(owners[e]) == 1)
print(f"\nreachable from exactly ONE interpreter: {len(solo)}")
by_owner = defaultdict(list)
for e in solo:
    by_owner[owners[e][0]].append(e)
for k in sorted(by_owner, key=lambda k: -len(by_owner[k])):
    print(f"  {k}: {len(by_owner[k])} — {', '.join(by_owner[k][:6])}"
          f"{' ...' if len(by_owner[k]) > 6 else ''}")

best = max(per, key=lambda k: len(per[k]))
print(f"\nwidest single interpreter: {best} at {len(per[best])} "
      f"({len(union) - len(per[best])} envs it cannot reach)")
json.dump({"union": union, "per_interpreter": {k: sorted(v) for k, v in per.items()}},
          open("/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/census/union.json", "w"),
          indent=2)
