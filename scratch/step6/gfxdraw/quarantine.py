"""Mark the shards under this directory that must not label a tier, idempotently.

`scratch/step6/report.py` runs `certify.classify` over every Step-3-shaped shard it finds, and a
shard that is unusable, that describes superseded code, or that is one arm of a before/after pair
has to be out of its reach. Three ways of doing that were tried and only the third works:

* **Prose beside the shard** does not stop a generator reading it. That is how `CPU_FASTER` reached
  the Step 6 page for four configurations whose own findings file says `CPU_PARITY`.
* **A directory name** (`superseded_*/`) only works if the generator happens not to recurse, which
  nobody has promised, and it does not survive `rsync` -- the file still exists at its original path
  on the cluster, so the next pull recreates the top-level copy beside the quarantined one.
* **The shape of the file.** `certify.merge` reads `d.get("envs", [])`, so moving the rows to
  `envs_not_for_certification` and leaving `envs: []` contributes no runs from anywhere in the tree,
  keeps the whole file readable, and is re-applied by running this again after a pull.

Run after every `pull_artifacts.sh`; it is a no-op on a file already marked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[3] / "examples/results/step6/gfxdraw"

EXCLUDED = {
    "sweep/superseded_node2338/cpu_1thread_before_node2338.json":
        "block-timed denominator on a node at load average 47; see the README beside it",
    "sweep/superseded_node2338/cpu_1thread_after_node2338.json":
        "block-timed denominator that collapsed 2.3x within one job; see the README beside it",
    "sweep/rejected_high_spread_node2339/cpu_1thread_paired_node2339.json":
        "the reference varied 4.52x-10.37x across the interleaved readings; certifiable=false",
    "sweep/wave1_pre_gfxdraw2/gpu_before_gpu2104.json":
        "the pre-Step-6 baseline, and one arm of a pair that collides in merge's key",
    "sweep/wave1_pre_gfxdraw2/gpu_after_gpu2104.json":
        "superseded code (the planar-canvas patch before GFXDRAW-2), and one arm of a colliding pair",
    "sweep2/superseded_gpu2110/gpu_before_gpu2110.json":
        "block-timed denominator; see the README beside it",
    "sweep2/superseded_gpu2110/gpu_after_gpu2110.json":
        "block-timed denominator that would publish GPU_FAST for two rows; see the README",
    "sweep2/gpu_paired_before_gpu2264.json":
        "the kernel-off comparison arm. certify.merge keys on (env, device, host, threads) with no "
        "slot for an arm, so a before/after pair from one host collides and one silently supersedes "
        "the other -- on a tie, by path order. Only the arm reflecting the current tree is a "
        "certification row.",
}

HOW = ("certify.merge reads d.get('envs', []), so an empty list contributes no runs and no tier. "
       "The rows are kept under 'envs_not_for_certification'. Structural, so it holds wherever the "
       "file sits and survives a pull that recreates it.")


def main() -> int:
    changed = missing = 0
    for rel, why in sorted(EXCLUDED.items()):
        p = BASE / rel
        if not p.is_file():
            print(f"  MISSING {rel}")
            missing += 1
            continue
        d = json.loads(p.read_text())
        if d.get("envs") == [] and d.get("excluded_from_certification"):
            continue
        if isinstance(d.get("envs"), list) and d["envs"]:
            d["envs_not_for_certification"] = d.pop("envs")
        d["envs"] = []
        d["excluded_from_certification"] = why
        d["how_this_exclusion_works"] = HOW
        p.write_text(json.dumps(d, indent=1))
        print(f"  excluded {rel}")
        changed += 1
    live = []
    for p in sorted(BASE.rglob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:                                                 # noqa: BLE001
            continue
        if isinstance(d.get("provenance"), dict) and d.get("envs"):
            live.append(p.relative_to(BASE))
    print(f"{changed} newly excluded, {missing} missing; "
          f"{len(live)} shard(s) remain certifiable:")
    for p in live:
        print(f"  {p}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
