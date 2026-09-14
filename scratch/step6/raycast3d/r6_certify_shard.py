"""RAYCAST-6's Step-3-shaped shard: ``cpu-3d``'s paired producer, relabelled for this workstream.

``examples/bench_step3.py`` is the Step-3 producer and the RAYCAST-5 shards came from it, but it
times the official renderer **once, in a block, after the whole batch sweep**, and this workstream's
own noise measurement put the off-vs-off spread on identical code at 11.7 % on a shared batch node.
``examples/results/step6/gfxdraw/characterize/cpu_shard_attempts.json`` is the record of what that
does to a denominator: 2.3x between the two arms of one job on ``node2338``, far enough to relabel
four configurations a tier up. ``scratch/step6/cpu-3d/certify_shard_cpu3d.py`` already fixes exactly
that -- same schema, same cells, same decode bar, one reference burst BETWEEN every batch cell, the
median reported with its spread beside it and a ``--max-reference-spread`` that rejects the node
rather than publishing from it. So this runs that, unmodified, and only relabels the result.

Two things it changes after the fact, neither of them a measurement:

  * ``arm``/``shard``. The underlying script's ``--arm before`` also forces classic control's
    per-primitive render path, which is ``cpu-3d``'s control and has nothing to do with this ticket;
    none of the envs here are classic control, so it is always run at its default and the leg is
    recorded here instead, from ``SE6_LEG``.
  * ``provenance.se6_switches``. A shard classified as if it were Step 3 must still say which build
    produced it, or two legs of a pair are indistinguishable afterwards. ``COMPOSITE_COMPACT_MERGE``
    is absent on the integration tree and that absence is itself recorded, rather than defaulted.

    SE6_LEG=before PYTHONPATH=$INT python scratch/step6/raycast3d/r6_certify_shard.py --device cpu ...
"""
from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INNER = ROOT / "scratch" / "step6" / "cpu-3d" / "certify_shard_cpu3d.py"


def main() -> int:
    leg = os.environ.get("SE6_LEG")
    if leg not in ("before", "after"):
        raise SystemExit("set SE6_LEG=before|after: the leg is what distinguishes the two shards of "
                         "a pair, and a shard that does not say which build produced it is not "
                         "evidence about a change")
    if not INNER.is_file():
        raise SystemExit(f"{INNER} is missing; it arrives with step6/integration and this shard is "
                         "deliberately not a second copy of it")

    from stable_envs.render import raycast3d as R3
    switches = {"COMPOSITE_COMPACT_MERGE": getattr(R3, "COMPOSITE_COMPACT_MERGE", None),
                "COVER_COMPACTION": getattr(R3, "COVER_COMPACTION", None),
                "PRIMITIVE_BOUNDS": getattr(R3, "PRIMITIVE_BOUNDS", None),
                "module": R3.__file__}
    print(f"[r6_certify_shard] leg={leg} switches={switches}", flush=True)

    out = None
    argv = list(sys.argv[1:])
    for i, a in enumerate(argv):
        if a == "--out":
            out = Path(argv[i + 1])
    sys.argv = [str(INNER)] + argv
    try:
        runpy.run_path(str(INNER), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            raise

    if out and out.is_file():
        d = json.loads(out.read_text())
        d["arm"] = leg
        d["shard"] = f"raycast3d-paired-{leg}"
        pv = d.setdefault("provenance", {})
        pv["se6_switches"] = switches
        # A SINGLE $STABLE_ENVS_GIT_COMMIT IS WRONG FOR A PAIR. The inner script reads the commit
        # from `git rev-parse` in the tree it imported and falls back to that environment variable
        # when the tree has no .git -- which is both trees here, because rsync drops it. So a job
        # that exports one SHA stamps it on BOTH legs, and the before leg comes out claiming the
        # branch's commit while having measured $SCR/se6/integration. Observed in job 6368126, where
        # both legs record 7fa1856; `provenance.stable_envs_file` shows the truth and
        # `tree_commit` did not. $SE6_TREE_COMMIT is per leg, and if it is absent the field is set
        # to null with a reason rather than left holding the other leg's answer.
        want = os.environ.get("SE6_TREE_COMMIT")
        pv["tree_commit"] = want
        pv["git_commit_source"] = ("env:SE6_TREE_COMMIT (per leg)" if want else
                                   "unavailable: no .git in the measured tree and SE6_TREE_COMMIT "
                                   "was not set for this leg; see provenance.stable_envs_file for "
                                   "which tree was actually imported")
        pv["measured_tree"] = switches["module"]
        d["produced_by"] = ("scratch/step6/raycast3d/r6_certify_shard.py wrapping "
                            "scratch/step6/cpu-3d/certify_shard_cpu3d.py")
        out.write_text(json.dumps(d, indent=1))
        print(f"[r6_certify_shard] stamped {out} as leg={leg}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
