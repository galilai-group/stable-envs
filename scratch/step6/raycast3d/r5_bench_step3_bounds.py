"""Run ``examples/bench_step3.py`` with ``raycast3d.PRIMITIVE_BOUNDS`` pinned.

``bench_step3.py`` is the Step-3 producer and belongs to Step 3; it has no business growing a knob
for a Step-6 switch. This wrapper sets the switch on the interpreter and then runs it unmodified, so
the shard it writes is schema-identical to the frozen ones and the only difference is the flag --
which the shard records, below, in a field no Step-3 shard has and nothing downstream reads.
"""
from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    want = os.environ.get("SE6_PRIMITIVE_BOUNDS")
    from stable_envs.render import raycast3d as R3
    if want is not None:
        if not hasattr(R3, "PRIMITIVE_BOUNDS"):
            raise SystemExit("this stable_envs has no PRIMITIVE_BOUNDS; pinning it would label a "
                             "shard with a switch state it does not have")
        R3.PRIMITIVE_BOUNDS = want == "on"
    print(f"[bench_step3_bounds] PRIMITIVE_BOUNDS={R3.PRIMITIVE_BOUNDS} "
          f"_BOUNDED_KINDS={sorted(getattr(R3, '_BOUNDED_KINDS', ()))} in {R3.__file__}", flush=True)

    out = None
    for i, a in enumerate(sys.argv):
        if a == "--out":
            out = Path(sys.argv[i + 1])
    sys.argv[0] = str(ROOT / "examples/bench_step3.py")
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    # STAMP WHAT WAS MEASURED. A shard that is classified as if it were Step 3 must still say which
    # switch state produced it, or two legs of this pair are indistinguishable after the fact.
    if out and out.is_file():
        d = json.loads(out.read_text())
        d.setdefault("provenance", {})["se6_switches"] = {
            "PRIMITIVE_BOUNDS": R3.PRIMITIVE_BOUNDS,
            "_BOUNDED_KINDS": sorted(str(k) for k in getattr(R3, "_BOUNDED_KINDS", ())),
            "module": R3.__file__}
        out.write_text(json.dumps(d, indent=1))
        print(f"[bench_step3_bounds] stamped {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
