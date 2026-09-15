"""The Step-7 view of ``scratch/step6/verify/compare49.py``.

The diff engine is not rewritten: this imports Step 6's comparator by path and replaces only the two
tables that are a STATEMENT ABOUT THIS STEP rather than about the machinery -- the set of deltas that
are explained in advance and the subset of those allowed to move a default-appearance pixel. Keeping
one engine means a row that reads ``identical`` here means exactly what it meant in Step 6.

What Step 7 adds to Step 6's known set, and why:

  CraftaxClassic  W1 took ``swm/CraftaxClassicPixels-v1`` from partial to FULL coverage. The
                  observation grew from the 7x9 world window (112,144,3) to the whole SWM frame
                  (144,144,3) -- world window + 2-row inventory strip -- and the state window grew
                  from 127 to 149 floats to carry the strip, the sleeping bit and the day-cycle PRNG
                  key. The frozen Step-4 digest is a digest of a SMALLER PICTURE, so it cannot
                  reproduce; the coordinator's ruling (dashboard, W1 decisions) is that the replay
                  obligation for this cell is that the WORLD ROWS stay byte-identical at light 1.0,
                  which W1 asserted three ways and the reviewer reproduced (frozen digest
                  1aeabbe8bc6a9a2e). A moved digest here is therefore expected; a moved world row
                  would be a finding and is checked separately, not by this script.

Everything else that Step 6 recorded as expected-known is carried forward unchanged, because none of
those mechanisms was touched in Step 7 and all of them are properties of the frozen corpus or of the
reference stack rather than of a branch.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SRC = ROOT / "scratch" / "step6" / "verify" / "compare49.py"

spec = importlib.util.spec_from_file_location("compare49_step6", _SRC)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

C.KNOWN = dict(C.KNOWN)
C.KNOWN["CraftaxClassic"] = (
    "coverage change, not a renderer change: W1 took CraftaxClassicPixels partial->full, so the "
    "observation is (144,144,3) (world window + inventory strip) where the frozen digest is of the "
    "(112,144,3) world window alone, and the state window is 149 floats where the frozen corpus "
    "carries 127. The replay obligation the coordinator set for this cell is world-row byte "
    "identity at light 1.0 (frozen world-row digest 1aeabbe8bc6a9a2e), asserted by W1 and "
    "reproduced by its reviewer; it is not this table's digest")
C.KNOWN_PIXEL_MOVERS = set(C.KNOWN_PIXEL_MOVERS) | {"CraftaxClassic"}

if __name__ == "__main__":
    sys.exit(C.main())
