"""Re-measure ``GymMujoco.parity_note``'s mover-pixel statistics at the new default supersample.

The note quotes exact fractions on MOVER pixels and on mover pixels eroded by one, and every one of
them was measured at ``supersample=2``. Changing the default without re-measuring them would leave a
docstring that describes a render the family no longer produces -- which is worse than no docstring,
because it reads as evidence.

``scratch/mujoco_locomotion/parity.py`` is the script that produced those numbers, so it is imported
and called rather than reimplemented: same mover mask out of MuJoCo's own segmentation renderer, same
one-pixel erosion, same reduction. It is driven at BOTH supersamples here so the note can state the
change rather than only the new value, and its output is written under this module's own logs instead
of over that script's ``parity.json``.
"""
from __future__ import annotations

import json
import os
import sys

import common as K  # noqa: E402  (sets MUJOCO_GL before mujoco is imported)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "mujoco_locomotion"))

import parity as P  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    out = {"n_states": n, "host": os.uname().nodename, "envs": {}}
    for name in P.NAMES:
        cls = K.cls_of(name)
        row = {}
        original = cls.default_supersample
        try:
            for ss in (1, 2):
                cls.default_supersample = ss
                r = P.run(name, n=n)
                row[f"ss{ss}"] = r
                s = r["spec0.988"]
                print(f"{name} ss={ss}: all-exact {s['exact_frac']:.4f} "
                      f"interior {s['interior_exact_frac']} ({s['interior_bad_px']}"
                      f"/{s['interior_px']}) mean {s['mean_abs']:.4f} max {s['max_abs']} | "
                      f"mover exact {s['mover_exact']:.4f} within1 {s['mover_within1']:.4f} "
                      f"mean {s['mover_mean']:.4f} | eroded exact {s['mover_interior_exact']:.4f} "
                      f"within1 {s['mover_interior_within1']:.4f} mean "
                      f"{s['mover_interior_mean']:.4f} | nonmover exact {s['nonmover_exact']:.5f}",
                      flush=True)
        finally:
            cls.default_supersample = original
        out["envs"][name] = row
        with open(os.path.join(K.LOGS, "gym_parity_note.json"), "w") as f:
            json.dump(out, f, indent=1)
    print("wrote", os.path.join(K.LOGS, "gym_parity_note.json"), flush=True)


if __name__ == "__main__":
    main()
