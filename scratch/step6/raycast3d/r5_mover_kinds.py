"""Which primitive KINDS each live config actually casts, so the predicate's effect is evidence.

The per-kind table says a SPHERE never pays and a CAPSULE pays several times over. That explains the
Pointmaze regression and ball_in_cup's gain only if those scenes really are made of what I claim --
and the kinds come from the MuJoCo model at bake time, so it is not something to read off the source
and assert in a comment. This counts them.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

CONFIGS = """AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore
HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate
PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate
DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCFingerSpin
DMCFingerTurnEasy DMCFingerTurnHard DMCPendulumSwingup DMCReacherEasy DMCReacherHard""".split()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np
    import stable_envs as se
    from stable_envs.render import raycast3d as R3

    NAMES = {R3.PLANE: "PLANE", R3.SPHERE: "SPHERE", R3.CAPSULE: "CAPSULE",
             R3.CYLINDER: "CYLINDER", R3.BOX: "BOX"}
    note = {"tool": "r5_mover_kinds", "host": socket.gethostname(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "stable_envs_from": se.__file__,
            "bounded_kinds": sorted(NAMES.get(k, str(k)) for k in R3._BOUNDED_KINDS),
            "configs": {}}

    for name in a.configs:
        try:
            env = se.make(name, batch=2, device="cpu")
            env.render(np.asarray(env.sample_states(2), np.float32))      # bake
            # ``_scene`` is a METHOD, and the per-frame primitives live under DIFFERENT names in the
            # two families: dmcontrol keeps them on ``sc.prims``, ogbench on ``sc.prims_base`` (its
            # ``sc.sprims`` are the statics that went into the baked plate and are not cast per
            # frame). The first version of this probe treated the bound method as the scene and
            # reported "no scene.prims" for all 21 -- an error that at least announced itself.
            sc = env._scene()
            got = {}
            for attr in ("prims", "prims_base"):
                prims = getattr(sc, attr, None)
                if prims is None or getattr(prims, "kind", None) is None:
                    continue
                c = Counter(NAMES.get(int(k), f"kind{int(k)}")
                            for k in np.asarray(prims.kind).reshape(-1))
                bounded = sum(v for k, v in c.items() if k in note["bounded_kinds"])
                got[attr] = {"kinds": dict(sorted(c.items())), "total": int(sum(c.values())),
                             "bounded": int(bounded),
                             "unbounded": int(sum(c.values()) - bounded)}
            if not got:
                note["configs"][name] = {
                    "error": f"no prims on {type(sc).__name__}: {sorted(vars(sc))[:12]}"}
                continue
            note["configs"][name] = got
            print(f"{name:28s} {note['configs'][name]}", flush=True)
        except Exception as e:                                             # noqa: BLE001
            note["configs"][name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{name:28s} ERROR {type(e).__name__}: {e}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(note, indent=1))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
