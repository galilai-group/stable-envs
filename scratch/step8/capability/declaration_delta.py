"""``style_dim`` / ``default_style_digest`` for every cell the capability patch touches, before and
after, so the declaration move is a recorded number and not a claim.

Adding a style leaf is a DECLARATION change: it widens the style row every certified artifact for
that cell is keyed on, and Gate A's style checks re-run over the new dimensions. Run this from the
baseline tree and from the patched tree and diff the two files -- the digest of the DEFAULT row must
be read together with the pixel evidence, because a default digest CAN legitimately move (the row is
wider) while the rendered default frame must not.

    python scratch/step8/capability/declaration_delta.py out.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np  # noqa: E402

from stable_envs.core.env import REGISTRY, make  # noqa: E402

CELLS = ["DMCAcrobotSwingup", "DMCBallInCupCatch", "DMCCartpoleBalance", "DMCCartpoleSwingup",
         "DMCFingerSpin", "DMCFingerTurnEasy", "DMCFingerTurnHard", "DMCManipulatorBringBall",
         "DMCPendulumSwingup", "DMCPointMassEasy", "DMCReacherEasy", "DMCReacherHard",
         "DMCCheetahRun", "DMCWalkerWalk", "DMCHopperHop", "DMCHumanoidStand", "DMCQuadrupedWalk",
         "OGBCube", "OGBScene", "FetchPush", "FetchReach"]


def digest(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    out = {}
    for cell in CELLS:
        if cell not in REGISTRY:
            out[cell] = {"skipped": "not registered in this interpreter"}
            continue
        try:
            env = make(cell, batch=2, size=84, device="cpu")
            d = np.asarray(env.default_style(), np.float32).reshape(-1)
            st = env.sample_states(2, np.random.default_rng(0))
            f = env.render(st)
            out[cell] = {"style_dim": int(d.size), "default_style_digest": digest(d),
                         "leaves": [p for p, _, _ in env.variation_space.flat_layout()],
                         "default_frame_digest": digest(f.detach().cpu().numpy()),
                         "states_digest": digest(np.asarray(st, np.float64))}
        except Exception as e:                                    # pragma: no cover - env dependent
            out[cell] = {"error": repr(e)[:300]}
        print(cell, out[cell].get("style_dim"), str(out[cell].get("default_frame_digest"))[:12],
              flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


main()
