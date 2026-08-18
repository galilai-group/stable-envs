"""nvdiffrast at the resolution and batch the storage benchmark uses, on the two VISUAL-tier families.

``docs/research/nvdiffrast.md`` Section 5 measured 224 px and 64 px and concluded 0.086x MJWarp on a
RoboCasa kitchen. The question the storage benchmark asks is narrower and has to be answered at its
own operating point: **256 px, batch 1024**, against 15,723-18,743 f/s of 8-worker JPEG-out-of-Lance.
So this runs the same ``bench.run`` at 256 px on the two scenes that matter -- a RoboCasa kitchen and
a HumanoidBench H1 scene -- and nothing else, one process per scene because nvdiffrast answers "too
big" by killing the interpreter.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

from bench import run  # noqa: E402
from common import emit, save  # noqa: E402

HB = "/mnt/vast/home/randall.balestriero/src/humanoid-bench/humanoid_bench/assets/envs"
RC = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out/corpus"

SCENES = {
    "robocasa-L52S03": (os.path.join(RC, "L52S03", "ep000.xml"), "robot0_agentview_left"),
    "hb-h1-walk": (os.path.join(HB, "h1_pos_walk.xml"), None),
    "hb-h1hand-balance": (os.path.join(HB, "h1hand_pos_balance_simple.xml"), None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, choices=sorted(SCENES))
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 64, 256, 1024])
    a = ap.parse_args()
    import torch
    emit("device", {"name": torch.cuda.get_device_name(0), "torch": torch.__version__})
    xml, cam = SCENES[a.only]
    if not os.path.exists(xml):
        raise SystemExit(f"no such scene: {xml}")
    rows = run(xml, a.res, a.batches, a.only, cam)
    save(f"bench256_{a.only}_{a.res}.json", rows)


if __name__ == "__main__":
    main()
