#!/bin/bash
# One cell per process. A CUDA 700 poisons the interpreter, so a loop turns the first real failure
# into N fabricated ones.
V=/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python
L=/mnt/vast/home/randall.balestriero/logs/dmc2
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export MUJOCO_GL=egl PYTHONPATH=. WARP_CACHE_PATH=$HOME/.cache/warp
for E in DMCCheetahRun DMCWalkerWalk DMCHopperHop DMCHumanoidStand DMCSwimmerSwimmer6; do
  echo "===== $E"
  $V scratch/dmcontrol/visual_calib.py --env $E --n 64 --draws 3 > $L/visual_$E.txt 2>&1 \
    && grep -E "draw |@@wrote" $L/visual_$E.txt || tail -25 $L/visual_$E.txt
done
