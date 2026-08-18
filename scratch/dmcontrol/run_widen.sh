#!/bin/bash
# One domain per process, one process per srun step: two MuJoCo envs alive in one interpreter can
# return byte-identical frames from cameras that provably differ, and an EGL context is never
# released, so a loop inside one python is not a cheaper version of this.
V=/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export MUJOCO_GL=egl PYTHONPATH=.
for CELL in "reacher easy" "reacher hard" "finger spin" "finger turn_easy" "ball_in_cup catch" \
            "fish upright" "fish swim" "swimmer swimmer6" "cheetah run" "walker walk" \
            "hopper hop" "humanoid stand" "point_mass easy"; do
  set -- $CELL
  echo "===== $1-$2"
  $V scratch/dmcontrol/probe_widen.py --domain "$1" --task "$2" \
     > /mnt/vast/home/randall.balestriero/logs/dmc2/widen_$1-$2.txt 2>&1 \
     && echo "  ok" || tail -25 /mnt/vast/home/randall.balestriero/logs/dmc2/widen_$1-$2.txt
done
