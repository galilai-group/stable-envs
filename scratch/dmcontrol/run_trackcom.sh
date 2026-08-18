#!/bin/bash
# One domain per process. A CUDA 700 poisons the interpreter (docs/research/metaworld.md S4), so a
# loop inside one python turns the first real failure into N fabricated ones.
V=/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python
L=/mnt/vast/home/randall.balestriero/logs/dmc2
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export MUJOCO_GL=egl PYTHONPATH=. WARP_CACHE_PATH=$HOME/.cache/warp
for CELL in "cheetah run" "walker walk" "hopper hop" "humanoid stand" "fish upright" \
            "fish swim" "swimmer swimmer6" "reacher easy" "finger spin" "ball_in_cup catch"; do
  set -- $CELL
  echo "===== $1-$2"
  $V scratch/dmcontrol/probe_trackcom.py --domain "$1" --task "$2" --ladder \
     > $L/track_$1-$2.txt 2>&1 && grep -E '"mean_abs"|"ssim"|"within8_frac"|camera_vs' $L/track_$1-$2.txt | head -6 \
     || tail -20 $L/track_$1-$2.txt
done
