#!/bin/bash
V=/mnt/vast/home/randall.balestriero/venvs/senvdiff/bin/python
cd /mnt/vast/home/randall.balestriero/code/stable-envs
for E in HalfCheetah-v5 Hopper-v5 Walker2d-v5 Ant-v5 Humanoid-v5 Swimmer-v5; do
  for RUNG in "L0:" "L1:--refl0" "L2:--refl0 --shadow0" "L3:--refl0 --shadow0 --floortex0"; do
    NAME=${RUNG%%:*}; FLAGS=${RUNG#*:}
    echo "===== $E $NAME"
    $V scratch/nvdiff/probe_loco.py --env $E --states 8 --tag _$NAME $FLAGS 2>&1 | grep -E "^@@score|Traceback|line [0-9]+, in"
  done
done
