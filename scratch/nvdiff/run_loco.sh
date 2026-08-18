#!/bin/bash
# One process per env: two MuJoCo envs alive together can return byte-identical frames from cameras
# that provably differ.
V=/mnt/vast/home/randall.balestriero/venvs/senvdiff/bin/python
cd /mnt/vast/home/randall.balestriero/code/stable-envs
for E in HalfCheetah-v5 Hopper-v5 Walker2d-v5 Ant-v5 Humanoid-v5 Swimmer-v5; do
  echo "===== $E L0 (official scene)"
  $V scratch/nvdiff/probe_loco.py --env $E --states "${1:-8}" --tag _L0 2>&1 | grep -E "^@@camera|^@@score|^@@scene|Error|Traceback"
done
