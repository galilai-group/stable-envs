#!/bin/bash
export MUJOCO_GL=egl
export PYTHONPATH=/mnt/vast/home/randall.balestriero/code/stable-envs
cd /mnt/vast/home/randall.balestriero/code/stable-envs
LOG=/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out/pytest_mine.log
/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python -m pytest tests/ -q -rs \
  -k "Pointmaze or Antmaze or Humanoidmaze or PushT or registry" >"$LOG" 2>&1
echo "EXIT=$?" >>"$LOG"
