#!/bin/bash
# deliberately does NOT export MUJOCO_GL: proves the env module sets it itself
unset MUJOCO_GL
export PYTHONPATH=/mnt/vast/home/randall.balestriero/code/stable-envs
cd /mnt/vast/home/randall.balestriero/code/stable-envs
LOG=/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out/pytest_nogl.log
/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python -m pytest tests/ -q -rs >"$LOG" 2>&1
echo "EXIT=$?" >>"$LOG"
