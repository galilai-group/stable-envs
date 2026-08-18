#!/bin/bash
#SBATCH --job-name=se_gvc
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/census/gvc_%x_%j.log

# Regenerate examples/results/bench_gpu_vs_cpu.{json,md}.
#
# The committed copy of that pair was taken with `--ss 1`, which is not what `se.make(name)` gives for
# the 44 classes declaring `default_supersample = 2`, and its markdown still carries the retracted
# claim that the GPU column repeats across nodes to 0.4%. Both are properties of the script rather than
# of the numbers, so the fix is to re-run it: the current script defaults to `--ss native` and emits a
# `cpu%` column, which is what says whether a given row travels between nodes at all.
#
# usage: sbatch scratch/census/gpu_vs_cpu.sbatch [extra args]
R=/mnt/vast/home/randall.balestriero/code/stable-envs
P=/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
export PYTHONPATH=$R

cd $R/examples
$P bench_gpu_vs_cpu.py --out $R/examples/results "$@"
