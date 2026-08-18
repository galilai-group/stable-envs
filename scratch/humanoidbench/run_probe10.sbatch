#!/bin/bash
#SBATCH --job-name=hb-p10
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-070,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/stable_envs_logs/humanoidbench/slurm/p10a_%A_%a.out

# One task per PROCESS and per JOB: a CUDA 700 poisons the process, and two MuJoCo envs alive at once
# can return byte-identical frames from cameras that provably differ.
set -u
LIST=/mnt/vast/home/randall.balestriero/stable_envs_logs/humanoidbench/deadlist.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
export MUJOCO_GL=egl
cd /mnt/vast/home/randall.balestriero/code/stable-envs
/mnt/vast/home/randall.balestriero/venvs/se-hb/bin/python \
    scratch/humanoidbench/probe10_deaddims.py $LINE
