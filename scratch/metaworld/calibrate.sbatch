#!/bin/bash
# Per-task calibration for the Meta-World family. ONE TASK PER PROCESS, hence one ARRAY TASK per task:
# a CUDA 700 out of mujoco_warp poisons the process, so a loop would report one real failure and then
# invent the rest ("Failed to allocate 4 bytes", seven times, in probe_backend2).
#
#   sbatch --array=0-49%10 scratch/metaworld/calibrate.sbatch
#   sbatch --array=0-49%10 scratch/metaworld/calibrate.sbatch --size 224
#
#SBATCH --job-name=mw-cal
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-070,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/se_logs/metaworld/cal_%A_%a.out
set -uo pipefail
# Repo root hardcoded: $BASH_SOURCE is the staged copy under SLURM, not this file.
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl
PY=/mnt/vast/home/randall.balestriero/venvs/se-fetch/bin/python

TASK=$("$PY" -c "from stable_envs.envs.metaworld import TASKS; print(TASKS[${SLURM_ARRAY_TASK_ID}])")
echo "task=$TASK node=$(hostname)"
"$PY" -u scratch/metaworld/calibrate.py "$TASK" --dims "$@"
echo "done task=$TASK rc=$?"
