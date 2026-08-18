#!/bin/bash
# Generic one-shot runner for the audit probes.
#   sbatch --job-name=X --export=ALL,PY=<interp>,SCRIPT=<path>,ARGS="..." scratch/audit/run.sbatch
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/audit/%x_%j.log
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
PY=${PY:-/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python}
echo "node=$(hostname) py=$PY script=$SCRIPT args=${ARGS:-}"
"$PY" -u "$SCRIPT" ${ARGS:-}
echo "rc=$? node=$(hostname)"
