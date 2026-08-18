#!/bin/bash
# Generic runner for the raster speed work.
#   sbatch scratch/fastraster/run.sbatch <venv> <script> [args...]
# e.g. sbatch scratch/fastraster/run.sbatch se-hb scratch/fastraster/profile_baseline.py highway
#SBATCH --job-name=se-raster
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/raster_%j.out
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
export TRITON_CACHE_DIR=/mnt/vast/home/randall.balestriero/.triton_cache_$SLURM_JOB_ID
VENV=$1; shift
if [ "$VENV" = "se-hb" ]; then export STABLE_ENVS_HUMANOIDBENCH=1; fi
PY=/mnt/vast/home/randall.balestriero/venvs/$VENV/bin/python
if [ "$VENV" = "test" ]; then PY=/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python; fi
echo "node=$(hostname) venv=$VENV cmd=$*"
"$PY" -u -X faulthandler "$@" 2>&1
rc=${PIPESTATUS[0]}
rm -rf "$TRITON_CACHE_DIR"
echo "raster job=${SLURM_JOB_ID:-manual} node=$(hostname) rc=$rc"
exit $rc
