#!/bin/bash
# The real bit-exactness gate: parity against the OFFICIAL renderer, plus batch invariance, for the
# envs this work touches. se-envs is the only interpreter with highway_env installed.
#   sbatch scratch/fastraster/run_parity.sbatch -k "Highway or Merge"
#SBATCH --job-name=se-parity
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/parity_%j.out
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
export TRITON_CACHE_DIR=/mnt/vast/home/randall.balestriero/.triton_cache_$SLURM_JOB_ID
PY=/mnt/vast/home/randall.balestriero/venvs/se-envs/bin/python
echo "node=$(hostname) args=$*"
"$PY" -u -m pytest tests/gpu -x -q -rs "$@"
rc=$?
rm -rf "$TRITON_CACHE_DIR"
echo "parity job=${SLURM_JOB_ID:-manual} node=$(hostname) rc=$rc"
exit $rc
