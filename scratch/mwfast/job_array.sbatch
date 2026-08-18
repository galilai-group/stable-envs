#!/bin/bash
#SBATCH --job-name=mwfarr
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/mwfast/%x_%A_%a.out

# $1 = file with one ENV NAME per line, $2... = the script and its remaining arguments.
set -euo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=.:scratch/mwfast:scratch/altrender/rendernet:scratch/altrender
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1
export PYTHONUNBUFFERED=1
# A private warp kernel cache per job: 18 concurrent jobs compiling the same megakernel into
# the shared ~/.cache/warp raced and produced WarpCodegenError on 17 of 50 tasks.
export WARP_CACHE_PATH=/mnt/vast/home/randall.balestriero/.cache/warp_jobs/$SLURM_JOB_ID
LIST="$1"; shift
ENV_NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
echo "node $SLURMD_NODENAME  job $SLURM_JOB_ID  env $ENV_NAME  $*"
srun /mnt/vast/home/randall.balestriero/venvs/se-hb/bin/python "$1" "$ENV_NAME" "${@:2}"
