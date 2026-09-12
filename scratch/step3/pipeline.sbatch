#!/bin/bash
#SBATCH --job-name=se-step3-pipe
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONPATH=$SCR/stable-envs
cd $SCR/stable-envs || exit 1
nvidia-smi --query-gpu=name --format=csv,noheader
# Reacher is the project-level 3-D representative AND the known HOST_BOUND row; DMCHumanoidStand is
# carried beside it as the non-host-bound 3-D control, so a loader finding cannot be confused with
# Reacher's python-loop problem.
"$SCR/venvs/se-step3/bin/python" examples/bench_step3_pipeline.py --device cuda \
    --envs PushT,DMCReacherHard,DMCHumanoidStand --batches 256,1024,4096 \
    --out "$SCR/step3-out/pipeline/3090_pipeline.json"
echo "PIPELINE DONE rc=$?"
