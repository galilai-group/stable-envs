#!/bin/bash
#SBATCH --job-name=se-s3b-bench
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONPATH=$SCR/stable-envs
P=$SCR/venvs/se-s3b-fetch311/bin/python
cd $SCR/stable-envs || exit 1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$P" -c "import importlib.metadata as m;print('mujoco',m.version('mujoco'),'mujoco-warp',m.version('mujoco-warp'),'gym-robotics',m.version('gymnasium-robotics'))"
F=FetchPickAndPlace,FetchPickAndPlaceDense,FetchPickAndPlaceDict,FetchPush,FetchPushDense,FetchPushDict,FetchReach,FetchReachDense,FetchReachDict,FetchSlide,FetchSlideDense,FetchSlideDict

# GPU first, saving the state pool. The CPU pass then RENDERS THE SAME STATES rather than drawing its
# own, so the two shards are comparable to the byte and the ratio Step 2 asks for is well defined.
"$P" examples/bench_step3.py --device cuda --envs "$F" \
    --out "$SCR/step3-out/gpu/3090_fetch311.json" --max-cell-s 30 \
    --states-out "$SCR/step3-out/states/fetch"
"$P" examples/bench_step3.py --device cpu --threads 1 --envs "$F" \
    --out "$SCR/step3-out/cpu/oscar_fetch311_1thread.json" --max-cell-s 12 \
    --states-in "$SCR/step3-out/states/fetch"
echo "S3B BENCH DONE rc=$?"
