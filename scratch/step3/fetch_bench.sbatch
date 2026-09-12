#!/bin/bash
#SBATCH --job-name=se-step3-fetch
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONPATH=$SCR/stable-envs
cd $SCR/stable-envs || exit 1
V=$SCR/venvs/se-step3-fetch/bin/python
"$V" -c "import mujoco,gymnasium_robotics;print('mujoco',mujoco.__version__,'gym-robotics',gymnasium_robotics.__version__)"
F=FetchPickAndPlace,FetchPickAndPlaceDense,FetchPickAndPlaceDict,FetchPush,FetchPushDense,FetchPushDict,FetchReach,FetchReachDense,FetchReachDict,FetchSlide,FetchSlideDense,FetchSlideDict
# GPU first, then the single-threaded CPU sweep in the same interpreter so both shards see the same
# package set and the same node -- Step 2 requires every ratio to be same-host.
"$V" examples/bench_step3.py --device cuda --envs "$F" \
    --out "$SCR/step3-out/gpu/3090_fetch.json" --resume --max-cell-s 30
"$V" examples/bench_step3.py --device cpu --threads 1 --envs "$F" \
    --out "$SCR/step3-out/cpu/oscar_x86_fetch_1thread.json" --resume --max-cell-s 12
echo "FETCH BENCH DONE rc=$?"
