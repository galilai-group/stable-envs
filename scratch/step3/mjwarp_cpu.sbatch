#!/bin/bash
#SBATCH --job-name=se-step3-mjwarpcpu
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail
# device="cpu" ON a GPU node. The CPU-partition run of these four returned "Found no NVIDIA driver",
# which is ambiguous between "this renderer has no CPU path" and "this node has no driver". Running
# the same CPU sweep where a driver exists separates the two, and the answer decides whether the four
# mjwarp DM Control rows are a blocker or a genuine absence of CPU support.
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONPATH=$SCR/stable-envs
cd $SCR/stable-envs || exit 1
nvidia-smi --query-gpu=name --format=csv,noheader
"$SCR/venvs/se-step3/bin/python" examples/bench_step3.py --device cpu --threads 1 \
    --envs DMCCheetahRun,DMCHopperHop,DMCHumanoidStand,DMCWalkerWalk \
    --out "$SCR/step3-out/cpu/oscar_x86_mjwarp_1thread.json" --max-cell-s 12
echo "MJWARP CPU PROBE DONE rc=$?"
