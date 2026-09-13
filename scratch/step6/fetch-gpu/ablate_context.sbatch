#!/bin/bash
#SBATCH --job-name=se6-fetch-gpu-ablate2
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 6D, part 3. The stage profile says the Fetch plateau is per-frame DEVICE work, so the only
# lever on BATCH_SCALES is making that work smaller. This bounds how much of it is reachable through
# flags mujoco_warp and this repo already expose, and digests every cell against the control so a
# saving that moves pixels cannot be mistaken for one that does not.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch WARP_CACHE_PATH=$SCR/.cache/warp
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip HF_HOME=$SCR/.cache/hf
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export WS=$SCR/se6/fetch-gpu PYTHONPATH=$SCR/se6/fetch-gpu
cd "$WS"
PY=$SCR/venvs/se-s3b-fetch311/bin/python
set -x
nvidia-smi --query-gpu=name --format=csv,noheader
$PY scratch/step6/fetch-gpu/ablate_context.py --env FetchReach --batch 1 256 4096 \
    --repeat 8 --warmup 3 --states-dir "$WS/examples/results/step4/states" \
    --out "$WS/examples/results/step6/fetch-gpu/ablation" || echo "ablate rc=$?"

echo "STEP6D_ABLATE_DONE"
ls -la "$WS/examples/results/step6/fetch-gpu/ablation" 2>/dev/null
