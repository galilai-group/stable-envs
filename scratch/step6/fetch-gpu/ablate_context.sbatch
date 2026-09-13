#!/bin/bash
#SBATCH --job-name=se6-fetch-gpu-ablate
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

# profile_note.py reached its write step with a NameError on the baseline (`n["host"]` for
# `note["host"]`), so job 6343470 measured both envs at B=4096 and then wrote nothing. Fixed in this
# worktree and re-run here so profile_before covers the qualifier batch with a real artifact.
for E in FetchReach FetchPickAndPlace; do
  $PY scratch/step6/profile_note.py --env $E --device cuda \
      --batch 1 16 64 256 1024 4096 --repeat 10 --warmup 3 --workstream step6/fetch-gpu \
      --states-dir "$WS/examples/results/step4/states" \
      --out "$WS/examples/results/step6/fetch-gpu/profile_before_4096" || echo "profile_note $E rc=$?"
done

echo "STEP6D_ABLATE_DONE"
ls -la "$WS/examples/results/step6/fetch-gpu/ablation" 2>/dev/null
