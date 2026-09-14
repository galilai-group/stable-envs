#!/bin/bash
#SBATCH --job-name=se6-cpu-3d-gpu-postmerge
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The cuda frames of the four trackcom cells must be byte-identical before (integration tree) and
# after (this branch: a dispatch change only) -- one card, one job, the reacher-correctness method.
# Also: a fresh cuda frame stack from the branch tree (the cross-backend oracle) and the frozen
# Step-4 replay of the dmc shard on cuda.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
WS=$SCR/se6/cpu-3d
BEFORE=$SCR/se6/integration
PY=$SCR/venvs/se-step5/bin/python
OUT=$WS/examples/results/step6/cpu-3d/replay/postmerge
STATES=$WS/examples/results/step4/states
ENVS=DMCCheetahRun,DMCHopperHop,DMCWalkerWalk,DMCHumanoidStand
mkdir -p "$OUT/gpu"
cd "$WS"
echo "NODE $(hostname -s) JOB $SLURM_JOB_ID"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee "$OUT/gpu/card.txt"
set -x
PYTHONPATH=$BEFORE $PY scratch/step6/reacher-correctness/verify_pixels.py --envs "$ENVS" --states "$STATES" --out "$OUT/gpu/pixels_before_integration.json"
PYTHONPATH=$WS     $PY scratch/step6/reacher-correctness/verify_pixels.py --envs "$ENVS" --states "$STATES" --out "$OUT/gpu/pixels_after_branch.json"
PYTHONPATH=$WS $PY - "$OUT/gpu/pixels_before_integration.json" "$OUT/gpu/pixels_after_branch.json" <<'PYEOF'
import json, sys
a = json.load(open(sys.argv[1]))["envs"]; b = json.load(open(sys.argv[2]))["envs"]
bad = 0
for k in sorted(a):
    same = a[k].get("digest_ours") == b[k].get("digest_ours") and a[k].get("digest_default") == b[k].get("digest_default")
    bad += (not same)
    print(f"{k:22} cuda_pixels_identical={same}  mean {a[k].get('mean_pixel')} -> {b[k].get('mean_pixel')}")
print("CUDA PIXEL VERDICT:", "IDENTICAL" if not bad else f"{bad} ENV(S) MOVED")
PYEOF
export PYTHONPATH=$WS
$PY scratch/step6/cpu-3d/gpu_frames.py --out "$OUT/gpu/frames_cuda_$(hostname -s).npz"
$PY examples/fidelity_step4.py --shard dmc --configs "$ENVS" --device cuda --states-dir "$STATES" --out "$OUT/gpu/dmc_cuda.json" 2>&1 | tail -30
$PY scratch/step6/reacher-correctness/compare_replay.py "$WS/examples/results/step4/conformance/dmc.json" "$OUT/gpu/dmc_cuda.json" 2>&1 | tee "$OUT/gpu/compare_frozen_cuda.txt"
set +x; echo "GPU REPLAY DONE"
