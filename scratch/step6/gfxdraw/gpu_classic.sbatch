#!/bin/bash
#SBATCH --job-name=se6-gfxdraw-gpuclassic
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Two things that both need a CUDA device.
#
# 1. CarRacing, which the CPU consumer check cannot reach at all ("Expected a cuda device, but got:
#    cpu" -- its frame comes from the Triton kernel). It uses only G.Gfx from render/gfxdraw, so it
#    is structurally out of reach of the canvas change, and this is the evidence for that rather
#    than the argument. LunarLander is re-run here too, on CUDA this time.
# 2. GFXDRAW-2's fused classic-control kernel, first light: the identity tests against the
#    per-primitive path and against the torch form of the same closed expression.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export TRITON_CACHE_DIR=$SCR/.cache/triton
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
PY=$SCR/venvs/se-step6/bin/python
BOX=$SCR/venvs/se6-gfxdraw-box2d/bin/python
OUT=$SCR/se6/gfxdraw/examples/results/step6/gfxdraw
NODE=$(hostname -s)
mkdir -p "$OUT/consumers" "$OUT/tests"
echo "NODE $NODE JOB $SLURM_JOB_ID"; nvidia-smi --query-gpu=name --format=csv,noheader
set -x

# ---- 1. the two Box2D consumers, both arms, on CUDA -------------------------------------------
for ARM in baseline gfxdraw; do
  cd "$SCR/se6/$ARM"
  PYTHONPATH=$SCR/se6/$ARM "$BOX" \
      $SCR/se6/gfxdraw/scratch/step6/gfxdraw/consumer_digests.py \
      --envs LunarLander,CarRacing --device cuda --batch 6 \
      --snapshot-out "$OUT/consumers/consumers_cuda_${ARM}_${NODE}.json"
  echo "CONSUMERS CUDA $ARM rc=$?"
done
cd "$SCR/se6/gfxdraw"
PYTHONPATH=$SCR/se6/gfxdraw "$BOX" scratch/step6/gfxdraw/consumer_digests.py \
    --envs LunarLander,CarRacing --device cuda --batch 6 \
    --snapshot-in "$OUT/consumers/consumers_cuda_baseline_${NODE}.json"
echo "CONSUMERS CUDA BRANCH-VS-BASELINE rc=$?"

# ---- 2. the fused classic kernel, first light --------------------------------------------------
PYTHONPATH=$SCR/se6/gfxdraw $PY -m pytest tests/gpu/test_gfxdraw_classic.py \
    -v --tb=short > "$OUT/tests/classic_kernel_${NODE}.txt" 2>&1
echo "CLASSIC KERNEL TESTS rc=$?"
grep -E "PASSED|FAILED|ERROR|passed|failed" "$OUT/tests/classic_kernel_${NODE}.txt" | tail -40

PYTHONPATH=$SCR/se6/gfxdraw $PY -m pytest tests/gpu/test_fast_paths.py \
    -q -k "gfxdraw-classic" --tb=short > "$OUT/tests/classic_fastpath_${NODE}.txt" 2>&1
echo "CLASSIC FAST-PATH rc=$?"
tail -20 "$OUT/tests/classic_fastpath_${NODE}.txt"

# the CarRacing identity test must stay green: its kernel is untouched but its Gfx is shared
PYTHONPATH=$SCR/se6/gfxdraw $PY -m pytest tests/gpu/test_gfxdraw_fused.py \
    -q --tb=short > "$OUT/tests/fused_${NODE}.txt" 2>&1
echo "CARRACING FUSED rc=$?"
grep -E "passed|failed" "$OUT/tests/fused_${NODE}.txt" | tail -3
echo "GPU CLASSIC JOB DONE"
