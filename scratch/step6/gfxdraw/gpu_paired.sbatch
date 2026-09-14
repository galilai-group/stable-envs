#!/bin/bash
#SBATCH --job-name=se6-gfxdraw-gpupaired
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Two things the first GPU job could not settle.
#
# 1. R. Its denominator moved 2x between the two arms of one job on one node (Acrobot 428.2 then
#    229.8), and the class-R GPU_FAST bar of 500 sits inside the resulting range. paired_gpu.py
#    interleaves the reference and the renderer round by round, in one process, both arms.
# 2. CarRacing's fused identity test, which collected 4 tests and SKIPPED all four because
#    se-step6 has no Box2D -- the suite's own coverage guard turned that into a non-zero exit,
#    correctly. Re-run in the Box2D venv so it actually executes.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export TRITON_CACHE_DIR=$SCR/.cache/triton
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
PY=$SCR/venvs/se-step6/bin/python
BOX=$SCR/venvs/se6-gfxdraw-box2d/bin/python
OUT=$SCR/se6/gfxdraw/examples/results/step6/gfxdraw
NODE=$(hostname -s)
mkdir -p "$OUT/sweep2" "$OUT/tests"
cd "$SCR/se6/gfxdraw"
echo "NODE $NODE JOB $SLURM_JOB_ID"; nvidia-smi --query-gpu=name --format=csv,noheader
set -x

# CarRacing's kernel is untouched but it shares Gfx, so its identity test is the evidence
PYTHONPATH=$SCR/se6/gfxdraw "$BOX" -m pytest tests/gpu/test_gfxdraw_fused.py -v --tb=short \
    > "$OUT/tests/fused_box2d_${NODE}.txt" 2>&1
echo "CARRACING FUSED (box2d) rc=$?"
grep -E "PASSED|FAILED|ERROR|passed|failed|SKIPPED" "$OUT/tests/fused_box2d_${NODE}.txt" | tail -12

# R, paired, at the headline batch and at the batch the sweep called best
for B in 4096 1024; do
  PYTHONPATH=$SCR/se6/gfxdraw $PY scratch/step6/gfxdraw/paired_gpu.py \
      --batch $B --rounds 7 --out "$OUT/sweep2/paired_gpu_B${B}_${NODE}.json"
  echo "PAIRED GPU B=$B rc=$?"
done
echo "GPU PAIRED JOB DONE"
