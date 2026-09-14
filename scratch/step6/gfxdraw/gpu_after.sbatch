#!/bin/bash
#SBATCH --job-name=se6-gfxdraw-gpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# gfxdraw GPU: the mandatory batch sweep before and after ON THE SAME CARD, the frozen Step 4
# replay, and the path tests the brief names.
#
# geforce3090 by name because Step 3's GPU row for these five was measured on a 3090; a before on a
# 3090 and an after on an L40S is not a before/after. Both arms run in this one job for the same
# reason the CPU job does.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
PY=$SCR/venvs/se-step6/bin/python
ENVS=Acrobot,CartPole,MountainCar,MountainCarContinuous,Pendulum
OUT=$SCR/se6/gfxdraw/examples/results/step6/gfxdraw
NODE=$(hostname -s)
mkdir -p "$OUT/sweep" "$OUT/replay" "$OUT/profile_after"
echo "NODE $NODE JOB $SLURM_JOB_ID"; nvidia-smi --query-gpu=name --format=csv,noheader
set -x

# ---- frozen Step 4 replay, on CUDA, which is where the frozen digests were taken --------------
cd "$SCR/se6/gfxdraw"
PYTHONPATH=$SCR/se6/gfxdraw $PY examples/fidelity_step4.py --configs "$ENVS" --device cuda \
    --out "$OUT/replay/step4_replay_cuda.json" || echo "REPLAY rc=$?"
PYTHONPATH=$SCR/se6/gfxdraw $PY scratch/step6/gfxdraw/replay_digests.py --device cuda \
    --out "$OUT/replay/digests_cuda_${NODE}.json" || echo "DIGESTS rc=$?"

# ---- the path tests the brief names, plus the generic suite restricted to these five ----------
PYTHONPATH=$SCR/se6/gfxdraw $PY -m pytest tests/gpu/test_gfxdraw_fused.py tests/gpu/test_fast_paths.py \
    -q 2>&1 | tail -25
PYTHONPATH=$SCR/se6/gfxdraw $PY -m pytest tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py \
    -q -k "Acrobot or CartPole or MountainCar or MountainCarContinuous or Pendulum" 2>&1 | tail -25

# ---- the mandatory batch sweep, both arms, same card ------------------------------------------
cd "$SCR/se6/baseline"
PYTHONPATH=$SCR/se6/baseline $PY examples/bench_step3.py --device cuda --envs "$ENVS" \
    --max-cell-s 30 --out "$OUT/sweep/gpu_before_${NODE}.json" || echo "SWEEP BEFORE rc=$?"
cd "$SCR/se6/gfxdraw"
PYTHONPATH=$SCR/se6/gfxdraw $PY examples/bench_step3.py --device cuda --envs "$ENVS" \
    --max-cell-s 30 --out "$OUT/sweep/gpu_after_${NODE}.json" || echo "SWEEP AFTER rc=$?"

# ---- section-8 profile notes on the same card -------------------------------------------------
for E in Acrobot CartPole MountainCar MountainCarContinuous Pendulum; do
  cd "$SCR/se6/baseline"
  PYTHONPATH=$SCR/se6/baseline $PY scratch/step6/profile_note.py --env $E --device cuda \
      --batch 1 16 64 256 1024 4096 --repeat 10 --warmup 3 --workstream step6/gfxdraw \
      --states-dir "$SCR/se6/baseline/examples/results/step4/states" \
      --out "$OUT/profile_before_gpu" || echo "GPU PROFILE BEFORE $E rc=$?"
  cd "$SCR/se6/gfxdraw"
  PYTHONPATH=$SCR/se6/gfxdraw $PY scratch/step6/profile_note.py --env $E --device cuda \
      --batch 1 16 64 256 1024 4096 --repeat 10 --warmup 3 --workstream step6/gfxdraw \
      --states-dir "$SCR/se6/gfxdraw/examples/results/step4/states" \
      --out "$OUT/profile_after_gpu" || echo "GPU PROFILE AFTER $E rc=$?"
done
echo "GPU JOB DONE"
