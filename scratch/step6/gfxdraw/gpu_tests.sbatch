#!/bin/bash
#SBATCH --job-name=se6-gfxdraw-gputests
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The generic conformance and reproducibility suites for the five gfxdraw configurations.
#
# Separate from the main GPU job because that job piped pytest through `tail -25` and dm_control's
# EGL contexts are torn down in __del__ AFTER pytest's summary line, so the last 25 lines were
# "Exception ignored in: <function GLContext.__del__>" repeated and the pass/fail count was gone.
# A test result that scrolled off is not a test result. Here the run goes to a file and the summary
# is grepped out of it.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
PY=$SCR/venvs/se-step6/bin/python
OUT=$SCR/se6/gfxdraw/examples/results/step6/gfxdraw/tests
mkdir -p "$OUT"
cd "$SCR/se6/gfxdraw"
echo "NODE $(hostname -s) JOB $SLURM_JOB_ID"; nvidia-smi --query-gpu=name --format=csv,noheader

K='Acrobot or CartPole or MountainCar or MountainCarContinuous or Pendulum'
PYTHONPATH=$SCR/se6/gfxdraw $PY -m pytest tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py \
    -v -k "$K" --tb=short > "$OUT/conformance_reproducibility.txt" 2>&1
echo "pytest conformance/reproducibility rc=$?"
grep -E "PASSED|FAILED|ERROR|passed|failed" "$OUT/conformance_reproducibility.txt" | tail -40

PYTHONPATH=$SCR/se6/gfxdraw $PY -m pytest tests/gpu/test_gfxdraw_fused.py tests/gpu/test_fast_paths.py \
    -q --tb=short > "$OUT/gfxdraw_fused_fast_paths.txt" 2>&1
echo "pytest fused/fast-paths rc=$?"
grep -E "passed|failed|error" "$OUT/gfxdraw_fused_fast_paths.txt" | tail -5
echo "GPU TESTS JOB DONE"
