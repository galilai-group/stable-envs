#!/bin/bash
#SBATCH --job-name=se-online-step
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
set -euo pipefail
stage=/oscar/scratch/sboughan/se-semantic-recovery/online-step-20261001-v4
py=/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python
cd "$stage/source"
export PYTHONPATH="$PWD" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$stage/cache" MPLCONFIGDIR="$stage/cache/matplotlib" MPLBACKEND=Agg
nvidia-smi
"$py" -m pytest tests/unit/test_step_capture.py tests/unit/test_native_classic_semantics.py tests/unit/test_native_classic_remaining_semantics.py tests/unit/test_public_api.py -q > "$stage/api-tests.log" 2>&1
"$py" scripts/benchmarks/online_stepping/fetch_probe.py --baseline-fetch baseline_fetch.py --device cuda --output "$stage/fetch-gpu.json" --gate-only --gate-batch 64
"$py" scripts/benchmarks/online_stepping/fetch_probe.py --baseline-fetch baseline_fetch.py --device cpu --output "$stage/fetch-cpu.json" --gate-only --gate-batch 64
echo ONLINE_STEP_JOB_PASS
