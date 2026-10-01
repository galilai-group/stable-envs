#!/bin/bash
#SBATCH --job-name=se-step-slices
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090|l40s|a40|a5000|a5500
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
set -euo pipefail
stage=/oscar/scratch/sboughan/se-semantic-recovery/step-slices-20261001-v2
mkdir "$stage/source"
tar -xzf "$stage/source.tar.gz" -C "$stage/source"
cd "$stage/source"
export PYTHONPATH="$PWD" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$stage/cache" MPLCONFIGDIR="$stage/cache/matplotlib" MPLBACKEND=Agg
py=/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python
nvidia-smi
"$py" -m pytest tests/unit/test_step_capture.py tests/unit/test_native_classic_semantics.py tests/unit/test_native_classic_remaining_semantics.py tests/unit/test_public_api.py -q > "$stage/api-tests.log" 2>&1
"$py" scripts/benchmarks/online_stepping/fetch_probe.py --baseline-fetch "$stage/baseline_fetch.py" --device cpu --output "$stage/fetch-cpu.json" --gate-only --gate-batch 64
"$py" scripts/benchmarks/online_stepping/fetch_probe.py --baseline-fetch "$stage/baseline_fetch.py" --device cuda --output "$stage/fetch-gpu.json" --gate-only --gate-batch 64
echo STEP_SLICES_PASS
