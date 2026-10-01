#!/bin/bash
#SBATCH --job-name=se-step-native-gpu
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
set -euo pipefail
stage=/oscar/scratch/sboughan/se-semantic-recovery/online-step-gpu-smoke-20261001
source=/oscar/scratch/sboughan/se-semantic-recovery/online-step-20261001-v3/source
export PYTHONPATH="$source" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME="$stage/cache" TRITON_CACHE_DIR="$stage/cache/triton"
cd "$source"
/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python -m pytest "$stage/test_step_capture.py" -k gpu_sequence -q > "$stage/tests.log" 2>&1
echo NATIVE_GPU_STEP_CAPTURE_PASS
