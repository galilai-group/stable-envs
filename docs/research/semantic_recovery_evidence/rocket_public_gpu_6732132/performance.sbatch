#!/bin/bash
#SBATCH --job-name=se-rocket-performance
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090|l40s|a40|a5000|a5500|quadrortx
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/rocket-public-d2516cb6/slurm-%j.log
set -euo pipefail
root=/oscar/scratch/sboughan/se-semantic-recovery/rocket-public-d2516cb6
run=$root/run-$SLURM_JOB_ID
mkdir -p "$run/source" "$run/cache"
tar -xzf "$root/source.tar.gz" -C "$run/source"
export PYTHONPATH="$run/source:/oscar/scratch/sboughan/venvs/se-vis311/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step6/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step4-swm/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se7-rocket/lib/python3.12/site-packages"
export XDG_CACHE_HOME="$run/cache"
export TORCH_EXTENSIONS_DIR="$run/cache/torch-extensions"
export CUDA_CACHE_PATH="$run/cache/cuda"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
nvidia-smi > "$run/nvidia-smi.txt"
lscpu > "$run/lscpu.txt"
cd "$run/source"
set +e
/oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python scripts/research_rocket_production/replay_performance.py --gpu --output "$run/performance" > "$run/performance.log" 2>&1
code=$?
set -e
printf '{"job_id": "%s", "exit_code": %s}\n' "$SLURM_JOB_ID" "$code" > "$run/exit.json"
exit "$code"
