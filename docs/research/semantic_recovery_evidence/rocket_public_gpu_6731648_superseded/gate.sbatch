#!/bin/bash
#SBATCH --job-name=se-rocket-public
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/rocket-public-be53af62/slurm-%j.log
set -euo pipefail
root=/oscar/scratch/sboughan/se-semantic-recovery/rocket-public-be53af62
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
/oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python scripts/research_rocket_production/public_gate.py --gpu --output "$run/results" > "$run/driver.log" 2>&1
code=$?
set -e
printf '{"job_id": "%s", "exit_code": %s}\n' "$SLURM_JOB_ID" "$code" > "$run/exit.json"
exit "$code"
