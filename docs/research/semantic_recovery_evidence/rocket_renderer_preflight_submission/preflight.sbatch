#!/bin/bash
#SBATCH --job-name=se-rocket-render-preflight
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -euo pipefail
bundle="$PWD"
sha256sum -c UPLOAD-SHA256SUMS
run="$bundle/run-$SLURM_JOB_ID"
mkdir "$run" "$run/source" "$run/input" "$run/cache"
tar -xzf source.tar.gz -C "$run/source"
tar -xzf input.tar.gz -C "$run/input"
export PYTHONPATH="$run/source:/oscar/scratch/sboughan/venvs/se-vis311/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step6/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step4-swm/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se7-rocket/lib/python3.12/site-packages"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$run/cache" TORCH_EXTENSIONS_DIR="$run/cache/torch"
nvidia-smi > "$run/nvidia-smi.txt"
cd "$run/source"
status=0
/oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python scripts/research_rocket_production/renderer_preflight.py gpu --source "$run/input" --output "$run/results" > "$run/driver.log" 2>&1 || status=$?
printf '{"exit_code":%s,"job":"%s"}\n' "$status" "$SLURM_JOB_ID" > "$run/exit.json"
cd "$run"
tar -czf "$bundle/evidence-$SLURM_JOB_ID.tar.gz" results driver.log nvidia-smi.txt exit.json
exit "$status"
