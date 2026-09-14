#!/bin/bash
#SBATCH --job-name=se6-lance-qprobe
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --constraint=cascade
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch PIP_CACHE_DIR=$SCR/.cache/pip
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/lance
export PYTHONPATH=$WS
PY=$SCR/venvs/se6-verify/bin/python
set -x
cd "$WS"
$PY scratch/step6/lance/quality_probe.py --frames-dir $SCR/se6/lance_frames --chunk 256 --reps 8 \
  --out $WS/examples/results/step6/lance/quality_probe_${SLURMD_NODENAME}_${SLURM_JOB_ID}.json \
  2>&1 || echo "QPROBE rc=$?"
echo "QPROBE done"
