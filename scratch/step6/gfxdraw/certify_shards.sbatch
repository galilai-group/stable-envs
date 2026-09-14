#!/bin/bash
#SBATCH --job-name=se6-gfxdraw-shardcpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# A CPU certification shard whose denominator is paired, so the Step 6 report generator has
# something correct to label from. See scratch/step6/gfxdraw/certify_shard.py and
# examples/results/step6/gfxdraw/sweep/superseded_node2338/README.md.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=$SCR/venvs/se-step6/bin/python
OUT=$SCR/se6/gfxdraw/examples/results/step6/gfxdraw/sweep
NODE=$(hostname -s)
mkdir -p "$OUT"
cd "$SCR/se6/gfxdraw"
echo "NODE $NODE JOB $SLURM_JOB_ID"; uptime
set -x
PYTHONPATH=$SCR/se6/gfxdraw $PY scratch/step6/gfxdraw/certify_shard.py \
    --device cpu --threads 1 --max-cell-s 12 \
    --out "$OUT/cpu_1thread_paired_${NODE}.json"
echo "CPU SHARD rc=$?"
echo "SHARD CPU JOB DONE"
