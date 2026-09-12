#!/bin/bash
#SBATCH --job-name=se-step3-cpux86
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch
export MUJOCO_GL=egl MPLBACKEND=Agg
# warp resolves its kernel cache from its OWN path, not XDG_CACHE_HOME, and lands in HOME --
# which is already SOFT_EXCEEDED at 100.88/100 GB with a 10-day grace. Measured at 2.8 MB so it
# is not what fills HOME, but nothing in this campaign has a reason to write there.
export WARP_CACHE_PATH=$SCR/.cache/warp
export PYTHONPATH=$SCR/stable-envs
cd $SCR/stable-envs || exit 1
echo "node=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK load=$(cat /proc/loadavg)"
lscpu | grep "Model name"
# The certified CPU number is single-threaded (Step 2 section 5.3). The all-core run is a SCALING
# result and is taken separately, into its own shard, so the two can never be confused for each other.
"$SCR/venvs/se-step3/bin/python" examples/bench_step3.py --device cpu --threads "${THREADS:-1}" \
    --out "$SCR/step3-out/cpu/${OUTNAME:-oscar_x86_1thread}.json" \
    --envs "${ENVS:-auto}" --resume --max-cell-s "${MAXCELL:-12}"
echo "CPU SWEEP DONE rc=$?"
