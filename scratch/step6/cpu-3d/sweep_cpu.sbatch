#!/bin/bash
#SBATCH --job-name=se6-cpu-3d-sweep
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Sweep only, with the load guard ON: a certifiable all-cell shard needs a node whose co-tenants do
# not move the paired reference by more than the 1.25x spread bound. On a loaded node this aborts in
# a second and writes a rejected shard (never silently dropped); resubmit until a quiet node lands.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export WS=$SCR/se6/cpu-3d
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$WS
export STABLE_ENVS_GIT_COMMIT=${STABLE_ENVS_GIT_COMMIT:-$(cat "$WS/GIT_COMMIT" 2>/dev/null || true)}
OUT=$WS/examples/results/step6/cpu-3d/sweep/after; mkdir -p "$OUT"; cd "$WS"
NODE=$(hostname -s); echo "NODE $NODE JOB $SLURM_JOB_ID commit $STABLE_ENVS_GIT_COMMIT load $(cat /proc/loadavg)"; grep -m1 'model name' /proc/cpuinfo
$SCR/venvs/se-step6/bin/python scratch/step6/cpu-3d/certify_shard_cpu3d.py --device cpu --threads 1 --reps 10 --max-cell-s 60 --max-load 12 \
    --envs DMCCheetahRun,DMCHopperHop,DMCWalkerWalk,DMCHumanoidStand --out "$OUT/cpu_1thread_paired_after_${NODE}_${SLURM_JOB_ID}.json" 2>&1 | tail -10
echo "SWEEP rc=$?"; echo "SWEEP JOB DONE"
