#!/bin/bash
#SBATCH --job-name=se7-dmc-quadruped-sweep
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The paired Step-3-shaped sweep for DMCQuadrupedWalk, alone: certify_cpu.sbatch's step 4 aborted on
# node1913 at load 34.5 (> the 12 guard) before measuring anything -- the shared-batch-node situation
# Step 6 recorded (cpu-3d findings: three load-guarded attempts, best node 1.65x reference spread).
# One attempt with the guard kept; a shard the guard refuses is reported, never relabelled.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export WS=$SCR/se7/dmc-quadruped
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$WS
export STABLE_ENVS_GIT_COMMIT=${STABLE_ENVS_GIT_COMMIT:-$(cat "$WS/GIT_COMMIT" 2>/dev/null || true)}
OUT=$WS/examples/results/step7/dmc-quadruped/phase1/cpu_oscar; mkdir -p "$OUT"; cd "$WS"
NODE=$(hostname -s); echo "NODE $NODE JOB $SLURM_JOB_ID commit $STABLE_ENVS_GIT_COMMIT merged $(cat "$WS/MERGED_COMMIT" 2>/dev/null) load $(cat /proc/loadavg)"; grep -m1 'model name' /proc/cpuinfo
$SCR/venvs/se-step6/bin/python scratch/step6/cpu-3d/certify_shard_cpu3d.py --device cpu --threads 1 --reps 10 --max-cell-s 60 --max-load 12 \
    --envs DMCQuadrupedWalk --out "$OUT/cpu_1thread_paired_${NODE}_${SLURM_JOB_ID}.json" 2>&1 | tail -12
echo "SWEEP rc=$?"; echo "SWEEP JOB DONE"
