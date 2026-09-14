#!/bin/bash
#SBATCH --job-name=se6-gfxdraw-cpufloor2
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The same floors, with the torch arm swept over BATCH on the same node.
#
# The first run timed torch at B=8 and the compiled walks at their batch-independent cost, which
# credits a fused kernel with the dispatch amortisation that batching already provides. The
# certified number is at the best batch, so that is the comparison that decides GFXDRAW-2-CPU.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export PIP_CACHE_DIR=$SCR/.cache/pip NUMBA_CACHE_DIR=$SCR/.cache/numba
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
NB=$SCR/venvs/se6-gfxdraw-box2d/bin/python
OUT=$SCR/se6/gfxdraw/examples/results/step6/gfxdraw/characterize
NODE=$(hostname -s)
mkdir -p "$OUT"
cd "$SCR/se6/gfxdraw"
echo "NODE $NODE JOB $SLURM_JOB_ID"; uptime; lscpu | grep "Model name"
set -x
PYTHONPATH=$SCR/se6/gfxdraw "$NB" scratch/step6/gfxdraw/cost_floor.py \
    --env CartPole Acrobot Pendulum MountainCar --batch 8 --reps 5 \
    --torch-batches 1,16,64,256 --out "$OUT/cost_floor_batched_${NODE}.json"
echo "COST FLOOR BATCHED rc=$?"
echo "CPU FLOOR2 JOB DONE"
