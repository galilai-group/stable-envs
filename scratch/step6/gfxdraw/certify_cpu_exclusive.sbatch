#!/bin/bash
#SBATCH --job-name=se6-gfxdraw-cpuexcl
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# One self-contained 1-thread CPU certification shard, on a node with no co-tenants.
#
# Two attempts at this have already failed for the same reason in two different disguises:
# node2338's block-timed reference collapsed 2.3x (SLURM 6346431) and node2339's INTERLEAVED
# reference varied 4.52-10.37x (SLURM 6353689). Both were shared nodes. The measurement is a ratio
# to a single-threaded pygame loop, so anything else running on the socket is inside the number.
#
# --exclusive is the fix: OverSubscribe=NO means jobs do not share a core, but they do share a
# NODE, and a 48-core batch node with eleven other tenants is what produced both failures. The QOS
# allows cpu=64 per user, so one whole 48-core node fits.
#
# The sleep is not padding. getloadavg() is a one-minute average, so immediately after an
# allocation it still describes the PREVIOUS tenant; the check would pass on a node that is only
# just quiet and fail on one that is only just busy. Sleeping past the window makes the reading
# describe this job's node.
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
echo "NODE $NODE JOB $SLURM_JOB_ID  attempt ${ATTEMPT:-?}"
echo "load at allocation: $(cat /proc/loadavg)"
sleep 75
echo "load after settling: $(cat /proc/loadavg)"
nproc
set -x
PYTHONPATH=$SCR/se6/gfxdraw $PY scratch/step6/gfxdraw/certify_shard.py \
    --device cpu --threads 1 --max-cell-s 12 --max-load 4 \
    --out "$OUT/cpu_1thread_paired_after_${NODE}.json"
RC=$?
set +x
echo "CERTIFY SHARD rc=$RC (0 clean, 2 aborted on load, 3 rejected on spread)"
echo "ATTEMPT ${ATTEMPT:-?} NODE $NODE VERDICT $([ $RC -eq 0 ] && echo CLEAN || echo REJECTED)"
echo "CPU EXCLUSIVE JOB DONE"
