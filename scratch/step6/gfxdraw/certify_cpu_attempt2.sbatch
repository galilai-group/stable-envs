#!/bin/bash
#SBATCH --job-name=se6-gfxdraw-cpuatt
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Attempt 2/3 at a self-contained 1-thread CPU shard, WITHOUT --exclusive, with a pre-flight gate.
#
# Why not exclusive. The normal QOS caps a user at cpu=64 and this partition's nodes are 48 to 192
# cores, so an exclusive allocation can only ever land on a 48-core node -- on a 192-core one it
# would charge 192 against the 64 cap and pend forever. The 48-core nodes are saturated: attempt 1
# (6353993) has been pending since 21:50 and its estimate slipped from 02:03 to 05:34.
#
# Why the load-average gate is the wrong instrument here and the pre-flight is the right one. On a
# SHARED node, total load says nothing about whether OUR core is contended: a 192-core node with 64
# CPUs busy reads a load of 64 and may be a perfectly quiet place to run one thread, while the
# --max-load 4 bound written for an exclusive node would abort on it every time. So this measures
# the quantity we actually care about instead of a proxy for it: time the official renderer twelve
# times before committing to the sweep, and abort if those readings already vary more than the
# acceptance bound. Seconds to find out, against a whole job.
#
# The acceptance criterion is unchanged and is still the thing that decides: reference spread over
# the full interleaved run <= 1.25x, recorded in the shard as `certifiable`.
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
echo "NODE $NODE JOB $SLURM_JOB_ID attempt ${ATTEMPT:-2}"
echo "loadavg $(cat /proc/loadavg)  cores $(nproc)  node cores $(getconf _NPROCESSORS_ONLN)"
set -x
PYTHONPATH=$SCR/se6/gfxdraw $PY scratch/step6/gfxdraw/certify_shard.py \
    --device cpu --threads 1 --max-cell-s 12 --preflight-bursts 12 \
    --out "$OUT/cpu_1thread_paired_after_${NODE}.json"
RC=$?
set +x
echo "CERTIFY SHARD rc=$RC (0 clean, 3 rejected on run spread, 4 rejected on pre-flight)"
echo "ATTEMPT ${ATTEMPT:-2} NODE $NODE VERDICT $([ $RC -eq 0 ] && echo CLEAN || echo REJECTED)"
echo "CPU ATTEMPT JOB DONE"
