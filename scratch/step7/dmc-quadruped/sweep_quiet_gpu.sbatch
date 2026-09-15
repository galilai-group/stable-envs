#!/bin/bash
#SBATCH --job-name=se7-dmc-quadruped-quiet-gpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The class-S GPU tier for DMCQuadrupedWalk: the same Step-3 paired protocol on cuda
# (certify_shard_cpu3d.py --device cuda: same-node official renderer interleaved, one-core decode bar
# via bench_vs_decode.decode_baseline), with the four certified trackcom cells as the floor gate on
# the baseline tree before and after. The load wait is capped at 2 h here rather than 6: a GPU held
# idle is the thing the cluster rules forbid, and the cuda tier is bound by the card, not the host
# load -- the comparators (official, decode) are what the guard protects, and they are recorded.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export WS=$SCR/se7/dmc-quadruped
export BASE=$SCR/se7/baseline
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export STABLE_ENVS_GIT_COMMIT=${STABLE_ENVS_GIT_COMMIT:-$(cat "$WS/GIT_COMMIT" 2>/dev/null || true)}
PY=$SCR/venvs/se-step5/bin/python                  # mujoco_warp lives here
OUT=$WS/examples/results/step7/dmc-quadruped/phase1/tier_gpu; mkdir -p "$OUT"; cd "$WS"
NODE=$(hostname -s); FOUR=DMCCheetahRun,DMCHopperHop,DMCWalkerWalk,DMCHumanoidStand
echo "NODE $NODE JOB $SLURM_JOB_ID commit $STABLE_ENVS_GIT_COMMIT merged $(cat "$WS/MERGED_COMMIT" 2>/dev/null) load $(cat /proc/loadavg)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee "$OUT/card_${NODE}.txt"
T0=$(date +%s); LOADLOG="$OUT/load_wait_${NODE}_${SLURM_JOB_ID}.txt"; : > "$LOADLOG"
while true; do
  L=$(cut -d' ' -f1 /proc/loadavg); echo "$(date -u +%FT%TZ) load1=$L" >> "$LOADLOG"
  if awk -v l="$L" 'BEGIN{exit !(l < 12.0)}'; then echo "QUIET at load $L after $(( $(date +%s) - T0 )) s"; break; fi
  if [ $(( $(date +%s) - T0 )) -ge 7200 ]; then
    echo "QUARANTINED-LOAD: gpu node $NODE never fell below 12 in 2 h (last $L); not measuring" | tee "$OUT/QUARANTINED_LOAD_${NODE}_${SLURM_JOB_ID}.txt"
    exit 0
  fi
  sleep 300
done
set -x
echo "LOAD before A $(cat /proc/loadavg)"
PYTHONPATH=$BASE $PY scratch/step6/cpu-3d/certify_shard_cpu3d.py --device cuda --threads 1 --reps 10 --max-cell-s 60 --max-load 12 --envs $FOUR                  --out "$OUT/paired_A_baseline_${NODE}_${SLURM_JOB_ID}.json" 2>&1 | tail -8
echo "LOAD after A / before B $(cat /proc/loadavg)"
PYTHONPATH=$WS   $PY scratch/step6/cpu-3d/certify_shard_cpu3d.py --device cuda --threads 1 --reps 10 --max-cell-s 60 --max-load 12 --envs $FOUR,DMCQuadrupedWalk --out "$OUT/paired_B_branch_${NODE}_${SLURM_JOB_ID}.json"   2>&1 | tail -10
echo "LOAD after B / before C $(cat /proc/loadavg)"
PYTHONPATH=$BASE $PY scratch/step6/cpu-3d/certify_shard_cpu3d.py --device cuda --threads 1 --reps 10 --max-cell-s 60 --max-load 12 --envs $FOUR                  --out "$OUT/paired_C_baseline_${NODE}_${SLURM_JOB_ID}.json" 2>&1 | tail -8
echo "LOAD after C $(cat /proc/loadavg)"
set +x
echo "QUIET GPU SWEEP DONE"
