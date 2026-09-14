#!/bin/bash
#SBATCH --job-name=se6-cpu-3d-certify
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# CPU certification of the four trackcom cells' new CPU backend, 1 torch thread, on a batch node:
#  (i)  Gate-A mandatory + visual checks via examples/fidelity_step4.py, under the DECLARED reference
#       pair (se-vis311: mujoco 3.11.0 + dm_control 1.0.44) and, for the record, under 3.12 (se-step6,
#       where the certifier BLOCKS the reference-judged checks); scratch/step4/certify.py labels both.
#  (ii) cross-backend: CPU frames vs OUR MJWarp frames on the frozen states.
#  (iii) the paired Step-3 sweep (official llvmpipe interleaved, decode bar, spread) + frozen tiers.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export WS=$SCR/se6/cpu-3d
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$WS
OUT=$WS/examples/results/step6/cpu-3d
STATES=$WS/examples/results/step4/states
CONFIGS=DMCCheetahRun,DMCHopperHop,DMCWalkerWalk,DMCHumanoidStand
NODE=$(hostname -s)
cd "$WS"
echo "NODE $NODE JOB $SLURM_JOB_ID load $(cat /proc/loadavg)"; grep -m1 'model name' /proc/cpuinfo
set -x
for ARM in 311 312; do
  if [ "$ARM" = 311 ]; then PY=$SCR/venvs/se-vis311/bin/python; else PY=$SCR/venvs/se-step6/bin/python; fi
  D=$OUT/replay/certify${ARM}_cpu; rm -rf "$D"; mkdir -p "$D/conformance"
  $PY examples/fidelity_step4.py --shard dmc --configs "$CONFIGS" --device cpu --states-dir "$STATES" \
      --out "$D/conformance/dmc.json" 2>&1 | tail -25
  echo "FIDELITY $ARM rc=$?"
  $SCR/venvs/se-step6/bin/python scratch/step4/certify.py --shards "$D" --out "$D" --doc "$D/report.md" 2>&1 | tail -5
  echo "CERTIFY $ARM rc=$?"
done
PY=$SCR/venvs/se-step6/bin/python
$PY scratch/step6/cpu-3d/compare_frozen.py --frames-dir "$SCR/se6/texture-fidelity/examples/results/step6/texture-fidelity/localize312/frames" \
    --n 64 --out "$OUT/replay/cross_backend_$NODE" --sheet 2>&1 | grep -v 'mean \\|diff' | tail -6
echo "CROSS rc=$?"
mkdir -p "$OUT/sweep/after"
$PY scratch/step6/gfxdraw/certify_shard.py --device cpu --threads 1 --max-cell-s 12 --envs "$CONFIGS" \
    --out "$OUT/sweep/after/cpu_1thread_paired_after_${NODE}.json" 2>&1 | tail -12
echo "SWEEP rc=$?"
$PY scratch/step6/cpu-3d/tiers_after.py --shard "$OUT/sweep/after/cpu_1thread_paired_after_${NODE}.json" --out "$OUT/sweep/tiers_after_${NODE}.json"
echo "TIERS rc=$?"
set +x; echo "CERTIFY CPU DONE"
