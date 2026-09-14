#!/bin/bash
#SBATCH --job-name=se6-tilemap-cpuconfirm
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The certification sweep and the unit suite re-run on the FINAL commit.
#
# The first sweep (se6-tilemap-cpu) measured the tree one commit earlier: the only difference is a
# two-line guard that bounds the composited-atlas cache, and that guard cannot execute in a process
# that renders one tile size. "Cannot execute" is an argument, not a measurement, so this re-measures
# instead of asserting — the certification number should belong to the commit that is offered.
#
# Writes ..._final.json beside the first run rather than over it, so both are inspectable.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/tilemap
BASE=$SCR/se6/baseline
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/tilemap
cd "$WS"
echo "host=$(hostname) job=$SLURM_JOB_ID load=$(cat /proc/loadavg)"
$PY -c "import sys; sys.path.insert(0,'$WS'); import stable_envs; print('tree', stable_envs.__file__)"

for LEG in before after; do
  [ "$LEG" = before ] && TREE=$BASE || TREE=$WS
  PYTHONPATH=$TREE $PY "$TREE/examples/bench_step3.py" --device cpu --threads 1 \
    --envs CraftaxClassic --out "$OUT/sweep/cpu_${LEG}_final.json" || echo "BENCH $LEG rc=$?"
done
$PY scratch/step6/tilemap/tier.py --device cpu --env CraftaxClassic \
  --before "$OUT/sweep/cpu_before_final.json" --after "$OUT/sweep/cpu_after_final.json" \
  --out "$OUT/sweep/tier_cpu_final.json"

PYTHONPATH=$WS $PY -m pytest tests/unit -q 2>&1 | tail -8
PYTHONPATH=$WS $PY -m pytest tests/unit/test_craftax_atlas.py -q 2>&1 | tail -4
echo "TILEMAP_CPU_CONFIRM_DONE"
