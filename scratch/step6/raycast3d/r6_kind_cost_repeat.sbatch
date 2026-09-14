#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r6cost
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Three INDEPENDENT runs of the per-kind cost benchmark in the shape that ships. One run cannot say
# how much of its own ratio is noise, and the first two runs disagreed about that badly enough
# (worst_rsd 0.076 against 0.25) to flip every kind in the table. Three runs give an error bar on the
# ratio itself, which is the number the predicate consumes.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/raycast3d
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d/experiments
set -x
cd "$WS"
lscpu | grep -E "^Model name" | head -1
cat /proc/loadavg
for R in 1 2 3; do
  PYTHONPATH=$WS $PY scratch/step6/raycast3d/r5_kind_cost.py --threads 1 --shape nearest \
      --repeat 40 --warmup 8 --out "$OUT/r6_kind_cost_nearest_run$R.json" || echo "RUN$R rc=$?"
done
PYTHONPATH=$WS $PY scratch/step6/raycast3d/r6_kind_cost_agg.py \
    "$OUT/r6_kind_cost_nearest_run1.json" "$OUT/r6_kind_cost_nearest_run2.json" \
    "$OUT/r6_kind_cost_nearest_run3.json"
echo "R6_COST done"
