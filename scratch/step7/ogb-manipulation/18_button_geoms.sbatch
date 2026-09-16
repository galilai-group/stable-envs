#!/bin/bash
#SBATCH --job-name=se7-ogb-manipulation-buttons
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# A model compile and a colour diff -- no GPU, so this does not queue behind the GPU work, and by the
# standing rule it does not run on a login node either.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se7/ogb-manipulation
VENV=$SCR/venvs/se7-ogb
OUT=$WS/examples/results/step7/ogb-manipulation/m6/buttons
export PYTHONPATH=$WS:$SCR/se7/swm-src
mkdir -p "$OUT"
cd "$WS"
for GL in osmesa egl glfw; do
  MUJOCO_GL=$GL "$VENV/bin/python" -c "import mujoco; mujoco.GLContext(8,8); print('GL OK: $GL')" \
    2>/dev/null && { export MUJOCO_GL=$GL; break; }
done
echo "using MUJOCO_GL=${MUJOCO_GL:-unset}"
set -x
"$VENV/bin/python" scratch/step7/ogb-manipulation/probe_button_geoms.py \
  --out "$OUT" --env-id swm/OGBScene-v0 || echo "rc=$?"
cp "/oscar/scratch/$USER/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$OUT/slurm_${SLURM_JOB_ID}.log" 2>/dev/null || true
echo OK
