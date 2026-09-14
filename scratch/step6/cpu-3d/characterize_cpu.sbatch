#!/bin/bash
#SBATCH --job-name=se6-cpu-3d-char
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 6G cpu-3d Phase 1 characterisation, on a CPU certification node (no GPU, 1 torch thread).
# Four steps in three venvs; each is allowed to fail without taking the others down.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export WS=$SCR/se6/cpu-3d
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$WS
OUT=$WS/examples/results/step6/cpu-3d/characterize
mkdir -p "$OUT"
cd "$WS"
echo "host=$(hostname) job=$SLURM_JOB_ID cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
lscpu | grep -E 'Model name|Thread|Core|Socket|MHz|L2|L3' | head -12

# Which GL works on a node with no GPU: probe EGL (Mesa surfaceless), fall back to osmesa.
export MUJOCO_GL=egl
$SCR/venvs/se-step6/bin/python - <<'PROBE' || export MUJOCO_GL=osmesa
import mujoco, os
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
r = mujoco.Renderer(m, 16, 16); r.update_scene(mujoco.MjData(m)); r.render(); print("GL ok:", os.environ["MUJOCO_GL"])
PROBE
echo "MUJOCO_GL=$MUJOCO_GL"

echo "===== step 1: compiled cost floor"
( cd scratch/step6/cpu-3d && gcc -O3 -march=native -fno-math-errno -o "$OUT/cost_floor" cost_floor.c -lm ) \
  && { for K in 8 19; do "$OUT/cost_floor" dmc 84 $K; done; "$OUT/cost_floor" dmc 224 8;
       for T in 10000 30000 100000; do for S in 3 6 12; do "$OUT/cost_floor" tri 480 $T $S; done; done; } \
  | tee "$OUT/cost_floor_$(hostname -s).jsonl" || echo "STEP1 FAILED rc=$?"

echo "===== step 2: torch/raycast3d/official/decode (se-step6)"
$SCR/venvs/se-step6/bin/python scratch/step6/cpu-3d/cpu_micro.py --out "$OUT" --threads 1 || echo "STEP2 FAILED rc=$?"

echo "===== step 3: mujoco_warp sampling source (se-step5)"
$SCR/venvs/se-step5/bin/python scratch/step6/cpu-3d/mjw_dump.py "$OUT/mujoco_warp_sampling_source.txt" || echo "STEP3 FAILED rc=$?"

echo "===== step 4: Fetch census + same-host official (se-s3b-fetch311)"
$SCR/venvs/se-s3b-fetch311/bin/python scratch/step6/cpu-3d/census_fetch.py --out "$OUT" || echo "STEP4 FAILED rc=$?"

echo "===== done"; ls -la "$OUT"
