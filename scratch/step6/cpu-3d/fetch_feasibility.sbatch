#!/bin/bash
#SBATCH --job-name=se6-cpu-3d-fetchfeas
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# CPU3D-FETCH-1 (measurement only): a span rasteriser over the ACTUAL Fetch mover triangles on a
# certification core, against the frozen CPU_FAST bar at 480 px (decode 776-881 fps/core).
set -uo pipefail
export SCR=/oscar/scratch/$USER
export WS=$SCR/se6/cpu-3d
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$WS
OUT=$WS/examples/results/step6/cpu-3d/fetch_feasibility; mkdir -p "$OUT"; cd "$WS"
echo "NODE $(hostname -s) JOB $SLURM_JOB_ID"; grep -m1 'model name' /proc/cpuinfo
gcc -O3 -march=native -fno-math-errno -o "$OUT/span_raster" scratch/step6/cpu-3d/span_raster.c -lm || exit 1
for S in Reach PickAndPlace; do
  $SCR/venvs/se-s3b-fetch311/bin/python scratch/step6/cpu-3d/fetch_mesh_dump.py --scene $S --n 16 --out "$OUT/mesh_$S.bin" || echo "DUMP $S FAILED"
  "$OUT/span_raster" "$OUT/mesh_$S.bin" 5 | tee "$OUT/span_raster_${S}_$(hostname -s).jsonl"
done
for S in Reach PickAndPlace; do "$OUT/span_raster" "$OUT/mesh_$S.bin" 5 | tee -a "$OUT/span_raster_${S}_$(hostname -s).jsonl"; done
rm -f "$OUT"/mesh_*.bin "$OUT/span_raster"
echo "FETCH FEASIBILITY DONE"
