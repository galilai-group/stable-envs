#!/bin/bash
#SBATCH --job-name=se6-cpu-3d-fetchfeas
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# CPU3D-FETCH-1 (measurement only): a span rasteriser over the ACTUAL Fetch mover triangles on a
# certification core, with backface culling, a setup/bbox/shade phase breakdown, and the un-culled
# arm for comparison, against the frozen CPU_FAST bar at 480 px. The mesh dumps are KEPT (36 MB each,
# not committed) so the measurement is re-runnable without the dump step.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export WS=$SCR/se6/cpu-3d
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$WS
export STABLE_ENVS_GIT_COMMIT=${STABLE_ENVS_GIT_COMMIT:-$(cat "$WS/GIT_COMMIT" 2>/dev/null || true)}
OUT=$WS/examples/results/step6/cpu-3d/fetch_feasibility; mkdir -p "$OUT"; cd "$WS"
NODE=$(hostname -s)
echo "NODE $NODE JOB $SLURM_JOB_ID commit $STABLE_ENVS_GIT_COMMIT"; grep -m1 'model name' /proc/cpuinfo
gcc -O3 -march=native -fno-math-errno -o "$OUT/span_raster" scratch/step6/cpu-3d/span_raster.c -lm || exit 1
gcc -O3 -march=native -fno-math-errno -o "$OUT/cost_floor" scratch/step6/cpu-3d/cost_floor.c -lm || exit 1
python3 - "$OUT/provenance_${NODE}.json" <<PY
import json, os, platform, subprocess, sys
json.dump({"job": os.environ.get("SLURM_JOB_ID"), "host": platform.node(), "load_average_at_start": os.getloadavg(),
           "cpu_model": subprocess.run(["bash", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"], capture_output=True, text=True).stdout.strip(),
           "gcc": subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
           "cflags": "-O3 -march=native -fno-math-errno", "threads": 1, "commit": os.environ.get("STABLE_ENVS_GIT_COMMIT"),
           "mesh_dump_venv": "se-s3b-fetch311 (mujoco 3.11.0, gymnasium-robotics)"}, open(sys.argv[1], "w"), indent=1)
PY
for S in Reach PickAndPlace; do
  [ -s "$OUT/mesh_$S.bin" ] || $SCR/venvs/se-s3b-fetch311/bin/python scratch/step6/cpu-3d/fetch_mesh_dump.py --scene $S --n 16 --out "$OUT/mesh_$S.bin" || echo "DUMP $S FAILED"
  : > "$OUT/span_raster_${S}_${NODE}.jsonl"
  for MODE in full nocull setup bbox; do for REP in 1 2; do "$OUT/span_raster" "$OUT/mesh_$S.bin" 5 $MODE | tee -a "$OUT/span_raster_${S}_${NODE}.jsonl"; done; done
done
: > "$OUT/cost_floor_tri_${NODE}.jsonl"
for T in 10000 30000 100000; do for SZ in 3 6 12; do "$OUT/cost_floor" tri 480 $T $SZ | tee -a "$OUT/cost_floor_tri_${NODE}.jsonl"; done; done
rm -f "$OUT/span_raster" "$OUT/cost_floor"
echo "load at end $(cat /proc/loadavg)"; echo "FETCH FEASIBILITY DONE"
