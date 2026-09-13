#!/bin/bash
#SBATCH --job-name=se6-raycast3d-cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 6C ticket 1 -- CPU phase attribution for the raycast3d path.
#
# The shared profile-before notes (scratch/step6/profile_note.py) already give op-level counts on
# this partition. They do not answer this workstream's question: every raycast3d scene runs the
# SAME aten ops and they differ by 12x in wall time, so the unit that separates them is the PHASE,
# not the operator. profile_phases.py wraps the renderer's own functions -- _fk, cast_nearest,
# any_blocking, intersect (per primitive kind), shade, composite -- without editing any of them.
#
# batch partition and ONE torch thread, because that is the CPU certification metric.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/raycast3d
export PYTHONPATH=$WS
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d
set -x
cd "$WS"

# The bake needs an offscreen GL context. EGL is what the Step 6 profile-before notes were taken
# with on this partition (node2314), so it is tried first; osmesa is the software fallback. A bake
# that cannot get a context fails every DM Control and OGBench env identically, and SLURM still
# reports COMPLETED, so the probe runs BEFORE the sweep rather than being discovered in the log.
$PY - <<'PROBE' || export MUJOCO_GL=osmesa
import os
import mujoco
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
mujoco.Renderer(m, 16, 16).render()
print("GL OK:", os.environ["MUJOCO_GL"])
PROBE
echo "MUJOCO_GL=$MUJOCO_GL"

run () {  # env [extra args...]
  E=$1; shift
  $PY scratch/step6/raycast3d/profile_phases.py --env "$E" --device cpu --threads 1 \
      --batch 1 16 64 256 --repeat 5 --warmup 2 --cprofile 16 "$@" \
      --states-dir "$WS/examples/results/step4/states" \
      --out "$OUT/profile_phases_cpu" || echo "PHASES $E rc=$?"
}

run DMCReacherHard
run DMCFingerSpin
run DMCBallInCupCatch --fk-counterfactual
run PointmazeLargeNavigate
run AntmazeMediumExplore
run HumanoidmazeMediumNavigate

echo "PHASES_CPU done"
ls -la "$OUT/profile_phases_cpu/" | tail -20
