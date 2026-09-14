#!/bin/bash
#SBATCH --job-name=se6-verify-trackcom-cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The four DM Control trackcom cells on CPU, under the backend cpu-3d just merged. One batch node,
# one torch thread.
#
# TWO REFERENCE ARMS, because "the visual tier passes" and "the visual tier cannot be judged here"
# are different answers and only one of them is available in each interpreter:
#
#   se-step6   mujoco 3.12 + dm_control 1.0.45, NO mujoco_warp. The coordinator named it, and the
#              absence of mujoco_warp is the point: if these four cells render on CPU here, it is
#              cpu3d.py doing it and not MJWarp answering. Under 3.12 the visual checks are BLOCKED
#              by texture-fidelity's provenance gate, by design.
#   se-tex311  mujoco 3.11 + dm_control 1.0.44, no warp either -- the pair these cells' visual_fidelity
#              was DECLARED against. cpu3d.py needs no warp, so unlike the cuda path this arm is
#              actually runnable on CPU, and it is the arm in which the declared bounds mean something.
#
# The baseline tree has no CPU path for these cells at all, so the baseline leg is expected to fail
# to render. That is the measurement, not an error: it is what "cpu-3d adds a CPU backend" looks like
# from the outside, and it is the same shape as raster2d's PushT/SimplePointMaze result.
set -uo pipefail
export SCR=/oscar/scratch/$USER
TREE=${TREE:-$SCR/se6/verify5}
source "$TREE/scratch/step6/verify/_env.sh" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
TAG=${TAG:-_trackcom}
CFG=${CFG:-DMCCheetahRun,DMCHopperHop,DMCHumanoidStand,DMCWalkerWalk}
OUT=$OUTROOT/cpu
mkdir -p "$OUT"
cd "$TREE" || exit 1
banner
echo "configs: $CFG"

for ARM in se-step6 se-tex311; do
  PY=$SCR/venvs/$ARM/bin/python
  echo "######################## CPU arm $ARM"
  $PY -c "import mujoco,sys;print(' mujoco',mujoco.__version__);
try:
    import mujoco_warp; print(' mujoco_warp PRESENT -- this arm cannot prove a CPU backend'); sys.exit(0)
except ImportError: print(' mujoco_warp absent: a CPU render here is cpu3d.py')" 2>&1 | tail -3
  PYTHONPATH=$TREE $PY examples/fidelity_step4.py --configs "$CFG" --device cpu \
      --states-dir "$TREE/examples/results/step4/states" \
      --out "$OUT/trackcom_cpu_${ARM}${TAG}.json" 2>&1 | tail -20
  echo "ARM $ARM rc=$?"
  # Gate A for this arm, through the certifier that owns the bounds.
  CDIR=$OUT/certify_${ARM}${TAG}; mkdir -p "$CDIR/conformance"
  cp -f "$OUT/trackcom_cpu_${ARM}${TAG}.json" "$CDIR/conformance/dmc.json" 2>/dev/null
  PYTHONPATH=$TREE $PY scratch/step4/certify.py --shards "$CDIR" --out "$CDIR" \
      --doc "$SCR/logs/trackcom_${ARM}_${SLURM_JOB_ID}.md" 2>&1 | tail -3
done

echo "######################## does the BASELINE tree render these on CPU at all?"
PYTHONPATH=$BASE $SCR/venvs/se-step6/bin/python "$TREE/scratch/step6/raycast3d/replay_step4.py" \
    --device cpu --configs ${CFG//,/ } --out "$OUT/trackcom_cpu_baseline${TAG}.json" 2>&1 | tail -10
echo "BASELINE rc=$?"
PYTHONPATH=$TREE $SCR/venvs/se-step6/bin/python "$TREE/scratch/step6/raycast3d/replay_step4.py" \
    --device cpu --configs ${CFG//,/ } --out "$OUT/trackcom_cpu_integration${TAG}.json" 2>&1 | tail -10
echo "INTEGRATION rc=$?"

echo "######################## cpu-3d's own unit tests"
PYTHONPATH=$TREE $SCR/venvs/se-step6/bin/python -m pytest tests/unit -v -p no:randomly \
    2>&1 | keep "$OUT/test_unit${TAG}.results.txt"
echo "TRACKCOM_CPU_DONE $(date -u +%FT%TZ)"
