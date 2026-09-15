#!/bin/bash
#SBATCH --job-name=se7-closure-cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
#
# THE STEP-7 CLOSURE SWEEP, CPU HALF, on the certification host.
#
#   A. Gate A on device='cpu' for the Step-7 cells that HAVE a CPU backend (CraftaxClassic,
#      CraftaxFull, DMCManipulatorBringBall, DMCQuadrupedWalk), on the same committed corpora the
#      cuda arm uses, so the device coverage claim is measured on both devices rather than assumed
#      from one.
#   B. the three cells that raise on cpu BY DESIGN (OGBCube, OGBScene, PFRocketLanding): the raise is
#      CAPTURED -- type and message -- rather than being reported as an absence. A silent CUDA
#      fallback and an honest refusal look identical in a table that only records "no number".
#   C. tests/unit in each interpreter that can reach a different part of the registry.
#
# No timing is taken here and none of these numbers is a certification measurement.
set -uo pipefail
export SCR=/oscar/scratch/$USER
TREE=${TREE:-$SCR/se7/closure}
source "$TREE/scratch/step7/closure/_env.sh" || exit 1
# MUJOCO_GL stays egl (set in _env.sh). A batch node has no GPU, but EGL is what the Step-6 CPU
# verification jobs used on these same nodes and it is what lets dm_control import at all here:
# MUJOCO_GL=osmesa makes stable_envs.envs.dmcontrol fail to import with
# "AttributeError: 'NoneType' object has no attribute 'glGetError'", which silently removes every
# DM Control cell from the registry and would have turned a missing backend into an empty table.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
STAGES=${STAGES:-"A C"}
TAG=${TAG:-}
OUT=$OUTROOT
STATES=$SCR/se7/closure_states
cd "$TREE" || exit 1
banner
mkdir -p "$OUT/gate_a" "$OUT/unit_tests" "$STATES"
set +e
has () { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

PYTHONPATH=$TREE $(venv_for step6)/bin/python scratch/step7/closure/stage_corpora.py --out "$STATES" 2>&1 | tail -10

if has A; then
  stage A "Gate A on device=cpu for the Step-7 cells with a CPU backend"
  run_gate () {
    local key="$1" cells="$2" extra="${3:-}"
    local py; py=$(venv_for "$key")/bin/python
    echo "#### gate A cpu [$key] $cells"
    PYTHONPATH=$TREE${extra:+:$extra} $py scratch/step7/closure/gate_a_new_cells.py \
        --cells "$cells" --device cpu --states-dir "$STATES" \
        --out "$OUT/gate_a/cpu_${key}${TAG}.json" 2>&1 | tail -20
  }
  run_gate craftax "CraftaxClassic,CraftaxFull"
  run_gate vis311  "DMCManipulatorBringBall,DMCQuadrupedWalk"
fi

# Stage B -- the three cells that refuse device='cpu' BY DESIGN (OGBCube, OGBScene,
# PFRocketLanding) -- runs in the GPU job instead. mujoco_warp and nvdiffrast do not import on a
# batch node, so asking for the cpu refusal here would have recorded "module missing" where the
# question is "does the class refuse instead of silently falling back to CUDA". The refusal has to
# be asked on a machine where the CUDA backend is available and is nevertheless not used.

if has C; then
  stage C "tests/unit on the certification host"
  for KEY in step6 craftax rocket; do
    PY=$(venv_for $KEY)/bin/python
    [ -x "$PY" ] || { echo "SKIP venv $KEY"; continue; }
    $PY -c "import pytest" 2>/dev/null || { echo "SKIP venv $KEY (no pytest)"; continue; }
    EXTRA=""; [ "$KEY" = "ogbmanip" ] && EXTRA=":$SWMSRC"
    echo "======================================== tests/unit in $KEY"
    PYTHONPATH=$TREE$EXTRA timeout 7200 $PY -m pytest tests/unit -q -p no:randomly --timeout=900 \
        2>&1 | keep "$OUT/unit_tests/unit.${KEY}${TAG}.results.txt"
    echo "tests/unit [$KEY] exit=${PIPESTATUS[0]}"
  done
fi

echo; echo "CLOSURE_CPU_DONE $(date -u +%FT%TZ)"
