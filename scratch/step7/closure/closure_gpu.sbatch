#!/bin/bash
#SBATCH --job-name=se7-closure-gpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
# gpu2108/gpu2113: the OFFICIAL OGBench renderer is nondeterministic at 1/255 on these two (NODE-1,
# Step 6) and the suite aborts its own session over it. gpu2110/gpu2116: Step-7 exclude list.
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
#
# THE STEP-7 CLOSURE SWEEP, GPU HALF, in ONE allocation on ONE card.
#
# Four stages, in decreasing order of what a truncated job would cost:
#   1. the 49 frozen Step-4 render-distinct configurations replayed on cuda against their frozen
#      digests, each shard in the interpreter it was frozen in, then Gate A through
#      scratch/step4/certify.py and the field-by-field diff in compare49_step7.py
#   2. the six Mujoco locomotion nvdiff cells re-rendered and compared to the Step-7 re-freeze
#      artifact they were re-baselined to, plus their Gate A
#   3. Gate A on cuda for the seven cells Step 7 added, on their committed Step-7 corpora
#   4. tests/gpu -- test_conformance, test_reproducibility, test_fast_paths -- in every interpreter
#      that can reach a different part of the registry, so a green run is never green on a family it
#      never constructed
#
# Every stage writes its artifacts as it goes and no stage can abort another: `set +e` throughout and
# each stage guarded, because a 12-hour job that loses stage 1 because stage 4 crashed is worse than
# no job. Stage boundaries are timestamped so the next run can be budgeted from this one.
set -uo pipefail
export SCR=/oscar/scratch/$USER
TREE=${TREE:-$SCR/se7/closure}
source "$TREE/scratch/step7/closure/_env.sh" || exit 1
STAGES=${STAGES:-"1 2 3 5 4"}
TAG=${TAG:-}
OUT=$OUTROOT
STATES=$SCR/se7/closure_states
cd "$TREE" || exit 1
banner
echo "STAGES='$STAGES' TAG='$TAG' OUT=$OUT"
mkdir -p "$OUT/replay" "$OUT/refreeze" "$OUT/gate_a" "$OUT/gpu_tests" "$STATES"
set +e

has () { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# ---------------------------------------------------------------- 1. the 49-configuration replay
if has 1; then
  stage 1 "frozen Step-4 replay, 49 render-distinct configs, cuda"
  for S in dmc ogb raster swm fetch; do
    PY=$(venv_for "$S")/bin/python
    echo "#### shard $S via $PY"
    $PY -c "import torch,sys;print('  py',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.cuda.is_available())"
    PYTHONPATH=$TREE $PY examples/fidelity_step4.py --shard "$S" --device cuda \
        --states-dir "$TREE/examples/results/step4/states" \
        --out "$OUT/replay/${S}${TAG}.json" 2>&1 | tail -30
    echo "SHARD $S rc=${PIPESTATUS[0]} -> $OUT/replay/${S}${TAG}.json"
  done
  # The four visual-tier DM Control cells carry the 3.11 reference pair their declaration names, so
  # they are replayed a second time in that pair and kept in a shard of their own (compare49 reads
  # `dmcvis`); certify.py reads the reference version off the SHARD, so putting those rows in the
  # 3.12 document would have the certifier block the very numbers it was handed.
  echo "#### shard dmcvis (declared 3.11 pair) via $(venv_for vis311)/bin/python"
  PYTHONPATH=$TREE $(venv_for vis311)/bin/python examples/fidelity_step4.py \
      --configs DMCCheetahRun,DMCHopperHop,DMCWalkerWalk,DMCHumanoidStand --device cuda \
      --states-dir "$TREE/examples/results/step4/states" \
      --out "$OUT/replay/dmcvis${TAG}.json" 2>&1 | tail -20

  stage 1b "Gate A over the replayed shards, through scratch/step4/certify.py"
  CDIR=$OUT/replay/certify${TAG}
  mkdir -p "$CDIR/conformance"
  for S in dmc ogb raster swm fetch dmcvis; do
    cp -f "$OUT/replay/${S}${TAG}.json" "$CDIR/conformance/${S}.json" 2>/dev/null
  done
  PYTHONPATH=$TREE $(venv_for dmc)/bin/python scratch/step4/certify.py \
      --shards "$CDIR" --out "$CDIR" --doc "$SCR/logs/closure_step4_doc${TAG}_${SLURM_JOB_ID}.md" \
      2>&1 | tail -6
  echo "GATE_A_RC=$?"

  stage 1c "field-by-field diff against the frozen Step-4 oracle"
  PYTHONPATH=$TREE $(venv_for dmc)/bin/python scratch/step7/closure/compare49_step7.py \
      --replay "$OUT/replay" --suffix "$TAG" --gate-a "$CDIR/certification.json" \
      --out "$OUT/replay" 2>&1 | tail -80
fi

# ------------------------------------------------ 2. the six locomotion nvdiff cells vs re-freeze
if has 2; then
  stage 2 "six Mujoco locomotion nvdiff cells vs the Step-7 re-freeze artifact"
  PY=$(venv_for rocket)/bin/python
  PYTHONPATH=$TREE $PY scratch/step7/rocket-landing/refreeze_locomotion.py \
      --phase render --tag branch --out "$OUT/refreeze" 2>&1 | tail -30
  PYTHONPATH=$TREE $PY scratch/step7/rocket-landing/refreeze_locomotion.py \
      --phase gatea --out "$OUT/refreeze" 2>&1 | tail -30
  PYTHONPATH=$TREE $PY scratch/step7/closure/refreeze_check.py \
      --fresh "$OUT/refreeze/branch.json" --out "$OUT/refreeze/refreeze_check.json" 2>&1 | tail -20
fi

# ---------------------------------------------------------- 3. Gate A on the seven Step-7 cells
if has 3; then
  stage 3 "Gate A, cuda, on the seven Step-7 cells over their committed corpora"
  PYTHONPATH=$TREE $(venv_for step6)/bin/python scratch/step7/closure/stage_corpora.py \
      --out "$STATES" 2>&1 | tail -12
  # One interpreter per cell family, each the one that cell's declaration names.
  run_gate () {  # run_gate <venv-key> <cells> <extra PYTHONPATH>
    local key="$1" cells="$2" extra="${3:-}"
    local py; py=$(venv_for "$key")/bin/python
    echo "#### gate A cuda [$key] $cells"
    PYTHONPATH=$TREE${extra:+:$extra} $py scratch/step7/closure/gate_a_new_cells.py \
        --cells "$cells" --device cuda --states-dir "$STATES" \
        --out "$OUT/gate_a/cuda_${key}${TAG}.json" 2>&1 | tail -20
  }
  run_gate craftax  "CraftaxClassic,CraftaxFull"
  run_gate vis311   "DMCManipulatorBringBall,DMCQuadrupedWalk"
  run_gate ogbmanip "OGBCube,OGBScene" "$SWMSRC"
  run_gate rocket   "PFRocketLanding"
fi

# ------------------------------------- 5. the cells that refuse device='cpu' by design
if has 5; then
  stage 5 "the honest cpu refusal: OGBCube, OGBScene, PFRocketLanding"
  # Asked HERE, on the card, and not on a batch node: on a batch node mujoco_warp and nvdiffrast do
  # not import at all, so a refusal and a missing module are indistinguishable. The question is
  # whether device='cpu' raises on a machine where the CUDA backend IS available -- i.e. whether the
  # refusal is honest or a silent fallback.
  PYTHONPATH=$TREE:$SWMSRC $(venv_for ogbmanip)/bin/python scratch/step7/closure/gate_a_new_cells.py \
      --cells OGBCube,OGBScene --device cpu --states-dir "$STATES" \
      --out "$OUT/gate_a/cpu_ogbmanip${TAG}.json" 2>&1 | tail -12
  PYTHONPATH=$TREE $(venv_for rocket)/bin/python scratch/step7/closure/gate_a_new_cells.py \
      --cells PFRocketLanding --device cpu --states-dir "$STATES" \
      --out "$OUT/gate_a/cpu_rocket${TAG}.json" 2>&1 | tail -12
fi

# ------------------------------------------------------------------------------ 4. the GPU suite
if has 4; then
  stage 4 "tests/gpu across the interpreters that reach different parts of the registry"
  FILES=${FILES:-"tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py tests/gpu/test_fast_paths.py"}
  # se6-verify is the Step-6 broad interpreter (se-step6 + se-step3 + se-step5 + the swm checkout):
  # dm_control, ogbench, gym_pusht, mujoco_warp and pytest in one process. It has no jax, no
  # nvdiffrast and no PyFlyt, so the Craftax and Rocket cells SKIP there and are covered by the
  # three interpreters after it. Running the same three files in each is what makes "nothing else
  # failing" a claim about the whole registry rather than about one venv's reachable subset.
  for KEY in verify rocket craftax ogbmanip; do
    PY=$(venv_for $KEY)/bin/python
    [ -x "$PY" ] || { echo "SKIP venv $KEY (no interpreter)"; continue; }
    $PY -c "import pytest" 2>/dev/null || { echo "SKIP venv $KEY (no pytest)"; continue; }
    EXTRA=""; [ "$KEY" = "ogbmanip" ] && EXTRA=":$SWMSRC"
    echo "======================================== tests/gpu in $KEY"
    $PY -c "import torch,sys;print(' py',sys.version.split()[0],'torch',torch.__version__,torch.cuda.get_device_name(0))"
    # pytest-timeout is installed in se6-verify and se-step6 and NOT in se7-rocket/se7-craftax/
    # se7-ogb. Passing --timeout unconditionally makes pytest exit 4 with a usage error and write
    # NOTHING, which reads as "the suite produced no failures" in a results file that is empty
    # because the suite never ran. So the flag is added only where the plugin exists, and which
    # interpreters got it is printed.
    TO=(); $PY -c "import pytest_timeout" 2>/dev/null && TO=(--timeout=1500)
    echo "pytest-timeout in $KEY: ${TO[*]:-absent (per-test timeout not enforced here)}"
    for F in $FILES; do
      B=$(basename "$F" .py)
      echo "#### $F [$KEY]"
      PYTHONPATH=$TREE$EXTRA timeout 7200 $PY -m pytest "$F" -v -p no:randomly "${TO[@]}" \
          2>&1 | keep "$OUT/gpu_tests/${B}.${KEY}${TAG}.results.txt"
      echo "FILE $F [$KEY] exit=${PIPESTATUS[0]}"
    done
  done
fi

echo; echo "CLOSURE_GPU_DONE $(date -u +%FT%TZ)"
