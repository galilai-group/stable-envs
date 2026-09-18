#!/bin/bash
#SBATCH --job-name=se9-envs-mujoco-oracle
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
#
# GPU/full-venv arm of the Step-9 Workstream-F evidence (the 17 MuJoCo + non-42-id families).
# PREPARED, NOT SUBMITTED: ruling 8 says there are no per-workstream GPU jobs in Step 9, so this is
# a SECTION for the coordinator to fold into the single post-Wave-1 GPU job over all 42 ids.
#
# WHAT IS ACTUALLY OWED HERE, and it is more than "the cuda frames". The CPU arm on the Mac reached
# 24 of this workstream's cells in two venvs. The other TEN families were not reached, and they were
# not reached for two DIFFERENT reasons, which this job separates rather than lumping:
#
#   CUDA-ONLY BY CONSTRUCTION -- metaworld, robocasa, humanoidbench ray-cast their meshes with
#   mujoco_warp's batch raytracer. There is no CPU backend to fall back to and the Mac has no
#   NVIDIA device, so their frames were never rendered on this branch at all: the oracle for them is
#   UNESTABLISHED, not satisfied. humanoidbench additionally needs STABLE_ENVS_HUMANOIDBENCH=1
#   because it is not registered by default (mujoco_warp's batched render is not bit-reproducible
#   inside a long conformance session; see its ImportError text and docs/research/humanoidbench.md).
#
#   MISSING OPTIONAL PACKAGE, CPU-CAPABLE -- carracing and lunarlander need Box2D, griddly needs
#   griddly==1.6.7, jumanji needs jumanji+jax+matplotlib, overcooked needs jaxmarl, pgx needs
#   pgx+resvg_py+Pillow, procgen needs procgen. These would render on a CPU given the package; they
#   are here because this is the tree that HAS the packages, not because they need a GPU.
#
# and one claim that is NEW in Step 9 and is checked NOWHERE today for those ten families:
#
#   OBSERVATION_SPEC vs A REAL FRAME. Env.observation_spec assumes a SQUARE frame unless the env
#   declares frame_shape (env.py:1074). Four of this workstream's families are not square --
#   highway (300x600 for Parking), minihack (336x1264), lunarlander (400x600) and the two
#   tile-grid families griddly and overcooked (grid_h*T x grid_w*T) -- and all of them now declare
#   frame_shape. highway and minihack were found by tests/unit/test_envs_mujoco.py comparing the
#   spec against a real render; the other three could NOT be, because that test SKIPS a family
#   whose optional dependency is absent, which is every local venv. Stage 3 runs that same test
#   where the packages exist, which is the only place the three static declarations become measured.
#
# THE BASELINE COMES FROM THE TREE, NOT FROM A COPY. oracle.py does `git worktree add --detach
# 21cb30678b7a7d0f3344b59f176f33d3f3b09ffc` inside $WS, so $WS MUST be a real git clone whose
# history contains that commit and $TMPDIR must be writable. A missing SHA fails loudly rather than
# silently comparing the branch against itself.
#
# NOTHING IS ASSUMED ABOUT WHICH VENV HAS WHAT. Stage 0 prints, per interpreter, which of this
# workstream's families import and which do not WITH THEIR REASON, and the later stages drive the
# DISCOVERED cell list. A venv that turns out to carry none of them produces a named SKIP, not a
# green run over zero cells -- exit code 5 (no tests collected) is a warning here, never a pass.
set -uo pipefail
SCR=/oscar/scratch/sboughan
export UV_CACHE_DIR=$SCR/cache/uv PIP_CACHE_DIR=$SCR/cache/pip XDG_CACHE_HOME=$SCR/cache/xdg
export TORCH_HOME=$SCR/cache/torch WARP_CACHE_PATH=$SCR/cache/warp HF_HOME=$SCR/cache/hf
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

WS=$SCR/se9/envs-mujoco
OUT=$WS/examples/results/step9/envs-mujoco
TAG=gpu_${SLURM_JOB_ID:-local}
mkdir -p "$OUT" "$SCR/logs"
cd "$WS" || exit 1

#: The interpreters worth asking, and what each is expected to carry. Order matters only for the log.
#:   se-vis311   mujoco 3.11 + mujoco_warp 3.11 -- the ONLY arm where metaworld/robocasa/
#:               humanoidbench can render at all
#:   se-step6    torch cu126, general: the tile/raster families, if their packages are installed
#:   se7-ogb     ogbench + mjwarp; carries mujoco, so the four mujoco_locomotion cells
#:   se-s3b-fetch311  gymnasium-robotics: the four maze2d PointMaze cells
#:   se7-craftax-gpu  jax: pgx / overcooked / jumanji, if those packages were added
VENVS="se-vis311 se-step6 se7-ogb se-s3b-fetch311 se7-craftax-gpu"

#: This workstream's seventeen module basenames. A family is "mine" if its class's __module__ ends
#: in one of these; asked this way rather than by cell name so a family that gained a cell since
#: Step 8 is still counted.
FAMILIES="mujoco_locomotion humanoidbench metaworld robocasa maze2d carracing crafter griddly highway jumanji lunarlander minigrid minihack overcooked pgx procgen sokoban"

# humanoidbench is deliberately unregistered; ask for it explicitly so this job can measure it.
export STABLE_ENVS_HUMANOIDBENCH=1

{ echo "host $(hostname)"; echo "job ${SLURM_JOB_ID:-none}"; echo "date $(date -Is)"
  echo "head $(git -C "$WS" rev-parse HEAD 2>/dev/null)"
  echo "baseline_present $(git -C "$WS" cat-file -e 21cb30678b7a7d0f3344b59f176f33d3f3b09ffc^{commit} && echo yes || echo NO)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  for v in $VENVS; do
    p=$SCR/venvs/$v/bin/python
    if [ -x "$p" ]; then
      echo "venv $v: $($p -c 'import sys;print(sys.version.split()[0])' 2>&1 | tail -1)"
      $p -c "import torch;print('   torch', torch.__version__)" 2>&1 | tail -1
      $p -c "import mujoco;print('   mujoco', mujoco.__version__)" 2>&1 | tail -1
      $p -c "import mujoco_warp;print('   mujoco_warp present')" 2>&1 | tail -1
    else
      echo "venv $v: ABSENT"
    fi
  done
} 2>&1 | tee "$OUT/provenance_${TAG}.txt"

# --------------------------------------------------------------------------------------------------
echo "=== 0/4 which of this workstream's families does each interpreter carry, and why not ==="
# Written to a file AND echoed: this inventory is the job's most reusable output. It is what tells
# the coordinator which of the ten unreached families the cluster can certify at all, and which are
# blocked on an install that has to happen on the LOGIN node (compute nodes may have no internet).
cat > "$OUT/inventory_${TAG}.py" <<'PYEOF'
import json, os, sys
fams = set(sys.argv[1].split())
import stable_envs as snv
mine, errs = {}, {}
for r in snv.registry_table():
    base = r["module"].split(".")[-1]
    if base in fams:
        mine.setdefault(base, []).append(r["name"])
for fam, e in snv.import_errors().items():
    if fam.split(".")[-1] in fams:
        errs[fam] = str(e).replace("\n", " ")[:300]
rec = {"python": sys.executable, "registered": {k: sorted(v) for k, v in sorted(mine.items())},
       "cells": sorted(c for v in mine.values() for c in v), "import_errors": errs}
print("@@INV@@" + json.dumps(rec))
PYEOF
: > "$OUT/inventory_${TAG}.jsonl"
for v in $VENVS; do
    p=$SCR/venvs/$v/bin/python
    [ -x "$p" ] || { echo "  $v: ABSENT"; continue; }
    line=$(cd "$WS" && PYTHONPATH="$WS" $p "$OUT/inventory_${TAG}.py" "$FAMILIES" 2>/dev/null \
           | grep '^@@INV@@' | tail -1)
    if [ -z "$line" ]; then echo "  $v: import stable_envs FAILED"; continue; fi
    echo "${line#@@INV@@}" >> "$OUT/inventory_${TAG}.jsonl"
    echo "  $v:"
    echo "${line#@@INV@@}" | $p -c "
import json,sys
r=json.load(sys.stdin)
print('    cells:', len(r['cells']), '->', ', '.join(r['cells'][:14]) + (' ...' if len(r['cells'])>14 else ''))
for f,e in sorted(r['import_errors'].items()): print(f'    NO {f}: {e[:150]}')
"
done
echo "(inventory: $OUT/inventory_${TAG}.jsonl)"

# --------------------------------------------------------------------------------------------------
echo "=== 1/4 the two-tree oracle, cpu AND cuda, over the DISCOVERED cells in each venv ==="
# Same script as the CPU arm: byte-identical frames vs the detached 21cb306 worktree, plus
# describe / style / taxonomy / fidelity / registry / state_names / round-trip / sample_states, and
# the Step-9 functional checks (pure transition, StepResult keys, close() refusal). Both devices,
# because a family whose only backend is CUDA has no cpu row to compare and must not be recorded as
# if it did.
ORACLE_FAIL=0
for v in $VENVS; do
    p=$SCR/venvs/$v/bin/python
    [ -x "$p" ] || continue
    cells=$(grep -h . "$OUT/inventory_${TAG}.jsonl" | $p -c "
import json,sys
for ln in sys.stdin:
    r=json.loads(ln)
    if r['python'].startswith('$SCR/venvs/$v/'): print(','.join(r['cells'])); break
")
    if [ -z "$cells" ]; then echo "  $v: no cells from this workstream -- SKIPPED (not a pass)"; continue; fi
    for dev in cpu cuda; do
        echo "  --- $v / $dev : $(echo "$cells" | tr ',' '\n' | wc -l | tr -d ' ') cells"
        PYTHONPATH="$WS" $p scratch/step9/envs-mujoco/oracle.py \
            --python "$p" --device "$dev" --cells "$cells" --n-states 16 \
            --head-sha "$(git -C "$WS" rev-parse HEAD)" \
            --name "oracle_${TAG}_${v}_${dev}.json" 2>&1 | tail -40
        rc=${PIPESTATUS[0]}
        [ "$rc" -eq 0 ] || ORACLE_FAIL=1
        echo "  --- $v / $dev rc=$rc"
    done
done

# --------------------------------------------------------------------------------------------------
echo "=== 2/4 paired ALTERNATING before/after timings, same card, this job ==="
# Both legs run here, on this card, interleaved, so the ratio is a property of the patch and not of
# the node. Two hot paths: render (every family) and step (the seven steppable cells, where
# _advance -> _advance_into and the StepResult landed). The --arms head,head run gives the
# same-tree floor beside them rather than leaving it assumed.
PY_BENCH=$SCR/venvs/se-step6/bin/python
if [ -x "$PY_BENCH" ]; then
    for s in render step; do
        PYTHONPATH="$WS" $PY_BENCH scratch/step9/envs-mujoco/bench.py \
            --python "$PY_BENCH" --device cuda --set "$s" --rounds 5 --reps 5 \
            --tag "${TAG}_$s" 2>&1 | tail -30
    done
    PYTHONPATH="$WS" $PY_BENCH scratch/step9/envs-mujoco/bench.py \
        --python "$PY_BENCH" --device cuda --set render --rounds 3 --reps 5 \
        --arms head,head --tag "${TAG}_render_selftime" 2>&1 | tail -20
else
    echo "SKIPPED: $PY_BENCH absent"
fi

# --------------------------------------------------------------------------------------------------
echo "=== 3/4 tests/unit/test_envs_mujoco.py in every venv -- where the SKIPS become results ==="
# This is the stage that closes the frame_shape gap. On the Mac 72 pass and 57 SKIP, and the skips
# are exactly the ten families with no local package; here each venv turns some of them into
# results. The WHOLE output goes to a file and only the summary is echoed: MuJoCo's EGL teardown
# prints dozens of "Exception ignored in __del__" lines AFTER pytest's verdict and `tail` would hide
# the summary (Workstream H's job 6464319 produced no test counts for exactly that reason).
for v in $VENVS; do
    p=$SCR/venvs/$v/bin/python
    [ -x "$p" ] || continue
    log="$OUT/unit_envs_mujoco_${TAG}_${v}.log"
    PYTHONPATH="$WS" $p -m pytest -q -p no:cacheprovider \
        tests/unit/test_envs_mujoco.py tests/unit/test_foundation_api.py > "$log" 2>&1
    rc=$?
    echo "  $v: rc=$rc  $(grep -E '^[0-9]+ (passed|failed)|passed|failed|error' "$log" \
          | grep -v 'Exception ignored' | tail -1)"
    [ "$rc" -eq 5 ] && echo "  $v: WARNING exit 5 = NO TESTS COLLECTED, this is not a pass"
done

# --------------------------------------------------------------------------------------------------
echo "=== 4/4 the GPU suite selection that can see this workstream's code ==="
SEL="Mujoco or PointMaze or Highway or Merge or Roundabout or Racetrack or Parking or Intersection or Crafter or MiniHack or MiniGrid or BabyAI or Sokoban or Overcooked or Griddly or PGX or Procgen or CarRacing or LunarLander or Jumanji or Metaworld or RoboCasa or Humanoid"
for v in se-vis311 se-step6; do
    p=$SCR/venvs/$v/bin/python
    [ -x "$p" ] || continue
    log="$OUT/gpu_tests_${TAG}_${v}.log"
    PYTHONPATH="$WS" $p -m pytest -q -p no:cacheprovider \
        tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py tests/gpu/test_fast_paths.py \
        -k "$SEL" > "$log" 2>&1
    rc=$?
    echo "  $v: rc=$rc  $(grep -E 'passed|failed|error' "$log" | grep -v 'Exception ignored' | tail -1)"
    [ "$rc" -eq 5 ] && echo "  $v: WARNING exit 5 = NO TESTS COLLECTED, this is not a pass"
done
# Expected pre-existing failures, NOT this branch's: missing optional packages produce collection
# errors (Step-6 findings recorded 966 of them across the Oscar venvs), and the OGBench "NODE FAULT"
# nondeterminism is a known class on gpu2108/2110/2113/2116. Neither is in this workstream's files.

echo "ORACLE_FAIL=$ORACLE_FAIL"
echo "artifacts in \$SCR/se9/envs-mujoco/examples/results/step9/envs-mujoco:"
echo "  inventory_${TAG}.jsonl  oracle_${TAG}_<venv>_<dev>.json  timings_${TAG}_{render,step}.json"
echo "  timings_${TAG}_render_selftime.json  unit_envs_mujoco_${TAG}_<venv>.log"
echo "  gpu_tests_${TAG}_<venv>.log  provenance_${TAG}.txt"
echo "Scrub before committing: sed -i 's#/oscar/scratch/sboughan#\$SCR#g' (tests/unit/test_repo_contracts.py fails on absolute cluster paths)"
echo "ALLDONE $(date -Is)"
