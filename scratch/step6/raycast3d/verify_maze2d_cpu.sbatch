#!/bin/bash
#SBATCH --job-name=se6-raycast3d-maze2d
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# RAYCAST-2c' (d): the maze2d PointMaze* envs that RAYCAST-2b could not verify.
#
# They call R3.composite and they have no frozen Step-4 oracle, so nothing else would catch a change
# there. They were UNVERIFIED because `se-step6` has no gymnasium_robotics. This builds a SEPARATE
# venv for them and never touches se-step6:
#
#   $SCR/venvs/se-step6  --(python -m venv --system-site-packages)-->  $SCR/venvs/se6-raycast3d-maze2d
#
# so every pin se-step6 carries is inherited read-only and only gymnasium-robotics lands in the new
# tree. --no-deps because every dependency it declares (gymnasium, mujoco, numpy) is already there at
# the pin the rest of Step 6 measured against, and letting pip resolve them would silently move the
# stack this verification is supposed to hold still.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/raycast3d
BASE=$SCR/se6/baseline
VENV=$SCR/venvs/se6-raycast3d-maze2d
OUT=$WS/examples/results/step6/raycast3d
D=$SCR/se6/_dumps/maze2d
set -x
cd "$WS"

# --system-site-packages does NOT do what it looks like it does here: created from a venv's python
# it inherits the BASE interpreter's site-packages, not the parent venv's, so the first attempt got a
# tree with gymnasium_robotics and no numpy (job 6347001). The link that works is a .pth file, and it
# has to be a .pth rather than PYTHONPATH because PYTHONPATH is already carrying the baseline/branch
# switch that makes this a before/after measurement at all.
if [ ! -x "$VENV/bin/python" ]; then
  $SCR/venvs/se-step6/bin/python -m venv "$VENV"
  # --no-deps for every one of them: gymnasium, numpy, mujoco and torch all come from se-step6 at
  # the pins the rest of Step 6 measured against, and letting pip resolve them would move the stack
  # this verification exists to hold still. pettingzoo is gymnasium_robotics' own import-time
  # dependency (its multi-agent module is imported from __init__), found by job 6347125.
  "$VENV/bin/python" -m pip install --no-cache-dir --no-deps gymnasium-robotics pettingzoo
fi
STEP6_SITE=$($SCR/venvs/se-step6/bin/python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
VENV_SITE=$("$VENV/bin/python" -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
echo "$STEP6_SITE" > "$VENV_SITE/se-step6-inherit.pth"
PY=$VENV/bin/python
$PY -c "import gymnasium_robotics, gymnasium, mujoco, torch; \
print('gymnasium_robotics', gymnasium_robotics.__version__, '| gymnasium', gymnasium.__version__, \
'| mujoco', mujoco.__version__, '| torch', torch.__version__)"
# prove se-step6 was not modified
$SCR/venvs/se-step6/bin/python -c "
import importlib.util as u
print('se-step6 sees gymnasium_robotics:', u.find_spec('gymnasium_robotics') is not None)"

rm -rf "$D"; mkdir -p "$D"
M2D="PointMazeLarge PointMazeMedium PointMazeOpen PointMazeUMaze"

echo "######## COVER_COMPACTION on == off, on CPU, for the maze2d envs"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $M2D \
    --sample-states 20260913 --cover-compaction on  --dump-dir "$D/on" \
    --out "$OUT/replay/2b_maze2d_compact_on.json" || echo "ON rc=$?"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $M2D \
    --sample-states 20260913 --cover-compaction off --dump-dir "$D/off" \
    --out "$OUT/replay/2b_maze2d_compact_off.json" || echo "OFF rc=$?"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/off" "$D/on" \
    --out "$OUT/replay/2b_maze2d_compaction_identity.json" || echo "IDENT rc=$?"

echo "######## and the whole branch against the pristine baseline, same states"
PYTHONPATH=$BASE $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $M2D \
    --sample-states 20260913 --dump-dir "$D/base" \
    --out "$OUT/replay/2b_maze2d_baseline.json" || echo "BASE rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $M2D \
    --sample-states 20260913 --dump-dir "$D/branch" \
    --out "$OUT/replay/2b_maze2d_branch.json" || echo "BRANCH rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/base" "$D/branch" \
    --out "$OUT/replay/2b_maze2d_branch_vs_baseline.json" || echo "DIFF rc=$?"

echo "MAZE2D done"
