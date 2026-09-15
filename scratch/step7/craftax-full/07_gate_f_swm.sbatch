#!/bin/bash
#SBATCH --job-name=se7-craftax-full-gate-f-swm
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# Install stable_worldmodel into $SCR/venvs/se7-craftax from the rsynced CHECKOUT (not PyPI: W5 found
# 0.0.6 differs from it), then run Gate F for swm/CraftaxPixels-v1.
#
# **The install must not move a pin.** This venv is what every parity number on this branch was taken
# against, and stable_worldmodel depends on torch/torchvision/gymnasium, any of which pip would
# happily upgrade -- taking jax, craftax or torch with it and silently invalidating the digests. So:
# freeze the pins that matter into a CONSTRAINTS file first, install under it, and then re-run the
# parity certify at the end. If a dependency demands a different torch, pip fails loudly here instead
# of succeeding quietly and breaking the branch.
set -euo pipefail
source /oscar/scratch/sboughan/se7/craftax-full/scratch/step7/craftax-full/env_header.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
VENV=$SCR/venvs/se7-craftax
SWM=$SCR/se7/stable-worldmodel
P=scratch/step7/craftax-full
R=examples/results/step7/craftax-full/gate_f
mkdir -p $R
echo "== host $(hostname) arch $(uname -m) =="

# uv, not pip: this venv was built by ``uv venv`` (W1's 01_build_venv.sbatch) and has no pip in it
# at all -- job 6398628 died in one second on ``$VENV/bin/pip freeze`` under ``set -o pipefail``.
UV=/users/sboughan/.local/bin/uv
PY="$VENV/bin/python"
PINS='^(torch|torchvision|jax|jaxlib|craftax|numpy|gymnasium)=='

echo "== pins BEFORE =="
"$UV" pip freeze --python "$PY" | grep -iE "$PINS" | sort | tee $R/pins_before.txt
"$UV" pip freeze --python "$PY" | grep -iE '^(torch|jax|jaxlib|craftax|numpy|gymnasium)==' \
  | sort > $SCR/se7/craftax-full-constraints.txt
echo "-- constraints --"; cat $SCR/se7/craftax-full-constraints.txt

echo "== swm checkout =="
cat $SWM/.swm_commit

echo "== install (constrained) =="
# Build isolation is ON, deliberately. stable_worldmodel builds with the ``uv_build`` backend, which
# is not in this venv, so ``--no-build-isolation`` fails at the build step (job 6398700). Isolation
# fetches the backend into a throwaway environment and leaves the target venv's runtime packages
# alone, which is the property that matters here -- the constraints file is what protects the pins.
"$UV" pip install --python "$PY" -e "$SWM" \
  -c $SCR/se7/craftax-full-constraints.txt 2>&1 | tail -30

# torchvision must match the TORCH BUILD, not just its version. stable_worldmodel pulls torchvision,
# PyPI serves the wheel built against torch 2.14.0+cu130, and this venv carries +cu126 (W1 pinned it
# there because Oscar's 3090 driver is 12.9 and the cu130 wheel dies the moment it touches the card).
# The two link against different C extensions, so importing stable_worldmodel died with
# "operator torchvision::nms does not exist" -- job 6398740, which installed cleanly and still could
# not import the wrapper. Same fix W1 used for torch: take it from the cu126 index, and
# --reinstall-package because "+cu130" and "+cu126" both satisfy "==0.29.0" and uv would otherwise
# audit the wrong wheel as a match and change nothing.
echo "== torchvision, from the cu126 index to match torch =="
"$UV" pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu126 \
  --reinstall-package torchvision -c $SCR/se7/craftax-full-constraints.txt "torchvision==0.29.0" 2>&1 | tail -8
"$PY" -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__)"

# stable_worldmodel's base dependencies are not enough to IMPORT it: wrapper/visual.py imports cv2,
# which lives in the [env] extra. The extra as a whole is not installed, because it pulls
# gymnasium[all] and this venv's gymnasium is a pin every parity number depends on. These are the
# ones the import chain actually reaches, installed under the same constraints file -- so if any of
# them demanded a different numpy or gymnasium the job fails here rather than moving a pin.
echo "== the [env] extras the import chain needs =="
"$UV" pip install --python "$PY" -c $SCR/se7/craftax-full-constraints.txt \
  opencv-python pygame pymunk shapely 2>&1 | tail -12

echo "== pins AFTER =="
"$UV" pip freeze --python "$PY" | grep -iE "$PINS" | sort | tee $R/pins_after.txt
# torchvision is EXCLUDED from the comparison because stable_worldmodel requires it and it was not
# installed before, so it is new rather than moved. Everything the parity numbers depend on must be
# byte-identical.
if ! diff <(grep -ivE '^torchvision' $R/pins_before.txt) <(grep -ivE '^torchvision' $R/pins_after.txt); then
  echo "PIN MOVED -- the venv every parity number on this branch was taken against has changed"
  exit 1
fi
echo "pins unchanged"

echo "== Gate F for swm/CraftaxPixels-v1 =="
"$VENV/bin/python" $P/gate_f_swm.py --steps 120 --out $R

echo "== parity re-check: the venv still gives the same digests =="
"$VENV/bin/python" $P/certify.py --device cpu --tag "swmvenv_$(hostname)" --skip-gate-a \
  --corpus examples/results/step7/craftax-full/corpus/states.npz
