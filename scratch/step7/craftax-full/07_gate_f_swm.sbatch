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

echo "== pins BEFORE =="
"$VENV/bin/pip" freeze 2>/dev/null | grep -iE '^(torch|torchvision|jax|jaxlib|craftax|numpy|gymnasium)==' | tee $R/pins_before.txt
"$VENV/bin/pip" freeze 2>/dev/null \
  | grep -iE '^(torch|jax|jaxlib|craftax|numpy|gymnasium)==' > $SCR/se7/craftax-full-constraints.txt
cat $SCR/se7/craftax-full-constraints.txt

echo "== swm checkout =="
cat $SWM/.swm_commit 2>/dev/null || echo "(commit recorded by the rsync side)"

echo "== install (constrained) =="
"$VENV/bin/pip" install --no-build-isolation -e "$SWM" \
  -c $SCR/se7/craftax-full-constraints.txt 2>&1 | tail -25

echo "== pins AFTER =="
"$VENV/bin/pip" freeze 2>/dev/null | grep -iE '^(torch|torchvision|jax|jaxlib|craftax|numpy|gymnasium)==' | tee $R/pins_after.txt
if ! diff -q <(grep -ivE '^torchvision' $R/pins_before.txt) <(grep -ivE '^torchvision' $R/pins_after.txt); then
  echo "PIN MOVED -- the venv every parity number was taken against has changed"; exit 1
fi
echo "pins unchanged (torchvision excluded: it is new, not moved)"

echo "== Gate F for swm/CraftaxPixels-v1 =="
"$VENV/bin/python" $P/gate_f_swm.py --steps 120 --out $R

echo "== parity re-check: the venv still gives the same digests =="
"$VENV/bin/python" $P/certify.py --device cpu --tag "swmvenv_$(hostname)" --skip-gate-a \
  --corpus examples/results/step7/craftax-full/corpus/states.npz
