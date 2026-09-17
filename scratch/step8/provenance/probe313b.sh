#!/bin/bash
# Round 2 of the 3.13 evaluation: frozen-corpus DIGEST replay of a representative mjwarp subset,
# both arms in one job on one card. Round 1 used examples/fidelity_step4.py, whose universe is the
# 49 Step-4 configurations -- the two OGB cells are Step-7 cells and are not in it, which is why it
# measured zero of them. digest_replay.py reads the committed corpus for whichever cell it is given.
set -uo pipefail
: "${SCR:?set SCR to the scratch root before running, e.g. SCR=$SCR bash "$0"}"
export UV_CACHE_DIR=$SCR/cache/uv PIP_CACHE_DIR=$SCR/cache/pip XDG_CACHE_HOME=$SCR/cache/xdg
export TORCH_HOME=$SCR/cache/torch WARP_CACHE_PATH=$SCR/cache/warp HF_HOME=$SCR/cache/hf
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=$SCR/se8/reference-provenance
OUT=$CODE/examples/results/step8/provenance/mjwarp313
mkdir -p "$OUT"; cd "$CODE"; export PYTHONPATH=$CODE
echo "### node $(hostname) job ${SLURM_JOB_ID:-none}"

# se-s3b-fetch311 is the ONLY interpreter where the Fetch stack builds at all: gymnasium-robotics'
# set_joint_qpos assertion fires under mujoco >= 3.12 (docs/research/swm_matrix.md), so the Fetch
# row of this table is a statement about the reference package, not about our renderer.
for ARM in se-s3b-fetch311 se7-ogb se8-mjwarp313; do
  PY=$SCR/venvs/$ARM/bin/python
  echo "================================================== ARM $ARM"
  $PY -c "import importlib.metadata as m; print('mujoco', m.version('mujoco'), 'mujoco_warp', m.version('mujoco-warp'))" 2>&1 | tail -2
  $PY scratch/step8/provenance/texgap_probe.py --out "$OUT/texgap_$ARM.json" 2>&1 | tail -25
  $PY scratch/step8/provenance/digest_replay.py --device cuda \
      --cells OGBCube,OGBScene,DMCCheetahRun,FetchReach \
      --out "$OUT/digest_$ARM.json" 2>&1 | tail -12
done
echo "### done"; ls -l "$OUT"
