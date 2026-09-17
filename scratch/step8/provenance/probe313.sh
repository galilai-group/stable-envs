#!/bin/bash
# The mujoco / mujoco-warp 3.13 bump evaluation, run on a COMPUTE node.
#
# Two arms, one job, one card: the reference interpreter the OGB cells were measured in
# ($SCR/venvs/se7-ogb, mujoco 3.12.0 + mujoco_warp 3.12.0) and the candidate
# ($SCR/venvs/se8-mjwarp313, the SAME pin set with only the MuJoCo pair bumped to 3.13.0).
# A before on one card and an after on another is not a before/after, so both run here.
#
# Three questions, in the order the bump criteria ask them:
#   1. does 3.13 fix the OGBScene texcoord/bake defect (W5's texture_gaps: mesh with texcoordadr=-1)?
#   2. do the frozen digests of a representative mjwarp subset stay byte-identical?
#   3. does anything unrelated move?
set -uo pipefail
: "${SCR:?set SCR to the scratch root before running, e.g. SCR=$SCR bash "$0"}"
export UV_CACHE_DIR=$SCR/cache/uv PIP_CACHE_DIR=$SCR/cache/pip XDG_CACHE_HOME=$SCR/cache/xdg
export TORCH_HOME=$SCR/cache/torch WARP_CACHE_PATH=$SCR/cache/warp HF_HOME=$SCR/cache/hf
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=$SCR/se8/reference-provenance
OUT=$CODE/examples/results/step8/provenance/mjwarp313
mkdir -p "$OUT"
cd "$CODE"
export PYTHONPATH=$CODE STABLE_ENVS_GIT_COMMIT="${STABLE_ENVS_GIT_COMMIT:-unknown}"

echo "### node $(hostname) job ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

for ARM in se7-ogb se8-mjwarp313; do
  PY=$SCR/venvs/$ARM/bin/python
  echo "================================================== ARM $ARM"
  $PY -c "import mujoco, mujoco_warp, warp; import importlib.metadata as m; print('mujoco', m.version('mujoco'), 'mujoco_warp', m.version('mujoco-warp'), 'warp', m.version('warp-lang'))"
  # Q1: the texture gap OGBScene's two button geoms trigger, straight off the model.
  $PY scratch/step8/provenance/texgap_probe.py --out "$OUT/texgap_$ARM.json" 2>&1 | tail -20
  # Q2: frozen-corpus replay, OGB cells on their Step-7 corpus.
  $PY examples/fidelity_step4.py --configs OGBCube,OGBScene --device cuda \
      --states-dir examples/results/step7/ogb-manipulation/states \
      --out "$OUT/replay_ogb_$ARM.json" 2>&1 | tail -8
  # Q2: one DM Control mjwarp cell and one Fetch scene, on the frozen Step-4 corpus.
  $PY examples/fidelity_step4.py --configs DMCCheetahRun --device cuda \
      --out "$OUT/replay_dmc_$ARM.json" 2>&1 | tail -8
  $PY examples/fidelity_step4.py --configs FetchReach --device cuda \
      --out "$OUT/replay_fetch_$ARM.json" 2>&1 | tail -8
done
echo "### done"; ls -l "$OUT"
