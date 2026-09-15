#!/bin/bash
#SBATCH --job-name=se7-dmc-manipulator-o1
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# O1: diff the render-model signature of stable_worldmodel's mjcf ROUND-TRIP against suite.load.
#
# Gate A S4.2 requires the certified configuration to be the registered SWM default. The coordinator's
# ruling is that parity is measured against suite.load and the SWM default is reached THROUGH LEAVES
# (visibility.target_ball, target_ball.shape, light.diffuse_scale) -- which is only sound if the two
# compiled models are the SAME SCENE up to those leaves. SWM does not call suite.load; it round-trips
# the XML through dm_control.mjcf, and an mjcf round-trip can rename or reorder elements. _episode_slots
# and _prims key on model ORDER and on mj_id2name, so this has to be measured before certification.
#
# Needs stable_worldmodel, which only se-step5 has. No GPU, no renderer: it compiles two models and
# diffs arrays. HOME is at quota -- every cache goes under $SCR.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=dmc-manipulator
export PYTHONPATH=$SCR/se7/$WS
PY=$SCR/venvs/se-step5/bin/python
set -x
cd "$SCR/se7/$WS"

$PY -c "import mujoco, stable_worldmodel, dm_control; print('mujoco', mujoco.__version__)"
$PY scratch/step7/$WS/probe_swm_signature.py \
    --out "$SCR/se7/$WS/examples/results/step7/$WS/oscar"
rc=$?
set +x
echo "O1 signature probe rc=$rc"
ls -la "$SCR/se7/$WS/examples/results/step7/$WS/oscar" || true
exit $rc
