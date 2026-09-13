#!/bin/bash
#SBATCH --job-name=se-s6-vis311
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Does the MuJoCo 3.11 -> 3.12 texture change explain Step 4's visual-tier failures?
#
# Step 4 recorded that DMCCheetahRun, DMCHopperHop and DMCWalkerWalk DECLARED their visual-fidelity
# contracts under mujoco 3.11.0 + mujoco-warp 3.11.0 and were MEASURED under 3.12.0, and that the
# residual sat in the skybox -- which is a texture (800x4800 gradient, mark="random"). We have now
# shown 3.12 stops painting dm_control's checker texture on the reacher floor: same model, same
# declaration, quadrant spread 40.68 -> 7.49.
#
# If one upstream change explains both, replaying the FROZEN Step 4 corpora under 3.11 should put
# those three back inside their declared numbers. If it does not, the visual drift is a separate
# problem and the correctness workstream needs to keep looking.
#
# The frozen corpora are READ-ONLY (section 5) and the output goes to a Step 6 directory; nothing
# under examples/results/step4/ is touched.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.uv-cache PIP_CACHE_DIR=$SCR/.pip-cache XDG_CACHE_HOME=$SCR/.cache
export TORCH_HOME=$SCR/.torch WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONPATH=$SCR/stable-envs
VENV=$SCR/venvs/se-vis311
set -x
if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.12 "$VENV" || exit 1
  uv pip install --python "$VENV/bin/python" \
     --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple \
     --index-strategy unsafe-best-match \
     "torch==2.14.0+cu126" "mujoco==3.11.0" "dm_control==1.0.44" \
     "mujoco-warp==3.11.0" "warp-lang" numpy scipy imageio || exit 1
fi
cd "$SCR/stable-envs"
"$VENV/bin/python" -c "import mujoco, dm_control; print('mujoco', mujoco.__version__)"
"$VENV/bin/python" examples/fidelity_step4.py --shard dmc \
    --configs DMCCheetahRun DMCHopperHop DMCWalkerWalk DMCHumanoidStand \
    --states-dir "$SCR/stable-envs/examples/results/step4/states" \
    --out "$SCR/step6/visual311/dmc.json"
echo "VIS311 rc=$?"
