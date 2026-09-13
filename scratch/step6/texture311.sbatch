#!/bin/bash
#SBATCH --job-name=se-s6-tex311
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The 3.11 arm of the texture question. se-s3b-fetch311 pins mujoco 3.11 but ships no dm_control,
# so the first probe compared 3.12 against itself. dm_control 1.0.44 is the release that pairs with
# mujoco 3.11 -- stable_envs' own version-skew error names the compatible pairs.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.uv-cache PIP_CACHE_DIR=$SCR/.pip-cache XDG_CACHE_HOME=$SCR/.cache
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONPATH=$SCR/stable-envs
VENV=$SCR/venvs/se-tex311
set -x
rm -rf "$VENV"
uv venv --python 3.12 "$VENV" || exit 1
uv pip install --python "$VENV/bin/python" \
   --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple \
   --index-strategy unsafe-best-match \
   "torch==2.14.0+cu126" "mujoco==3.11.0" "dm_control==1.0.44" numpy imageio || exit 1
cd "$SCR/stable-envs"
"$VENV/bin/python" scratch/step6/texture_probe.py --out "$SCR/step6/texture"
echo "TEX311 rc=$?"
ls -la "$SCR/step6/texture"
