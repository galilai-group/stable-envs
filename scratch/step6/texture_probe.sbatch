#!/bin/bash
#SBATCH --job-name=se-s6-tex
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Same script, two interpreters, two MuJoCo versions. Needs a GPU only because dm_control renders
# through EGL here; the work is four frames.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache MUJOCO_GL=egl PYTHONUNBUFFERED=1
export PYTHONPATH=$SCR/stable-envs
set -x
cd "$SCR/stable-envs"
for V in se-step5 se-s3b-fetch311 se-step3; do
  PY=$SCR/venvs/$V/bin/python
  [ -x "$PY" ] || continue
  echo "### $V"
  $PY scratch/step6/texture_probe.py --out "$SCR/step6/texture" || echo "TEX $V rc=$?"
done
echo "TEX done"
ls -la "$SCR/step6/texture" 2>/dev/null
