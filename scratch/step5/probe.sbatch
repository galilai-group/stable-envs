#!/bin/bash
#SBATCH --job-name=se-s5-probe
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Verify the Step 5 interpreter on a GPU, then ask the only question that decides the Reacher
# harness: do stable_worldmodel's reacher renderer and ours draw the same picture, and does the
# reference draw its target at all.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.uv-cache PIP_CACHE_DIR=$SCR/.pip-cache
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp
export STABLEWM_HOME=$SCR/.stable_worldmodel
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export PYTHONPATH=$SCR/stable-envs
PY=$SCR/venvs/se-step5/bin/python
set -x

cd "$SCR/step5"
$PY - <<'PY'
import torch, numpy, sklearn, hydra, h5py, sys
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
import stable_worldmodel as swm; print("swm", swm.__file__)
import stable_pretraining; print("spt ok")
import mujoco, dm_control; print("mujoco", mujoco.__version__)
import stable_envs as se, stable_envs.envs as E
print("stable_envs families unavailable:", sorted(E.import_errors()))
e = se.make("DMCReacherHard", batch=1, size=224, device="cuda")
print("DMCReacherHard state_dim", e.state_dim, e.state_names)
PY
echo "VERIFY rc=$?"

cd "$SCR/stable-envs"
$PY examples/downstream_step5/reacher/parity_probe.py --n 32 --size 224 \
    --out "$SCR/step5/out/reacher"
echo "PROBE rc=$?"
