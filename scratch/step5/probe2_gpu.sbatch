#!/bin/bash
#SBATCH --job-name=se-s5-probe
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Everything that must be true before a downstream number is worth measuring:
#   1. the interpreter works on a GPU
#   2. the corpus has the columns the swap needs, and its frames show whether the target is drawn
#   3. our renderer and stable_worldmodel's draw the same reacher on identical states
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp
export STABLEWM_HOME=$SCR/.stable_worldmodel
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export PYTHONPATH=$SCR/stable-envs
PY=$SCR/venvs/se-step5/bin/python
OUT=$SCR/step5/out/reacher
set -x

cd "$SCR/step5"
$PY - <<'PY'
import torch, sys
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
x = torch.zeros(8, device="cuda"); print("cuda alloc ok", float(x.sum()))
import stable_envs as se, stable_envs.envs as E
print("stable_envs families unavailable:", sorted(E.import_errors()))
e = se.make("DMCReacherHard", batch=1, size=224, device="cuda")
print("DMCReacherHard state_dim", e.state_dim, e.state_names)
PY
echo "VERIFY rc=$?"

$PY inspect_dataset.py --dataset dmc/reacher_random.h5 --out "$OUT"
echo "INSPECT rc=$?"

cd "$SCR/stable-envs"
$PY examples/downstream_step5/reacher/parity_probe.py --n 32 --size 224 --out "$OUT"
echo "PROBE rc=$?"
