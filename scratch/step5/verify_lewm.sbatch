#!/bin/bash
#SBATCH --job-name=se-s5-lewm
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The verification the remap script could not do for itself.
#
# remap_lewm.py proved the key mapping is a bijection with equal shapes and that the result loads
# strictly. None of that can tell you whether q_proj got the query weights or the key weights --
# both are (192, 192) and both load without complaint. What can tell you is the task: this
# checkpoint has a recorded 86% on Reacher under this exact 50-step budget, and a random policy
# scored 8% on the same episodes. A scrambled attention lands near the floor. It does not land
# on 86%.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp
export STABLEWM_HOME=$SCR/.stable_worldmodel
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export PYTHONPATH=$SCR/stable-envs
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
PY=$SCR/venvs/se-step5/bin/python
POLICY=${POLICY:-reacher/lewm_tf5/weights.pt}
OUT=$SCR/step5/out/reacher/runs_lewm
set -x
cd "$SCR/stable-envs"
for ARM in official stable_envs blank; do
  $PY examples/downstream_step5/reacher/eval_reacher.py --arm $ARM --seed 42 --n 50 \
      --policy "$POLICY" --swm "$SCR/swm-repo" --target-mode offscreen --out "$OUT"
  echo "LEWM $ARM rc=$?"
done
ls -la "$OUT"
