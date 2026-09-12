#!/bin/bash
#SBATCH --job-name=se-s5-regen
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The control that decides whether the Reacher dataset-surface drop is about our renderer.
#
# Established so far, on the LeWM instrument at seed 42: official 52%, live-surface swap 52%
# (identical episode for episode), all-surface swap 22%, dataset-surface swap 18%. And the corpus
# probe shows every re-render of a stored state -- MuJoCo's included -- sits ~18/255 from the
# stored frame, because the corpus was recorded with a floor texture this environment no longer
# draws.
#
# So: regenerate the stored frames with MUJOCO and change nothing else. If this also collapses,
# the cost is regeneration against a drifted corpus and not the renderer that did it.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp STABLEWM_HOME=$SCR/.stable_worldmodel
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export PYTHONPATH=$SCR/stable-envs WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
PY=$SCR/venvs/se-step5/bin/python
POLICY=${POLICY:-reacher/lewm_tf5/weights.pt}
OUT=$SCR/step5/out/reacher/runs_lewm
set -x
cd "$SCR/stable-envs"
$PY examples/downstream_step5/reacher/eval_reacher.py --arm official_regen --seed 42 --n 50 \
    --policy "$POLICY" --swm "$SCR/swm-repo" --surfaces dataset --out "$OUT"
echo "REGEN rc=$?"
