#!/bin/bash
#SBATCH --job-name=se-s5-tf32
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The last lever that needs no change to anybody's model code.
#
# bf16 is unavailable: PreJEPA.encode hard-casts its input with .float(), so casting the backbone
# to bf16 raises "Input type (float) and bias type (c10::BFloat16) should be the same" -- their
# eval script's bf16 flag cannot work with this model.
#
# TF32 can. PyTorch leaves matmul TF32 OFF by default, so a 768-token, 16-head planner forward runs
# in full fp32 on a card whose tensor cores are idle. Enabling it is a torch backend flag set from
# OUR wrapper; the model, the solver, the search parameters and the config are untouched.
#
# It still changes the arithmetic, so this is a measurement: same seed, same episodes, official
# renderer, TF32 on. Compare success and wall clock against the fp32 run. Identical success means
# the campaign can afford six seeds; a moved success rate means fp32 and a longer schedule.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp STABLEWM_HOME=$SCR/.stable_worldmodel
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export PYTHONPATH=$SCR/stable-envs WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
PY=$SCR/venvs/se-step5/bin/python
POLICY=${POLICY:-reacher/prejepa/weights_epoch_3.pt}
OUT=$SCR/step5/out/reacher/precision
set -x
cd "$SCR/stable-envs"
$PY examples/downstream_step5/reacher/eval_reacher.py --arm official --seed 42 --n 50 \
    --policy "$POLICY" --swm "$SCR/swm-repo" --out "$OUT/tf32_bs1" --tf32
echo "TF32 bs1 rc=$?"
