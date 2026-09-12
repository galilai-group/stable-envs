#!/bin/bash
#SBATCH --job-name=se-s5-bf16
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Why an inference-only planner needs 18 GB, and whether it has to.
#
# CEM scores 300 candidate plans per iteration. Each is a 6-layer transformer forward over 768
# tokens (3 history frames x 256 DINOv2 patches), 16 heads. Attention is already fused, but it is
# called with an explicit boolean attn_mask, and in fp32 that drops SDPA to the math backend, which
# materialises 300 x 16 x 768 x 768 x 4 bytes ~ 11 GB of scores for a single layer. No gradients
# are involved: solve() is @torch.inference_mode(), get_action() adds no_grad, and the model is
# requires_grad_(False).
#
# bf16 halves the bytes and makes the memory-efficient kernel eligible, and eval_wm.py already
# implements the flag -- it casts the model and wraps the evaluation in autocast. reacher.yaml just
# never sets it. Run 1: bf16 at the committed batch_size, to compare success and time against the
# fp32 run at the same seed and episodes. Run 2: bf16 with a batch, if the memory now allows it.
#
# Both are the OFFICIAL arm. bf16 changes numerics, so it is a measurement and not an assumption --
# if the success rate moves, the campaign runs fp32 and takes as long as it takes.
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
    --policy "$POLICY" --swm "$SCR/swm-repo" --out "$OUT/bf16_bs1" --extra "+bf16=true"
echo "BF16 bs1 rc=$?"
$PY examples/downstream_step5/reacher/eval_reacher.py --arm official --seed 42 --n 50 \
    --policy "$POLICY" --swm "$SCR/swm-repo" --out "$OUT/bf16_bs4" \
    --extra "+bf16=true" "solver.batch_size=4"
echo "BF16 bs4 rc=$?"
