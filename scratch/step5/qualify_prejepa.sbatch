#!/bin/bash
#SBATCH --job-name=se-s5-qual2
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Qualify the trained instrument, in the order the brief sets out.
#
# (1) OFFICIAL RENDERER ONLY, 50 episodes. Section 5: reproduce the baseline before stable-envs
#     enters the experiment at all. The number to land near is the 79% stable_worldmodel records
#     for DINO-WM under this same 50-step budget; the floor to clear is the 8% a random policy
#     scored on the same episodes.
#
# (2) A small PAIRED SMOKE, three arms at n=10, written to a SEPARATE directory. Section 21 wants
#     the renderer switch, the pairing and the blank path exercised before 300 episodes are spent
#     on them, and section 12 says a smoke test is not to be interpreted statistically -- keeping
#     it out of the gate's directory is how that stops being a promise and starts being a fact.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp
export STABLEWM_HOME=$SCR/.stable_worldmodel
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export PYTHONPATH=$SCR/stable-envs
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
PY=$SCR/venvs/se-step5/bin/python
POLICY=${POLICY:-reacher/prejepa/weights_epoch_3.pt}
OUT=$SCR/step5/out/reacher
set -x
ls -la "$STABLEWM_HOME/checkpoints/reacher/prejepa" || exit 1
cd "$SCR/stable-envs"

$PY examples/downstream_step5/reacher/eval_reacher.py --arm official --seed 42 --n 50 \
    --policy "$POLICY" --swm "$SCR/swm-repo" --out "$OUT/runs_prejepa_qual"
echo "QUAL official rc=$?"

for ARM in official stable_envs blank; do
  $PY examples/downstream_step5/reacher/eval_reacher.py --arm $ARM --seed 99 --n 10 \
      --policy "$POLICY" --swm "$SCR/swm-repo" --target-mode offscreen --out "$OUT/smoke"
  echo "SMOKE $ARM rc=$?"
done
ls -la "$OUT/runs_prejepa_qual" "$OUT/smoke"
