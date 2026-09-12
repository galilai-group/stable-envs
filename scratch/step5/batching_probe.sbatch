#!/bin/bash
#SBATCH --job-name=se-s5-batch
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Does CEM's batch_size change the ANSWER, or only the wall clock?
#
# solver/cem.yaml ships batch_size=1, so `for start_idx in range(0, total_envs, batch_size)` plans
# for fifty environments one after another. LeWM did not care -- its predictor is a handful of
# tokens. DINO-WM plans over 768 patch tokens through six transformer layers, so serialising the
# envs turns a minutes-long evaluation into an hours-long one and leaves the GPU mostly idle.
#
# Raising it is NOT a planner tune: num_samples, topk, n_steps and horizon are untouched, and those
# are the search. But the solver draws from one shared generator, so re-chunking changes which
# samples each env receives and therefore the outcome. That makes equivalence a measurement.
#
# batch_size 50 and 10 were tried first and both ran out of memory on a 24 GB 3090: 300
# candidate sequences x 768 DINOv2 patch tokens through six layers does not fit. So the
# committed default of 1 is a memory constraint, not an oversight, and the question becomes how
# far it can be raised rather than whether it should be.
#
# Both runs are the OFFICIAL renderer arm, which is where section 7 allows this to be settled --
# before stable-envs enters the comparison. If the two success rates are indistinguishable, the
# campaign runs batched and the change is recorded. If they are not, the campaign runs as committed
# and takes as long as it takes.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp STABLEWM_HOME=$SCR/.stable_worldmodel
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export PYTHONPATH=$SCR/stable-envs WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
PY=$SCR/venvs/se-step5/bin/python
POLICY=${POLICY:-reacher/prejepa/weights_epoch_3.pt}
OUT=$SCR/step5/out/reacher/batching
set -x
cd "$SCR/stable-envs"
for BS in 5 2; do
  $PY examples/downstream_step5/reacher/eval_reacher.py --arm official --seed 42 --n 50 \
      --policy "$POLICY" --swm "$SCR/swm-repo" --out "$OUT/bs$BS" \
      --extra "solver.batch_size=$BS"
  echo "BATCH $BS rc=$?"
done
ls -la "$OUT"/*/ 2>/dev/null
