#!/bin/bash
#SBATCH --job-name=se-s5-camp
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The paired gate: the same episodes through three renderers, six seeds of fifty.
#
# Arms run back to back inside ONE allocation per seed, so nothing about the episode selection can
# drift between them -- it is a pure function of cfg.seed and the corpus, and running them in the
# same process on the same node removes the last excuse for a difference. The control arm is not
# optional: success here is decided by ReacherQPosMatchTask from qpos and never reads a pixel, so
# "the score did not move" is evidence only if the score can move.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp
export STABLEWM_HOME=$SCR/.stable_worldmodel
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export PYTHONPATH=$SCR/stable-envs
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
PY=$SCR/venvs/se-step5/bin/python
POLICY=${POLICY:?set POLICY to the frozen checkpoint}
SEEDS=${SEEDS:-"42 43 44 45 46 47"}
N=${N:-50}
TM=${TM:-offscreen}
OUT=${OUT:-$SCR/step5/out/reacher/runs}
set -x
cd "$SCR/stable-envs"
# Five arms per seed. The first three are the gate the brief specifies. The last two are the
# separation the Reacher diagnosis showed is needed and that PushT never needed:
#
#   stable_envs_live   only the LIVE observation is ours; the stored frames stay as recorded.
#                      This is the renderer-substitution question with nothing else moving.
#   official_regen     only the STORED frames are regenerated, by MUJOCO. Our renderer is not in
#                      this arm at all, so whatever it costs is the price of regenerating against
#                      a corpus whose environment has drifted -- measured, not argued.
#
# Without those two, an all-surface result would be reported as a renderer verdict when the corpus
# probe already shows every re-render of a stored state sits ~18/255 from the stored frame,
# MuJoCo's included.
for SEED in $SEEDS; do
  for SPEC in "official:all" "stable_envs:all" "blank:all" "stable_envs:live" "official_regen:dataset"; do
    ARM="${SPEC%%:*}"; SFC="${SPEC##*:}"
    tag=""; [ "$SFC" != "all" ] && tag="_$SFC"
    f="$OUT/seed${SEED}_n${N}_${ARM}${tag}.json"
    if [ -s "$f" ]; then echo "SKIP $f"; continue; fi
    $PY examples/downstream_step5/reacher/eval_reacher.py --arm $ARM --seed $SEED --n $N \
        --policy "$POLICY" --swm "$SCR/swm-repo" --target-mode $TM --surfaces $SFC --out "$OUT"
    echo "RUN seed=$SEED arm=$ARM surfaces=$SFC rc=$?"
  done
done
ls -la "$OUT"
