#!/bin/bash
#SBATCH --job-name=se-step4-dino
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail

# The representation tripwire, run with the repo's OWN script and its own STATE_SEED. Deliberately not
# driven from the Step 4 corpus: the committed dino_agreement.json that Step 2 froze its bounds
# against was measured on that seed, and re-seeding the states here would make this run incomparable
# with the distribution those bounds came from.
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg
export PYTHONPATH=$SCR/stable-envs
VENV=${VENV:-$SCR/venvs/se-step3}
TAG=${TAG:?set TAG=main|fetch|swm}

# Which shards this interpreter covers. A case statement rather than an --export variable because
# sbatch splits --export on commas, so a list passed that way arrives as one value plus bogus exports.
case "$TAG" in
  main)  SHARDS="dmc ogb raster" ;;
  fetch) SHARDS="fetch" ;;
  swm)   SHARDS="swm" ;;
  *)     echo "unknown TAG=$TAG"; exit 2 ;;
esac

cd $SCR/stable-envs || exit 1
# The env list comes out of the Step 4 universe, not out of this file: a hand-kept list here would
# drift from the configurations that were actually validated and the tripwire would quietly cover a
# different set than the report says it does.
ENVS=$("$VENV/bin/python" - "$SHARDS" <<'PY'
import sys
sys.path.insert(0, "scratch/step4")
import universe
want = set(sys.argv[1].split())
print(",".join(c["config_id"] for c in universe.configurations() if universe.shard_of(c) in want))
PY
)
echo "node=$(hostname) job=$SLURM_JOB_ID tag=$TAG shards='$SHARDS'"
echo "envs: $ENVS"
"$VENV/bin/python" examples/dino_agreement.py --n 32 --envs "$ENVS" --out "$SCR/step4-out/dino/$TAG"
echo "DINO $TAG DONE rc=$?"
