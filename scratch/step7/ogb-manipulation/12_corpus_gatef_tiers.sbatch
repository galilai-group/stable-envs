#!/bin/bash
#SBATCH --job-name=se7-ogb-manipulation-final
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The remaining cell work, in one GPU job: freeze the Step-4-schema corpora, run Gate F's arm D
# against swm, and measure the tiers with SAME-NODE comparators.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache PIP_CACHE_DIR=$SCR/.cache/pip
export TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface WARP_CACHE_PATH=$SCR/.cache/warp
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se7/ogb-manipulation
VENV=$SCR/venvs/se7-ogb
OUT=$WS/examples/results/step7/ogb-manipulation
export PYTHONPATH=$WS:$SCR/se7/swm-src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1      # the decode bar is a ONE-CORE number
set -x
cd "$WS"
nvidia-smi --query-gpu=name --format=csv || true
eval "$(PYTHONPATH=$PYTHONPATH "$VENV/bin/python" \
  scratch/step7/ogb-manipulation/gl_probe.py --require egl 2>&1 | tee /dev/stderr | tail -1)"
[ "$MUJOCO_GL" = "__NONE__" ] && { echo "FATAL: EGL unusable on $(hostname)"; exit 9; }

# The cells must be REGISTERED here -- no SE7_OGB_UNMEASURED. If _MEASURED is not filled this
# job fails rather than quietly measuring a placeholder build.
"$VENV/bin/python" -c "
import stable_envs as se
assert 'OGBCube' in se.REGISTRY and 'OGBScene' in se.REGISTRY, 'cells not registered'
print('registered:', se.REGISTRY['OGBCube'].visual_fidelity['mean_abs'],
      se.REGISTRY['OGBScene'].visual_fidelity['mean_abs'])" || exit 3

mkdir -p "$OUT/states" "$OUT/m3" "$OUT/m4"
"$VENV/bin/python" scratch/step7/ogb-manipulation/probe_corpus_gatef.py \
  --out "$OUT/m3" --states-dir "$OUT/states" --cells OGBCube OGBScene \
  --n 64 --pool 256 --steps 24 || echo "CORPUS/GATEF rc=$?"
"$VENV/bin/python" scratch/step7/ogb-manipulation/probe_m4_tiers.py \
  --out "$OUT/m4" --cells OGBCube OGBScene || echo "TIERS rc=$?"

for f in "$OUT/m3/corpus_gatef.json" "$OUT/m4/m4_tiers.json" \
         "$OUT/states/OGBCube.npz" "$OUT/states/OGBScene.npz"; do
  [ -s "$f" ] || { echo "MISSING $f"; exit 4; }
done
echo OK
