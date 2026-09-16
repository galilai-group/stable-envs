#!/bin/bash
#SBATCH --job-name=se7-ogb-manipulation-gputests
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The GATE, and the measurement the gate needs.
#
# (a) The OGB cells through tests/gpu in se7-ogb -- the interpreter where they actually construct.
#     The closure sweep found 14 failures here and my own sub-batch probe cannot reproduce the
#     invariance half on the current tree, so the first thing to establish is what still fails NOW,
#     on this tree, with the mechanism attached rather than inferred.
# (b) An EIGHT-draw fidelity re-measurement, because the three-draw spread understated the draw
#     variance: the conformance check draws its own states and got mean_abs 3.287 against a ceiling
#     of 3.2 set from worst-of-three x1.3.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp MPLBACKEND=Agg PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se7/ogb-manipulation
VENV=$SCR/venvs/se7-ogb
OUT=$WS/examples/results/step7/ogb-manipulation/m6
export PYTHONPATH=$WS:$SCR/se7/swm-src
mkdir -p "$OUT"
set -x
cd "$WS"
nvidia-smi --query-gpu=name --format=csv || true

# (a) the gate, OGB rows only, with the mechanism printed for every failure
"$VENV/bin/python" -m pytest tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py \
  tests/gpu/test_fast_paths.py -k "OGBCube or OGBScene" -q --no-header -rf --tb=long \
  > "$OUT/gputests_ogb.txt" 2>&1 || true
tail -40 "$OUT/gputests_ogb.txt"
grep -cE "^(FAILED|ERROR)" "$OUT/gputests_ogb.txt" || true

# (b) eight draws, for a ceiling that covers the variance
SE7_OGB_UNMEASURED=1 "$VENV/bin/python" scratch/step7/ogb-manipulation/probe_remeasure_chain.py \
  --out "$OUT/draw8" --states-dir "$OUT/draw8_states" --cells OGBCube OGBScene \
  --n 64 --draws 8 --pool 256 --steps 24 || echo "CHAIN rc=$?"
[ -s "$OUT/draw8/paste_me.json" ] || { echo "MISSING paste_me.json"; exit 4; }
cat "$OUT/draw8/paste_me.json"
cp "/oscar/scratch/$USER/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$OUT/slurm_${SLURM_JOB_ID}.log" 2>/dev/null || true
gzip -9f "$OUT/slurm_${SLURM_JOB_ID}.log" 2>/dev/null || true
echo OK
