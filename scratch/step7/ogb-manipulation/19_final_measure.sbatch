#!/bin/bash
#SBATCH --job-name=se7-ogb-manipulation-final-measure
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Verify the three fixes and take the declaration with the GATE'S OWN instrument.
#
# The md5 assert runs HERE rather than on a login node: a quick check is still compute, it belongs
# in the job by the standing rule, and running it on the compute node is strictly better evidence --
# it proves the file this GPU actually imported is the file at HEAD, and the proof is in the log.
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
cd "$WS"

WANT=03398d5130e07bfc7ee8bd73e8570e5a
GOT=$(md5sum stable_envs/envs/ogb_manip.py | cut -d' ' -f1)
echo "ogb_manip.py md5 want=$WANT got=$GOT"
[ "$WANT" = "$GOT" ] || { echo "MD5 MISMATCH -- the node is not running HEAD; refusing"; exit 3; }

set -x
nvidia-smi --query-gpu=name --format=csv || true

# (a) the declaration, taken with `check_visual_fidelity` itself
"$VENV/bin/python" scratch/step7/ogb-manipulation/probe_gate_fidelity.py \
  --out "$OUT/final8" --cells OGBCube OGBScene --n 64 --draws 8 || echo "PROBE rc=$?"
[ -s "$OUT/final8/paste_me.json" ] || { echo "MISSING paste_me.json"; exit 4; }
cat "$OUT/final8/paste_me.json"

# (b) the OGB rows, to see whether the resource-guard and styleless fixes hold. The fidelity row is
#     EXPECTED to fail here: its ceiling is pasted from (a), which has only just been measured.
"$VENV/bin/python" -m pytest tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py \
  tests/gpu/test_fast_paths.py -k "OGBCube or OGBScene" -q --no-header -rf --tb=long \
  > "$OUT/gputests_ogb_final.txt" 2>&1 || true
grep -E "^(FAILED|ERROR)|passed|failed|resources:" "$OUT/gputests_ogb_final.txt" || true

cp "/oscar/scratch/$USER/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$OUT/slurm_${SLURM_JOB_ID}.log" 2>/dev/null || true
gzip -9f "$OUT/slurm_${SLURM_JOB_ID}.log" 2>/dev/null || true
echo OK
