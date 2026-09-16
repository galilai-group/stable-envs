#!/bin/bash
#SBATCH --job-name=se7-ogb-manipulation-gate
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
# THE CONSTRAINT IS NOT OPTIONAL HERE. `sinfo -p 3090-gcondo` lists a5000, a5500 AND geforce3090
# nodes, so without it this lands on the wrong card and the gate's "on a 3090" is not satisfied.
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# THE GATE: the three tests/gpu files WHOLE -- no -k -- plus Gate A on both cells.
#
# Run through `stable_envs.testing.supervisor`, the way `scripts/run_gpu_tests.sbatch` does, and not
# through bare pytest: that module exists because this suite has aborted itself out of MuJoCo, EGL or
# the CUDA driver, and a bare pytest reports NOTHING when it does -- no name for the test that died,
# nothing run after it, and a shell exit code for an artifact. A gate whose failure mode is silence
# is not a gate.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp MPLBACKEND=Agg PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se7/ogb-manipulation
VENV=$SCR/venvs/se7-ogb
OUT=$WS/examples/results/step7/ogb-manipulation/m6/gate
export PYTHONPATH=$WS:$SCR/se7/swm-src
mkdir -p "$OUT"
cd "$WS"

WANT=__MD5__
GOT=$(md5sum stable_envs/envs/ogb_manip.py | cut -d' ' -f1)
echo "ogb_manip.py md5 want=$WANT got=$GOT"
[ "$WANT" = "$GOT" ] || { echo "MD5 MISMATCH -- the node is not running HEAD; refusing"; exit 3; }

set -x
nvidia-smi --query-gpu=name --format=csv || true

"$VENV/bin/python" scratch/step7/ogb-manipulation/probe_gate_a.py --out "$OUT" \
  --cells OGBCube OGBScene > "$OUT/gate_a.txt" 2>&1; echo "GATE A rc=$?"
cat "$OUT/gate_a.txt"

RUN="$OUT/run"
mkdir -p "$RUN"
"$VENV/bin/python" -u -m stable_envs.testing.supervisor \
  --rundir "$RUN" --json "$RUN/result.json" \
  -- tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py tests/gpu/test_fast_paths.py \
  -v -rs --durations=25 > "$OUT/gputests_full.txt" 2>&1
echo "SUITE rc=$?"
grep -E "^(FAILED|ERROR)|passed|failed|resources:" "$OUT/gputests_full.txt" | tail -40
echo "--- OGB rows only ---"
grep -E "^(FAILED|ERROR).*(OGBCube|OGBScene)" "$OUT/gputests_full.txt" || echo "NO OGB FAILURES"

cp "/oscar/scratch/$USER/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$OUT/slurm_${SLURM_JOB_ID}.log" 2>/dev/null || true
gzip -9f "$OUT/slurm_${SLURM_JOB_ID}.log" 2>/dev/null || true
echo OK
