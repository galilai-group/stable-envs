#!/bin/bash
#SBATCH --job-name=se7-ogb-manipulation-remeasure
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The WHOLE re-measurement chain after the cell review's B1-B6 fixes, in one job, on the fixed tree.
#
# The steps are not independent and the last round proved it: thresholds come from the draws, the
# corpus is selected from a pool only `sample_states` can produce, and Gate F and the tiers have to
# be measured on the same build as the fidelity numbers. Split across jobs, that is how a committed
# pool ended up not being the pool the corpus was selected from.
#
# THE TREE IS VERIFIED BEFORE ANYTHING RUNS. The md5 of `ogb_manip.py` on the node must match the
# one the submitter recorded from HEAD; a measurement of a tree that is not the committed one is
# worse than no measurement, because it looks like evidence.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache PIP_CACHE_DIR=$SCR/.cache/pip
export TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface WARP_CACHE_PATH=$SCR/.cache/warp
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1      # the decode bar is a ONE-CORE number
WS=$SCR/se7/ogb-manipulation
VENV=$SCR/venvs/se7-ogb
OUT=$WS/examples/results/step7/ogb-manipulation
export PYTHONPATH=$WS:$SCR/se7/swm-src
set -x
cd "$WS"
nvidia-smi --query-gpu=name --format=csv || true

# ---- the tree is the committed tree ---------------------------------------------------------
GOT=$(md5sum "$WS/stable_envs/envs/ogb_manip.py" | awk '{print $1}')
echo "ogb_manip.py md5 on node: $GOT"
if [ -n "${EXPECT_MD5:-}" ] && [ "$GOT" != "$EXPECT_MD5" ]; then
  echo "FATAL: ogb_manip.py on the node ($GOT) is not the submitted tree ($EXPECT_MD5)"; exit 3
fi

eval "$(PYTHONPATH=$PYTHONPATH "$VENV/bin/python" \
  scratch/step7/ogb-manipulation/gl_probe.py --require egl 2>&1 | tee /dev/stderr | tail -1)"
[ "$MUJOCO_GL" = "__NONE__" ] && { echo "FATAL: EGL unusable on $(hostname)"; exit 9; }

# THIS IS THE MEASURING PASS, so it runs WITH the placeholder flag -- and the first version of this
# script had that exactly backwards. It asserted the cells were already registered, which they cannot
# be until this job has produced the numbers that register them: the module refuses to register
# without measurements, and measuring needs the module. `SE7_OGB_UNMEASURED=1` is the documented
# escape for precisely this process and nothing else.
#
# What is guarded instead is the thing that actually matters: that the placeholder never leaves this
# job. The declaration under the flag carries sentinel -1 values and a `measured_at` that says so, so
# a build that shipped it could not be mistaken for measured; and the artifacts this job writes are
# `paste_me.json` and `remeasure.json`, never a modified `ogb_manip.py`.
export SE7_OGB_UNMEASURED=1
"$VENV/bin/python" -c "
import stable_envs as se
c = se.REGISTRY['OGBCube']
assert c.visual_fidelity['mean_abs'] == -1.0, 'expected the placeholder in the measuring pass'
assert 'UNMEASURED' in c.visual_fidelity['measured_at']
print('measuring pass: placeholder in force, cells constructible')" || exit 3

mkdir -p "$OUT/m3/fidelity" "$OUT/states"
"$VENV/bin/python" scratch/step7/ogb-manipulation/probe_remeasure_chain.py \
  --out "$OUT/m3/fidelity" --states-dir "$OUT/states" \
  --cells OGBCube OGBScene --n 64 --draws 3 --pool 256 --steps 24 || echo "CHAIN rc=$?"

for f in "$OUT/m3/fidelity/remeasure.json" "$OUT/m3/fidelity/paste_me.json" \
         "$OUT/states/OGBCube.npz" "$OUT/states/OGBCube_pool.npz" \
         "$OUT/states/OGBScene.npz" "$OUT/states/OGBScene_pool.npz"; do
  [ -s "$f" ] || { echo "MISSING $f"; exit 4; }
done

# ---- the log-commit step: the job files its OWN log beside the artifacts it produced ---------
# Copied here rather than left in $SCR/logs, because logs rotate and artifacts are committed; and
# gzipped because the Warp teardown noise is most of the bulk.
cp "/oscar/scratch/$USER/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
   "$OUT/m3/fidelity/slurm_${SLURM_JOB_ID}.log" 2>/dev/null || true
gzip -9f "$OUT/m3/fidelity/slurm_${SLURM_JOB_ID}.log" 2>/dev/null || true
ls -l "$OUT/m3/fidelity" "$OUT/states"
echo OK
