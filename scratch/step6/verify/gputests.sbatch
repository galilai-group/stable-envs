#!/bin/bash
#SBATCH --job-name=se6-verify-gputests
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The FULL tests/gpu suite on the integration branch, unfiltered, one file at a time so a per-file
# pass/skip/fail count exists and a crash in one file cannot hide the rest.
#
# ONE interpreter for all eight files, which is the only reason this is one job: $SCR/venvs/se6-verify
# (built by this workstream, installs nothing -- a --without-pip venv whose .pth chains se-step6's
# site-packages, se-step3's and the stable_worldmodel checkout). se-step6 alone has pytest but no
# mujoco_warp, so every MJWarp cell would have skipped as "No module named 'warp'" and the suite
# would have reported green on a family it never constructed.
#
# SUBSETTABLE for the coordinator's post-merge re-run:
#     sbatch --export=ALL,FILES='tests/gpu/test_gfxdraw_fused.py',TAG='_post_gfxdraw' gputests.sbatch
#     sbatch --export=ALL,KFILTER='PushT or TwoRoom',TAG='_swmnav' gputests.sbatch
set -uo pipefail
export SCR=/oscar/scratch/$USER
TREE=${TREE:-$SCR/se6/verify}
source "$TREE/scratch/step6/verify/_env.sh" || exit 1
FILES=${FILES:-"tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py tests/gpu/test_fast_paths.py tests/gpu/test_gfxdraw_fused.py tests/gpu/test_gfxdraw_classic.py tests/gpu/test_rendernet.py tests/gpu/test_crash_isolation.py tests/gpu/test_rollout_on_device.py tests/gpu/test_distributed.py"}
KFILTER=${KFILTER:-}
TAG=${TAG:-}
PY=$(venv_for verify)/bin/python
OUT=$OUTROOT/gpu
mkdir -p "$OUT"
cd "$TREE" || exit 1
banner
$PY -c "import torch,mujoco,pytest,mujoco_warp,sys
print('py',sys.version.split()[0],'torch',torch.__version__,'mujoco',mujoco.__version__,
      'mjwarp',mujoco_warp.__version__,'pytest',pytest.__version__,torch.cuda.get_device_name(0))
print('cuda devices', torch.cuda.device_count())"
export PYTHONPATH=$TREE
K=(); [ -n "$KFILTER" ] && K=(-k "$KFILTER")

for F in $FILES; do
  B=$(basename "$F" .py)
  echo "######################## $F"
  timeout 5400 $PY -m pytest "$F" -v -p no:randomly --timeout=1200 "${K[@]}" \
      2>&1 | keep "$OUT/${B}${TAG}.results.txt"
  echo "FILE $F exit=${PIPESTATUS[0]}"
done
echo "GPUTESTS_DONE $(date -u +%FT%TZ)"
