#!/bin/bash
#SBATCH --job-name=se6-reacher-correctness-repro
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# VISIBILITY-1: tests/gpu/test_reproducibility.py in FULL, no -k, on the merged branch.
#
# The whole point of this job is that the previous runs of this file were -k filtered, which is how
# the new visibility test's selector bug survived: it swept 11 DM Control cells and only the 4 it was
# written for were ever collected. No selection here.
#
# gpu2108 and gpu2113 are excluded by the coordinator: their official OGBench renderer is
# nondeterministic, and this file's reference-renderer checks would blame the repo for the node.
#
# VENV: se-step6, NOT se-step5. se-step5 is the only venv with mujoco_warp and it has NO pytest at
# all (`No module named pytest`), so it cannot run a test file; se-step6 has pytest 9.1.1 and no
# mujoco_warp, which costs nothing here because every cell this file's visibility sweep touches is
# DMControlSuite (plate + ray-cast, raycast3d only) and mujoco_warp is imported lazily. The trackcom
# cells skip themselves on the missing import, which the summary below reports.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=reacher-correctness
SRC=$SCR/se6/$WS
PY=$SCR/venvs/se-step6/bin/python
OUT=$SRC/examples/results/step6/$WS/gpu
mkdir -p "$OUT"
cd "$SRC"
export PYTHONPATH=$SRC
# Raw output never lands in the tree; see gpu_tests.sbatch's `keep` for why.
keep () {
  local dest="$1" full="$SCR/logs/${SLURM_JOB_ID:-adhoc}_$(basename "$1")"
  tee "$full" | grep -E "^(tests/\S+ (PASSED|FAILED|SKIPPED|XFAIL|XPASS)|=+ .*(passed|failed|error).*|[0-9]+ (passed|failed).*|visibility leaves declared by)" \
    > "$dest" || true
  tail -3 "$dest"
}
hostname
nvidia-smi --query-gpu=name --format=csv,noheader
$PY -c "import torch,mujoco,pytest;print('torch',torch.__version__,'mujoco',mujoco.__version__,'pytest',pytest.__version__,torch.cuda.get_device_name(0))"

set -x
$PY -m pytest tests/gpu/test_reproducibility.py -v -p no:randomly \
    2>&1 | keep "$OUT/test_reproducibility_full.results.txt"
set +x
echo "--- the visibility sweep, in full ---"
grep "baked_scene_does_not_inherit\|actually_declares_a_hideable" "$OUT/test_reproducibility_full.results.txt"
echo "DONE $(date -u +%FT%TZ)"
