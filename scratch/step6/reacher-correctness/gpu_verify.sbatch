#!/bin/bash
#SBATCH --job-name=se6-reacher-correctness-verify
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 6B verification, all of it, in ONE GPU job on ONE card.
#
# Four things, in this order, because each one is only worth reading if the one before it held:
#
#  1. PIXELS. The frozen Step-4 corpora rendered at the DEFAULT appearance by the baseline tree and
#     by this branch, on the same card, in the same job. That is the same-card rule from
#     AGENT_COMMON applied to pixels: a "before" on one node and an "after" on another is not a
#     before/after even for a digest, because the two would be different GPUs.
#  2. THE STEP-4 REPLAY of the whole dmc shard against the frozen oracle. The whole shard, not the
#     four affected cells: this patch edits code every DM Control cell runs through, and
#     AGENT_COMMON says replay every affected render-distinct configuration.
#  3. THE NEW GUARD, printed per object, on the frozen corpus, as part of that replay.
#
# se-step5 is the venv because the dmc shard's five trackcom cells need mujoco_warp at mujoco 3.12
# and only that venv has it. It has NO PYTEST, which is why the targeted test suites are a SEPARATE
# job (gpu_tests.sbatch, on se-step6) rather than a stanza here. Nothing installs anything.
set -uo pipefail
# Raw pytest output NEVER lands in the repository. It carries absolute
# /oscar/.../se6/... interpreter paths, and tests/unit/test_repo_contracts.py's citation scanner
# reads every .txt under examples/ -- so committing a raw log grew its unresolved-citation set from
# 2 to 18 and failed the suite on a file that is not documentation. ``keep`` tees the full log to
# $SCR/logs (outside the tree) and writes only the OUTCOME lines into the results directory.
keep () {  # keep <results-file> ; reads stdin
  local dest="$1" full="$SCR/logs/${SLURM_JOB_ID:-adhoc}_$(basename "$1")"
  tee "$full" | grep -E "^(tests/\S+ (PASSED|FAILED|SKIPPED|XFAIL|XPASS)|=+ .*(passed|failed|error).*|[0-9]+ (passed|failed).*|== summary ==|DMC\S+ +margin |FAILED |ERROR )" \
    > "$dest" || true
  tail -3 "$dest"
}
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=reacher-correctness
SRC=$SCR/se6/$WS
BASE=$SCR/se6/baseline
PY=$SCR/venvs/se-step5/bin/python
OUT=$SRC/examples/results/step6/$WS
STATES=$SRC/examples/results/step4/states
ENVS=DMCReacherEasy,DMCReacherHard,DMCFingerTurnEasy,DMCFingerTurnHard
mkdir -p "$OUT/replay" "$OUT/gpu"
cd "$SRC"
nvidia-smi --query-gpu=name --format=csv,noheader | tee "$OUT/gpu/card.txt"
$PY -c "import torch,mujoco;print('torch',torch.__version__,'mujoco',mujoco.__version__,torch.cuda.get_device_name(0))"

set -x
# ---- 1. pixels: baseline vs branch, same card, same job ----------------------------------------
PYTHONPATH=$BASE  $PY scratch/step6/$WS/verify_pixels.py --envs "$ENVS" --states "$STATES" \
    --out "$OUT/gpu/pixels_baseline.json"
PYTHONPATH=$SRC   $PY scratch/step6/$WS/verify_pixels.py --envs "$ENVS" --states "$STATES" \
    --out "$OUT/gpu/pixels_branch.json"
PYTHONPATH=$SRC   $PY - "$OUT/gpu/pixels_baseline.json" "$OUT/gpu/pixels_branch.json" <<'PYEOF'
import json, sys
a = json.load(open(sys.argv[1]))["envs"]; b = json.load(open(sys.argv[2]))["envs"]
bad = 0
for k in sorted(a):
    same = a[k].get("digest_ours") == b[k].get("digest_ours") and \
           a[k].get("digest_default") == b[k].get("digest_default")
    bad += (not same)
    print(f"{k:22} default_pixels_identical={same}  style_dim {a[k].get('style_dim')}"
          f"->{b[k].get('style_dim')}  mean {a[k].get('mean_pixel')} -> {b[k].get('mean_pixel')}")
print("PIXEL VERDICT:", "IDENTICAL" if not bad else f"{bad} ENV(S) MOVED")
PYEOF

# ---- 2. the Step-4 replay, whole dmc shard ----------------------------------------------------
export PYTHONPATH=$SRC
$PY examples/fidelity_step4.py --shard dmc --device cuda --states-dir "$STATES" \
    --out "$OUT/replay/dmc.json" 2>&1 | tail -40
$PY scratch/step6/$WS/compare_replay.py \
    "$SRC/examples/results/step4/conformance/dmc.json" "$OUT/replay/dmc.json" \
    2>&1 | tee "$OUT/replay/compare.txt"
set +x
echo "DONE $(date -u +%FT%TZ)"
