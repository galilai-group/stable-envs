#!/bin/bash
#SBATCH --job-name=se6-reacher-correctness-tests
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The targeted test suites, scoped to the four cells this patch touches.
#
# SPLIT FROM gpu_verify.sbatch FOR ONE REASON, recorded so nobody re-merges them: $SCR/venvs/se-step5
# is the venv that has mujoco_warp -- which the dmc shard's five trackcom cells need, so the Step-4
# REPLAY has to run there -- and it has NO PYTEST. se-step6 has pytest and no mujoco_warp, which is
# fine here because stable_envs/render/mjwarp.py imports mujoco_warp lazily inside its functions and
# the four cells under test are all DMControlSuite (plate + ray-cast, raycast3d only). The -k filter
# means no trackcom cell is ever constructed.
#
# One GPU job per agent at a time (AGENT_COMMON), so this is submitted only after gpu_verify has
# finished, not alongside it.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=reacher-correctness
SRC=$SCR/se6/$WS
PY=$SCR/venvs/se-step6/bin/python
PY5=$SCR/venvs/se-step5/bin/python          # the only venv with mujoco_warp; no pytest
BASE=$SCR/se6/baseline
STATES=$SRC/examples/results/step4/states
#: Every DM Control cell, parity AND trackcom. The first replay reported three TRACKCOM cells
#: (CheetahRun, HopperHop, HumanoidStand) whose default digest differs from the frozen Step-4 one,
#: and the frozen run was taken in a DIFFERENT venv on a DIFFERENT node (se-step6 on gpu2105, job
#: 6176998) -- so that comparison cannot separate "this patch" from "this machine". Rendering the
#: BASELINE tree and this branch back to back on ONE card settles it: if the baseline also differs
#: from the frozen digest, the drift is environmental and predates this branch.
ALL=DMCAcrobotSwingup,DMCBallInCupCatch,DMCCartpoleBalance,DMCCartpoleSwingup,DMCFingerSpin,DMCFingerTurnEasy,DMCFingerTurnHard,DMCPendulumSwingup,DMCPointMassEasy,DMCReacherEasy,DMCReacherHard,DMCCheetahRun,DMCHopperHop,DMCHumanoidStand,DMCSwimmerSwimmer6,DMCWalkerWalk
OUT=$SRC/examples/results/step6/$WS/gpu
ENVS=DMCReacherEasy,DMCReacherHard,DMCFingerTurnEasy,DMCFingerTurnHard
mkdir -p "$OUT"
cd "$SRC"
export PYTHONPATH=$SRC
export SE_REQUIRE_ENVS="$ENVS"
nvidia-smi --query-gpu=name --format=csv,noheader
$PY -c "import torch,mujoco,pytest;print('torch',torch.__version__,'mujoco',mujoco.__version__,'pytest',pytest.__version__,torch.cuda.get_device_name(0))"

set -x
# ---- 0. baseline vs branch over EVERY dm cell, same card, same job -----------------------------
PYTHONPATH=$BASE $PY5 scratch/step6/$WS/verify_pixels.py --envs "$ALL" --states "$STATES" \
    --out "$OUT/pixels_all_baseline.json"
PYTHONPATH=$SRC  $PY5 scratch/step6/$WS/verify_pixels.py --envs "$ALL" --states "$STATES" \
    --out "$OUT/pixels_all_branch.json"
$PY - "$OUT/pixels_all_baseline.json" "$OUT/pixels_all_branch.json" \
     "$SRC/examples/results/step4/conformance/dmc.json" <<'PYEOF' | tee "$OUT/pixels_all.txt"
import json, sys
a = json.load(open(sys.argv[1]))["envs"]; b = json.load(open(sys.argv[2]))["envs"]
frozen = json.load(open(sys.argv[3]))["configs"]
moved = froze = 0
for k in sorted(a):
    same = (a[k].get("digest_ours") == b[k].get("digest_ours")
            and a[k].get("digest_default") == b[k].get("digest_default"))
    fz = ((frozen.get(k) or {}).get("oracle") or {}).get("digest_ours")
    vs_frozen = "n/a" if fz is None else ("same" if fz == a[k].get("digest_ours") else "DIFFERS")
    moved += (not same); froze += (vs_frozen == "DIFFERS")
    print(f"{k:22} baseline_vs_branch={'identical' if same else 'MOVED':10} "
          f"baseline_vs_frozen_step4={vs_frozen}")
print(f"VERDICT: {moved} env(s) moved between the two TREES; "
      f"{froze} env(s) where the BASELINE tree already differs from the frozen Step-4 digest")
PYEOF

$PY -m pytest tests/gpu/test_conformance.py -v -p no:randomly \
    -k "Reacher or FingerTurn or required_envs or registry_is_not_empty or relaxed_parity" \
    2>&1 | tee "$OUT/test_conformance.txt" | tail -70
$PY -m pytest tests/gpu/test_reproducibility.py -v -p no:randomly \
    -k "Reacher or FingerTurn" 2>&1 | tee "$OUT/test_reproducibility.txt" | tail -70
$PY -m pytest tests/gpu/test_fast_paths.py -v -p no:randomly \
    -k "Reacher or FingerTurn or unknown_geom or flat_polygon or declared_fast_path" \
    2>&1 | tee "$OUT/test_fast_paths.txt" | tail -40
$PY -m pytest tests/unit -q -p no:randomly 2>&1 | tee "$OUT/test_unit.txt" | tail -20
set +x
echo "DONE $(date -u +%FT%TZ)"
