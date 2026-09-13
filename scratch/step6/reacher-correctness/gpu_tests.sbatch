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
OUT=$SRC/examples/results/step6/$WS/gpu
ENVS=DMCReacherEasy,DMCReacherHard,DMCFingerTurnEasy,DMCFingerTurnHard
mkdir -p "$OUT"
cd "$SRC"
export PYTHONPATH=$SRC
export SE_REQUIRE_ENVS="$ENVS"
nvidia-smi --query-gpu=name --format=csv,noheader
$PY -c "import torch,mujoco,pytest;print('torch',torch.__version__,'mujoco',mujoco.__version__,'pytest',pytest.__version__,torch.cuda.get_device_name(0))"

set -x
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
