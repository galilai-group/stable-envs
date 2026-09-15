#!/bin/bash
#SBATCH --job-name=se7-dmc-manipulator-paired311
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The frozen Step-4 DMC replay as a PAIRED control: baseline 76dc511 and this branch, in the SAME
# job, on the SAME node, at the SAME pin, on the SAME device.
#
# Why this exists. The first 3.11 replay produced digests that differ from examples/results/step4/
# conformance/dmc.json on 10 of 14 cells, and that is NOT a regression signal: the frozen shard's own
# provenance records device "cuda" on gpu2105, and this arm is device=cpu. Comparing them is comparing
# two renderers. The regression question is "did MY commits move an existing cell", and the only sound
# way to ask it on this node is to run both trees here -- the same rule AGENT_COMMON states for GPU
# before/after numbers, applied to a CPU digest.
#
# $SCR/se7/baseline is a pristine git-archive of 76dc511 and is never modified.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=dmc-manipulator
PY=$SCR/venvs/se-vis311/bin/python
OUT=$SCR/se7/$WS/examples/results/step7/$WS
mkdir -p "$OUT/replay"
$PY -c "import mujoco; from importlib.metadata import version as v; \
        print('mujoco', mujoco.__version__, 'dm_control', v('dm_control')); \
        assert mujoco.__version__.startswith('3.11'), 'wrong pin'"
echo "node $(hostname)"
set -x
# BEFORE, on the pristine baseline. Its states-dir is its own frozen corpus, which is identical in
# both trees (examples/results/step4 is read-only in this workstream).
cd "$SCR/se7/baseline"
PYTHONPATH=$SCR/se7/baseline $PY examples/fidelity_step4.py --shard dmc --device cpu \
    --states-dir "$SCR/se7/baseline/examples/results/step4/states" \
    --out "$OUT/replay/dmc_cpu_oscar311_BASE.json"
# AFTER, on this branch
cd "$SCR/se7/$WS"
PYTHONPATH=$SCR/se7/$WS $PY examples/fidelity_step4.py --shard dmc --device cpu \
    --states-dir "$SCR/se7/$WS/examples/results/step4/states" \
    --out "$OUT/replay/dmc_cpu_oscar311_HEAD.json"
set +x
ls -la "$OUT/replay"
