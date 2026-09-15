#!/bin/bash
#SBATCH --job-name=se7-dmc-manipulator-corpus311
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 3 of oscar_certify311.sbatch on its own: the candidate frozen corpus at the DECLARED pin.
# Split out because the first run of that script passed "--n 256" to probe_freeze.py, whose flags C3
# had just redefined (--pool is the sample, --n the SELECTED corpus), so select() kept all 256 rows --
# the exact defect C3 existed to fix, reproduced in the arm that matters. The 41-draw and replay steps
# of that job were correct and are not re-run here; only this step is.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=dmc-manipulator
export PYTHONPATH=$SCR/se7/$WS
PY=$SCR/venvs/se-vis311/bin/python
OUT=$SCR/se7/$WS/examples/results/step7/$WS
cd "$SCR/se7/$WS"
$PY -c "import mujoco; from importlib.metadata import version as v; \
        print('mujoco', mujoco.__version__, 'dm_control', v('dm_control')); \
        assert mujoco.__version__.startswith('3.11'), 'wrong pin: this arm must be 3.11'"
rm -rf "$OUT/corpus"
set -x
$PY scratch/step7/$WS/probe_freeze.py --size 224 --pool 256 --n 64 --out "$OUT/corpus"
rc=$?
set +x
echo "corpus rc=$rc"
ls -la "$OUT/corpus" "$OUT/corpus/states" || true
exit $rc
