#!/bin/bash
#SBATCH --job-name=se7-dmc-manipulator-cert311
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The CERTIFICATION arm for DMCManipulatorBringBall at the DECLARED DM Control reference pair.
#
# Everything local was measured at mujoco 3.12.0 + dm_control 1.0.45. The pair this family declares is
# mujoco 3.11.0 + dm_control 1.0.44, and Step 6 closed with the rule that a 3.11-certified cell travels
# in its OWN shard because certify.py reads the reference version off the shard -- a 3.11 row copied
# into a 3.12 document is BLOCKED by construction. So this runs in se-vis311 and writes its own shard.
#
# The interior claim is declared from a DRAW COUNT, not from one large evaluation: >=40 independent
# draws of 64 states, which is the rule PARITY_NOTE records and the reason MujocoPusher's floor failed
# 15 times in 101 after being declared off a single draw.
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
set -x
cd "$SCR/se7/$WS"

$PY -c "import mujoco; from importlib.metadata import version as v; \
        print('mujoco', mujoco.__version__, 'dm_control', v('dm_control')); \
        assert mujoco.__version__.startswith('3.11'), 'wrong pin: this arm must be 3.11'"

# 1. the draw count that declares parity_interior_min, at the certified size and supplementally at 84
$PY scratch/step7/$WS/probe_draws.py --size 224 --draws 41 --n 64 --out "$OUT/oscar"
$PY scratch/step7/$WS/probe_draws.py --size 84  --draws 41 --n 64 --out "$OUT/oscar"

# 2. the full Gate-A conformance run for the DMC shard, CPU, on this pin
$PY examples/fidelity_step4.py --shard dmc --device cpu \
    --out "$OUT/replay/dmc_cpu_oscar311.json"

# 3. the new cell's own frozen corpus (states + digests), proposed for the Step-8 freeze
$PY scratch/step7/$WS/probe_freeze.py --size 224 --n 256 --out "$OUT/corpus"

set +x
ls -la "$OUT/oscar" "$OUT/corpus" || true
