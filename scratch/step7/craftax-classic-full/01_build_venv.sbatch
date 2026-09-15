#!/bin/bash
#SBATCH --job-name=se7-craftax-classic-full-venv
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# Build the ONLY interpreter on Oscar that can run this workstream's parity check.
#
# No existing Oscar venv has jax. Step 6 recorded render_parity_with_official_renderer as
# blocked_environment for exactly that reason, and this workstream's whole claim needs it. A new
# venv is created rather than any existing one being modified (AGENT_COMMON: do not rebuild or
# install into the shared interpreters).
#
# The pins match the local reference venv byte for byte, because the parity bound is DECLARED
# against them: CraftaxClassic.reference_versions is {"craftax": "1.6.1", "jax": "0.11.1"} and a
# mismatch is judged blocked_environment rather than pass or fail.
set -euo pipefail
# SLURM copies the script to its spool dir, so $0 is NOT in the worktree: source the
# header from the workspace by absolute path.
source /oscar/scratch/sboughan/se7/craftax-classic-full/scratch/step7/craftax-classic-full/env_header.sh
VENV=$SCR/venvs/se7-craftax
if [ -d "$VENV" ]; then echo "exists: $VENV"; else
  uv venv --python 3.12 "$VENV"
fi
uv pip install --python "$VENV/bin/python" \
  "jax[cpu]==0.11.1" "craftax==1.6.1" "torch==2.14.0" "numpy==2.5.3" \
  gymnasium pytest imageio
"$VENV/bin/python" - <<'PY'
import jax, craftax, torch, numpy
print("jax", jax.__version__, "partitionable", bool(jax.config.jax_threefry_partitionable))
print("craftax", craftax.__version__ if hasattr(craftax, "__version__") else "?",
      "torch", torch.__version__, "numpy", numpy.__version__)
PY
