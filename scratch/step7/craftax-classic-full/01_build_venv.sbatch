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
  "jax[cpu]==0.11.1" "craftax==1.6.1" "numpy==2.5.3" gymnasium pytest imageio
# torch from the cu126 wheel index, NOT from PyPI. PyPI's torch 2.14.0 is the +cu130 build, and
# Oscar's 3090 nodes run a CUDA 12.9 driver (12090): the default wheel imports fine on a CPU node
# and dies with "the NVIDIA driver on your system is too old" the moment it touches the card, which
# is how job 6394648 failed in 14 seconds after successfully printing the GPU's name. cu126 is the
# build every Step-6 venv already carries on this cluster, so this matches rather than invents.
uv pip install --python "$VENV/bin/python" --index-url https://download.pytorch.org/whl/cu126 \
  "torch==2.14.0"
"$VENV/bin/python" - <<'PY'
import jax, craftax, torch, numpy
print("jax", jax.__version__, "partitionable", bool(jax.config.jax_threefry_partitionable))
print("craftax", craftax.__version__ if hasattr(craftax, "__version__") else "?",
      "torch", torch.__version__, "cuda", torch.version.cuda, "numpy", numpy.__version__)
assert torch.version.cuda and torch.version.cuda.startswith("12.6"), (
    f"torch is built for CUDA {torch.version.cuda}; Oscar's 3090 driver is 12.9 and will refuse "
    f"anything newer. Install the cu126 wheel.")
PY
