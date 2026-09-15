#!/bin/bash
#SBATCH --job-name=se7-craftax-classic-full-official-gpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# The official reference on the SAME CARD as ours, which the earlier run did not do: env_header.sh
# exports JAX_PLATFORMS=cpu and both tier jobs sourced it, so the reference never touched the GPU it
# was being compared on. Here jax runs on the card, with preallocation off so jax and torch can
# share it rather than jax claiming 75% of it at import.
set -euo pipefail
source /oscar/scratch/sboughan/se7/craftax-classic-full/scratch/step7/craftax-classic-full/env_header.sh
unset JAX_PLATFORMS
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# jax[cpu] cannot use a GPU, and the parity venv has jax[cpu]: the first attempt unset
# JAX_PLATFORMS and still reported jax_default_backend = "cpu". A SEPARATE venv carries the CUDA
# jaxlib, so the venv the parity numbers were measured in is left exactly as it was.
VENV=$SCR/venvs/se7-craftax-gpu
if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.12 "$VENV"
  uv pip install --python "$VENV/bin/python" "jax[cuda12]==0.11.1" "craftax==1.6.1" \
    "numpy==2.5.3" gymnasium imageio
  uv pip install --python "$VENV/bin/python" --index-url https://download.pytorch.org/whl/cu126 \
    --reinstall-package torch "torch==2.14.0"
fi
"$VENV/bin/python" -c "import jax;print('jax',jax.__version__,'backend',jax.default_backend(),jax.devices())"
echo "== host $(hostname) =="; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
"$VENV/bin/python" scratch/step7/craftax-classic-full/official_timing.py \
  --device cuda --batch 4096 --tag "$(hostname)"
