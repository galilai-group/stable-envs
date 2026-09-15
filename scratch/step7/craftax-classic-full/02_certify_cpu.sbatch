#!/bin/bash
#SBATCH --job-name=se7-craftax-classic-full-cert-cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# Gate A + parity on a batch node, with torch pinned to one thread the way a certification
# measurement is defined. The parity leg needs the jax venv from 01; everything else would run
# anywhere, and is run here so one artifact carries both.
set -euo pipefail
source "$(dirname "$0")/env_header.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
VENV=$SCR/venvs/se7-craftax
echo "== host $(hostname) =="
md5sum stable_envs/envs/craftax.py stable_envs/render/prng.py
"$VENV/bin/python" scratch/step7/craftax-classic-full/certify.py --device cpu --tag "$(hostname)"
"$VENV/bin/python" -m pytest tests/unit/test_craftax_prng.py tests/unit/test_craftax_strip.py \
  tests/unit/test_craftax_atlas.py -q
# the PRNG is bit-level integer arithmetic; arm64 and x86-64 agreement is checked, not assumed
"$VENV/bin/python" scratch/step7/craftax-classic-full/probe_prng.py
