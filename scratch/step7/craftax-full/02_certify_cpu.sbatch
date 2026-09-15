#!/bin/bash
#SBATCH --job-name=se7-craftax-full-cert-cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# Gate A + parity on x86-64 at one torch thread, plus the unit suite. The renderer is float32
# arithmetic over a bit-level integer PRNG; nothing about torch guarantees either across
# architectures, so arm64/x86 agreement is MEASURED (identical frame digests) rather than assumed.
set -euo pipefail
source /oscar/scratch/sboughan/se7/craftax-full/scratch/step7/craftax-full/env_header.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
VENV=$SCR/venvs/se7-craftax
P=scratch/step7/craftax-full
echo "== host $(hostname) arch $(uname -m) =="; cat /proc/loadavg
md5sum stable_envs/envs/craftax_full.py stable_envs/envs/_craftax_full_tex.py stable_envs/render/prng.py
"$VENV/bin/python" $P/certify.py --device cpu --tag "$(hostname)" \
  --corpus examples/results/step7/craftax-full/corpus/states.npz
"$VENV/bin/python" -m pytest tests/unit/test_craftax_full.py -q
