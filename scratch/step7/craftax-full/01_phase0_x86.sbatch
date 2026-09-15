#!/bin/bash
#SBATCH --job-name=se7-craftax-full-phase0-x86
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# Re-take every phase-0 bit-exactness claim on x86-64. The three reconstructions (world layer,
# lighting chain, strip) are float32 arithmetic and the PRNG underneath them is bit-level integer
# arithmetic; nothing about torch guarantees either across architectures, so agreement is measured
# rather than assumed -- exactly as W1 did for Classic.
set -euo pipefail
source /oscar/scratch/sboughan/se7/craftax-full/scratch/step7/craftax-full/env_header.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
VENV=$SCR/venvs/se7-craftax
P=scratch/step7/craftax-full
echo "== host $(hostname) arch $(uname -m) =="
md5sum $P/probe_composite.py $P/probe_chain.py $P/probe_strip.py stable_envs/render/prng.py
"$VENV/bin/python" $P/harvest.py
PYTHONPATH=$WS:$WS/$P "$VENV/bin/python" $P/probe_census.py
PYTHONPATH=$WS:$WS/$P "$VENV/bin/python" $P/probe_layers.py
PYTHONPATH=$WS:$WS/$P "$VENV/bin/python" $P/probe_composite.py
PYTHONPATH=$WS:$WS/$P "$VENV/bin/python" $P/probe_chain.py
PYTHONPATH=$WS:$WS/$P "$VENV/bin/python" $P/probe_strip.py
PYTHONPATH=$WS:$WS/$P "$VENV/bin/python" $P/probe_purity_state.py
