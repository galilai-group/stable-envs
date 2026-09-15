#!/bin/bash
#SBATCH --job-name=se7-craftax-full-bars-cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# The two certification bars on a batch node, one thread: the official renderer (un-jitted AND
# jitted) and one core of JPEG-q90 decode of this env's own frame. --exclusive because W1 measured
# a contended node at 6,219 f/s where the same code does 126,845 on a laptop, and a bar taken on a
# contended node is not a bar; /proc/loadavg is printed on both sides so the reader can check.
set -euo pipefail
source /oscar/scratch/sboughan/se7/craftax-full/scratch/step7/craftax-full/env_header.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
VENV=$SCR/venvs/se7-craftax
P=scratch/step7/craftax-full
echo "== host $(hostname) arch $(uname -m) =="; cat /proc/loadavg
"$VENV/bin/python" $P/harvest.py
"$VENV/bin/python" $P/probe_bars.py
PYTHONPATH=$WS "$VENV/bin/python" $P/probe_window.py
cat /proc/loadavg
