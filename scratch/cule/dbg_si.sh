#!/bin/bash
SCRATCH=/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/cule
VENV=/mnt/vast/home/randall.balestriero/venvs/cule
export CUDA_HOME=$SCRATCH/cudahome
source "$VENV/bin/activate"
cd "$SCRATCH/cule_mehrdad"
python -u "$SCRATCH/parity_cule_ale.py" "$SCRATCH/parity_dbg.json" --steps 20 --roms space_invaders
echo "EXIT=$?"
