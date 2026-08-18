#!/bin/bash
#SBATCH --job-name=atstorage
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-070,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/cule/storage.%j.out
SCRATCH=/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/cule
VENV=/mnt/vast/home/randall.balestriero/venvs/envpool
source "$VENV/bin/activate"
case "$(command -v python)" in "$VENV"/*) ;; *) echo "FATAL python=$(command -v python)"; exit 1;; esac
pip install -q opencv-python-headless
python "$SCRATCH/bench_atari_storage.py" "$SCRATCH/atari_storage.json"
echo "STORAGE_EXIT=$?"
