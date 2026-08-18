#!/bin/bash
#SBATCH --job-name=culeparity
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-070,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/cule/parity.%j.out
SCRATCH=/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/cule
VENV=/mnt/vast/home/randall.balestriero/venvs/cule
export CUDA_HOME=$SCRATCH/cudahome
source "$VENV/bin/activate"
case "$(command -v python)" in "$VENV"/*) ;; *) echo "FATAL python=$(command -v python)"; exit 1;; esac
cd "$SCRATCH/cule_mehrdad"
python "$SCRATCH/parity_cule_ale.py" "$SCRATCH/parity_cule.json" --steps 200
echo "PARITY_EXIT=$?"
