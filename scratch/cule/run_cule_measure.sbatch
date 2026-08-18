#!/bin/bash
#SBATCH --job-name=culemeas
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-070,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/cule/culemeas.%j.out
bash /mnt/vast/home/randall.balestriero/code/stable-envs/scratch/cule/run_cule_measure.sh
