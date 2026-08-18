#!/bin/bash
#SBATCH --job-name=b3d_mjwarp
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-070,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/backend3d/mjwarp_%j.log
D=/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/backend3d
/mnt/vast/home/randall.balestriero/venvs/se3d/bin/python $D/bench_mjwarp.py \
  --batches 1,8,64,256,1024,4096 --res 224 \
  --out $D/results/mjwarp_224.json --images $D/images
/mnt/vast/home/randall.balestriero/venvs/se3d/bin/python $D/bench_mjwarp.py \
  --batches 1024 --res 64 --out $D/results/mjwarp_64.json
