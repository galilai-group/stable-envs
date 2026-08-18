#!/bin/bash
#SBATCH --job-name=b3d_egl
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-070,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/backend3d/egl_%j.log
D=/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/backend3d
export MUJOCO_GL=egl
/mnt/vast/home/randall.balestriero/venvs/se3d/bin/python $D/bench_egl.py \
  --batch 512 --res 224 --threads 1,2,4,8,16 \
  --out $D/results/egl_224.json --images $D/images
