#!/bin/bash
#SBATCH --job-name=se_ddp4
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=01:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/census/ddp4_%j.log

# The paper's DDP table has ranks 1 and 2 only, which is two points and therefore not a scaling claim.
# This adds 4 and 8 on one node. Same script, same env, same steps, so the four rows are comparable.
R=/mnt/vast/home/randall.balestriero/code/stable-envs
cd $R
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
PY=/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python
for W in 4 8; do
  echo "### world_size=$W"
  $PY -m torch.distributed.run --nproc_per_node=$W --master_port=2963$W \
      examples/ddp_train_from_states.py --env PushT --steps 400 \
      --out $R/examples/results/ddp_w$W 2>&1 | tail -22
done
