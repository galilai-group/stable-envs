#!/bin/bash
#SBATCH --job-name=se_ddp
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/census/ddp_%j.log
R=/mnt/vast/home/randall.balestriero/code/stable-envs
cd $R
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
PY=/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python
echo "### GPUs:"; nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "### unit tests"
$PY -m pytest tests/gpu/test_distributed.py -v -rs 2>&1 | tail -40
echo "### ddp example, 2 ranks"
$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29611 \
    examples/ddp_train_from_states.py --env PushT --steps 60 2>&1 | tail -30
