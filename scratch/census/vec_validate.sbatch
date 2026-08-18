#!/bin/bash
#SBATCH --job-name=se_vec
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/census/vec_%j.log
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
echo "### speed after vectorization"
/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python -u scratch/census/dmc_profile.py 2>&1 | grep -E "^===|f/s"
echo
echo "### DM Control parity / invariance / determinism"
/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python -m pytest tests/gpu/ -q --no-header --tb=line \
   -k "DMC and (parity or batch_invariance or determinism or state_dim or frame_i)" 2>&1 | tail -12
echo
echo "### MuJoCo locomotion"
/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python -m pytest tests/gpu/ -q --no-header --tb=line \
   -k "Mujoco and (parity or batch_invariance or determinism or state_dim or frame_i)" 2>&1 | tail -12
