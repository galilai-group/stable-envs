#!/bin/bash
#SBATCH --job-name=se3d_ceil
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:30:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/ceil_%j.log
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
DMC=/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python
echo "### the whole DMC + gym MuJoCo registry still imports and registers"
$DMC -u -c "import stable_envs as se; r=sorted(se.REGISTRY); d=[n for n in r if n.startswith(('DMC','Mujoco'))]; print(len(d),'cells:',d)"
echo
echo "### ceiling decomposition, 256px b1024"
$DMC -u scratch/perf3d/ceiling.py --batch 1024 --size 256 \
  --json /mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/ceiling.json
echo
echo "### per-pass at the real chunk size"
$DMC -u scratch/perf3d/passes256.py --size 256 --batch 64 \
  --envs DMCReacherEasy,DMCFingerTurnEasy,DMCCartpoleBalance,DMCBallInCupCatch,MujocoPusher,MujocoReacher
