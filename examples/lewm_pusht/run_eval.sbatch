#!/bin/bash
#SBATCH --job-name=se_lewm_pusht
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-070,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/code/stable-envs/examples/lewm_pusht/logs/eval-%j.out
# RENDERER=official|stable_envs|stable_envs_resize512   SEED=42   NEVAL=50
set -x
EX=/mnt/vast/home/randall.balestriero/code/stable-envs/examples/lewm_pusht
cd "$EX"
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
PY=/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python
$PY eval_lewm_pusht.py \
    policy=pusht/lewm \
    renderer="${RENDERER:-official}" \
    seed="${SEED:-42}" \
    eval.num_eval="${NEVAL:-50}"
