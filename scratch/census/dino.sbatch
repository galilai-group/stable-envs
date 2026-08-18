#!/bin/bash
#SBATCH --job-name=se-dino
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_dino_%j.out
set -uo pipefail
R=/mnt/vast/home/randall.balestriero/code/stable-envs
cd $R; export PYTHONPATH=$R MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1
export HF_HUB_OFFLINE=1                      # the weights are already in ~/.cache/huggingface
UV=/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/uv
PY=/mnt/vast/home/randall.balestriero/venvs/se-all/bin/python
echo "### node=$(hostname)"
"$UV" pip install --python "$PY" --quiet timm 2>&1 | tail -3
"$PY" -c "import timm;print('timm', timm.__version__)"
"$PY" -u examples/dino_agreement.py --n 32 --out $R/examples/results/dino > $R/examples/results/dino/run.log 2>&1; tail -60 $R/examples/results/dino/run.log
echo "### dino done"
