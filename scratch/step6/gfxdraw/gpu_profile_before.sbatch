#!/bin/bash
#SBATCH --job-name=se6-gfxdraw-gpuprof
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The GPU profile-BEFORE, which the main GPU job could not write.
#
# $SCR/se6/baseline is pristine at 89fa5ca and still carries profile_note.py's NameError at the
# write step, so every before-arm run there measured the whole sweep and then died before writing
# a file. This runs the BRANCH's copy of the harness against the baseline TREE: the module under
# measurement comes from PYTHONPATH, the harness does not.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
PY=$SCR/venvs/se-step6/bin/python
OUT=$SCR/se6/gfxdraw/examples/results/step6/gfxdraw
mkdir -p "$OUT/profile_before_gpu"
echo "NODE $(hostname -s) JOB $SLURM_JOB_ID"; nvidia-smi --query-gpu=name --format=csv,noheader
set -x
for E in Acrobot CartPole MountainCar MountainCarContinuous Pendulum; do
  cd "$SCR/se6/baseline"
  PYTHONPATH=$SCR/se6/baseline $PY $SCR/se6/gfxdraw/scratch/step6/profile_note.py \
      --env $E --device cuda --batch 1 16 64 256 1024 4096 --repeat 10 --warmup 3 \
      --workstream step6/gfxdraw \
      --states-dir "$SCR/se6/baseline/examples/results/step4/states" \
      --out "$OUT/profile_before_gpu" || echo "GPU PROFILE BEFORE $E rc=$?"
done
echo "GPU PROFILE BEFORE JOB DONE"
