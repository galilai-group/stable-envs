#!/bin/bash
#SBATCH --job-name=se-randall-profiles
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090|l40s|a40|a5000|a5500|quadrortx
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/randall-renderer-636bc586/probe-%j.log
set -euo pipefail
root=/oscar/scratch/sboughan/se-semantic-recovery/randall-renderer-636bc586
run=$root/probe-$SLURM_JOB_ID
mkdir -p "$run"
nvidia-smi > "$run/nvidia-smi.txt"
lscpu > "$run/lscpu.txt"
export DISPLAY=:$((100 + SLURM_JOB_ID % 5000))
Xvfb "$DISPLAY" -screen 0 1024x768x24 -nolisten tcp > "$run/xvfb.log" 2>&1 &
xpid=$!
trap 'kill "$xpid" 2>/dev/null || true' EXIT
sleep 2
export MUJOCO_GL=glfw PYOPENGL_PLATFORM=glx LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export PYTHONPATH=/oscar/scratch/sboughan/venvs/se-vis311/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step6/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step4-swm/lib/python3.12/site-packages
/oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python "$root/profile_probe.py" "$run/normal.json" --gl
unset PYTHONPATH
/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python "$root/profile_probe.py" "$run/fetch.json" --gl
/oscar/scratch/sboughan/venvs/se-s3b-fetch311/bin/python "$root/profile_probe.py" "$run/fetch-base.json" --gl
/oscar/scratch/sboughan/se-semantic-recovery/ogb-renderer-294675f0-complete/run-6669909/profile/bin/python "$root/profile_probe.py" "$run/manipulation.json" --gl
/oscar/scratch/sboughan/se-semantic-recovery/stable-envs-craftax-b6a8242f-gpu/run-6679082/venv/bin/python "$root/profile_probe.py" "$run/craftax.json"
printf '{"job_id":"%s","status":"PROFILE_PROBE_COMPLETE"}\n' "$SLURM_JOB_ID" > "$run/exit.json"
