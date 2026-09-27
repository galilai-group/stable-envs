#!/bin/bash
#SBATCH --job-name=se-randall-measure
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090|l40s|a40|a5000|a5500|quadrortx
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/randall-renderer-636bc586/measure-%j.log
set -euo pipefail
root=/oscar/scratch/sboughan/se-semantic-recovery/randall-renderer-636bc586
run=$root/measure-$SLURM_JOB_ID
mkdir -p "$run/source" "$run/oracle"
tar -xzf "$root/source-v1.tar.gz" -C "$run/source"
tar -xzf /oscar/scratch/sboughan/se-semantic-recovery/state-bench-gpu-8d153702/oracle.tar.gz -C "$run/oracle"
tar -xzf "$root/randall-harness-v3.tar.gz" -C "$run/source"
export SWM_ROOT="$run/oracle"
export PYTHONPATH="$run/source:$run/oracle:/oscar/scratch/sboughan/venvs/se-s3b-fetch311/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step6/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step4-swm/lib/python3.12/site-packages"
export DISPLAY=:$((100 + SLURM_JOB_ID % 5000))
Xvfb "$DISPLAY" -screen 0 1024x768x24 -nolisten tcp > "$run/xvfb.log" 2>&1 &
xpid=$!
trap 'kill "$xpid" 2>/dev/null || true' EXIT
sleep 2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 SDL_VIDEODRIVER=dummy
export TORCH_EXTENSIONS_DIR="$run/torch-cache" WARP_CACHE_PATH="$run/warp-cache"
export XDG_CACHE_HOME="$run/cache"
nvidia-smi > "$run/nvidia-smi.txt"
lscpu > "$run/lscpu.txt"
sha256sum "$root/source-v1.tar.gz" > "$run/archive.sha256"
export FETCH_CPU_OVERLAY="$run/fetch-cpu-overlay"
/oscar/home/sboughan/.local/bin/uv pip install --python /oscar/scratch/sboughan/venvs/se-s3b-fetch311/bin/python --no-deps --target "$FETCH_CPU_OVERLAY" mujoco==3.8.1 numpy==2.2.6 gymnasium==1.2.3 > "$run/install.log" 2>&1
cd "$run/source"
PYTHONPATH="$FETCH_CPU_OVERLAY:$PYTHONPATH" MUJOCO_GL=glfw PYOPENGL_PLATFORM=glx LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe /oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python -m scripts.benchmarks.renderer_table.profile_probe "$run/cpu-gl.json" --gl
/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python -m scripts.benchmarks.renderer_table.run_cohort fetch "$run/results"
printf '{"status":"COHORT_RUNNER_FINISHED"}\n' > "$run/exit.json"
