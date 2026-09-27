#!/bin/bash
#SBATCH --job-name=se-randall-measure
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/randall-renderer-636bc586/measure-%j.log
set -euo pipefail
root=/oscar/scratch/sboughan/se-semantic-recovery/randall-renderer-636bc586
run=$root/measure-$SLURM_JOB_ID
mkdir -p "$run/source" "$run/oracle"
tar -xzf "$root/source-v1.tar.gz" -C "$run/source"
tar -xzf /oscar/scratch/sboughan/se-semantic-recovery/state-bench-gpu-8d153702/oracle.tar.gz -C "$run/oracle"
tar -xzf "$root/randall-harness-v6.tar.gz" -C "$run/source"
export SWM_ROOT="$run/oracle"
export PYTHONPATH="$run/source:$run/oracle:/oscar/scratch/sboughan/venvs/se-vis311/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step6/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step4-swm/lib/python3.12/site-packages"
export DISPLAY=:$((100 + SLURM_JOB_ID % 5000))
Xvfb "$DISPLAY" -screen 0 1024x768x24 -nolisten tcp > "$run/xvfb.log" 2>&1 &
xpid=$!
trap 'kill "$xpid" 2>/dev/null || true' EXIT
sleep 2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 SDL_VIDEODRIVER=dummy
export TORCH_EXTENSIONS_DIR="$run/torch-cache" WARP_CACHE_PATH="$run/warp-cache"
export XDG_CACHE_HOME="$run/cache"
true
lscpu > "$run/lscpu.txt"
sha256sum "$root/source-v1.tar.gz" > "$run/archive.sha256"
cd "$run/source"
MUJOCO_GL=glfw PYOPENGL_PLATFORM=glx LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe /oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python -m scripts.benchmarks.renderer_table.profile_probe "$run/cpu-gl.json" --gl
export PYTHONPATH="/oscar/scratch/sboughan/se-semantic-recovery/randall-renderer-636bc586/measure-6739712/fetch-cpu-overlay:$run/source:$run/oracle:/oscar/scratch/sboughan/venvs/se-s3b-fetch311/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step6/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step4-swm/lib/python3.12/site-packages"
MUJOCO_GL=glfw PYOPENGL_PLATFORM=glx LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe /oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python -m scripts.benchmarks.renderer_table.fetch_mutation_probe "$run/mutation.json"
