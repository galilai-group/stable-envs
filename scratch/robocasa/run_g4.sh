#!/usr/bin/env bash
# Re-measure G4 for RoboCasa. Runs on a GPU node via srun; logs under /mnt/vast.
set -u
R=/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa
L=$R/logs
mkdir -p "$L" "$R/out"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WARP_CACHE_PATH=/mnt/vast/home/randall.balestriero/.cache/warp
SE3D=/mnt/vast/home/randall.balestriero/venvs/se3d/bin/python
RC=/mnt/vast/home/randall.balestriero/miniconda3/envs/robocasa/bin/python

echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

case "${1:-all}" in
  egl)
    echo "--- EGL decomposition, mujoco 3.11 (se3d) ---"
    $SE3D $R/probe_g4_egl.py --tag egl_mj311 2>&1 | tee "$L/g4_egl_mj311.log"
    echo "--- EGL decomposition, mujoco 3.3.1 (robocasa env) ---"
    $RC $R/probe_g4_egl.py --tag egl_mj331 2>&1 | tee "$L/g4_egl_mj331.log"
    ;;
  mjwarp)
    echo "--- MJWarp batch regeneration ---"
    $SE3D $R/probe_g4_mjwarp.py 2>&1 | tee "$L/g4_mjwarp.log"
    ;;
  texgen)
    echo "--- texgen root cause / bake / BGRA ---"
    $SE3D $R/probe_texgen.py 2>&1 | tee "$L/texgen.log"
    ;;
  all)
    bash "$0" egl
    bash "$0" mjwarp
    bash "$0" texgen
    ;;
esac
