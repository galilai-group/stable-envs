#!/bin/bash
# Build $SCR/venvs/se8-mjwarp313: se7-ogb's pin set with ONLY mujoco and mujoco-warp bumped to
# 3.13.0. Everything else is held at the version the OGB cells were measured under, so a difference
# observed under this venv is attributable to the MuJoCo pair and not to a coincidental bump.
# Runs on a COMPUTE node only (srun --jobid=<alloc> --overlap ... or sbatch). Never on a login node.
set -euo pipefail
: "${SCR:?set SCR to the scratch root before running, e.g. SCR=$SCR bash "$0"}"
export UV_CACHE_DIR=$SCR/cache/uv PIP_CACHE_DIR=$SCR/cache/pip XDG_CACHE_HOME=$SCR/cache/xdg
export TORCH_HOME=$SCR/cache/torch WARP_CACHE_PATH=$SCR/cache/warp HF_HOME=$SCR/cache/hf
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$WARP_CACHE_PATH"
VENV=$SCR/venvs/se8-mjwarp313
UV=$(command -v uv)
$UV venv --python 3.12 "$VENV"
$UV pip install --python "$VENV/bin/python" \
  --torch-backend=cu126 \
  'mujoco==3.13.0' 'mujoco-warp==3.13.0' 'warp-lang==1.17.0' \
  'torch==2.14.0' 'torchvision==0.29.0' 'triton==3.8.0' \
  'numpy==2.5.2' 'scipy==1.18.1' 'dm-control==1.0.45' 'gymnasium==1.3.0' \
  'gymnasium-robotics==1.4.2' 'ogbench==1.2.1' 'pygame==2.6.1' 'pymunk==7.3.0' \
  'opencv-python-headless==5.0.0.93' 'imageio==2.37.4' 'pillow==12.3.0' 'h5py==3.16.0' \
  'pytest==9.1.1' 'loguru==0.7.3' 'shapely==2.1.2' 'tabulate==0.10.0' 'einops==0.8.2' \
  'pyopengl==3.1.10' 'glfw==2.10.2' 'lxml==6.1.3' 'labmaze==1.0.6' 'tqdm==4.70.1'
"$VENV/bin/python" -c "import mujoco, mujoco_warp, warp; print('mujoco', mujoco.__version__)"
