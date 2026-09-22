#!/bin/bash
#SBATCH --job-name=se-reacher-gate
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/logs/reacher-%j.out
set -euo pipefail
base=/oscar/scratch/sboughan/se-semantic-recovery
py=/oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=4
export TORCH_EXTENSIONS_DIR="$base/cache/torch" TRITON_CACHE_DIR="$base/cache/triton" XDG_CACHE_HOME="$base/cache" WARP_CACHE_PATH="$base/cache/warp"
export PYTHONPATH="$base/reacher:$base/swm-reference:/oscar/scratch/sboughan/venvs/se-vis311/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step6/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step4-swm/lib/python3.12/site-packages"
nvidia-smi
cd "$base/reacher"
"$py" scripts/recovery/reacher_gate.py --device cuda --output "$base/reacher-gate-gpu.json"
"$py" scripts/recovery/renderer_boundary.py --output "$base/reacher-renderer-gpu.json"
"$py" -c 'import json; from pathlib import Path; p=Path("/oscar/scratch/sboughan/se-semantic-recovery"); a=json.loads((p/"baseline-gpu.json").read_text()); b=json.loads((p/"reacher-renderer-gpu.json").read_text()); assert a==b; print("REACHER_AND_FROZEN_RENDERER_GATE_PASS", len(a), "cells")'
