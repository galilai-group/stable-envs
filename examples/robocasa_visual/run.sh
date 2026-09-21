#!/usr/bin/env bash
# The whole side-by-side pipeline, on one GPU node. Three stages, two interpreters, one job.
#
#   A  robocasa conda env   roll a real trajectory, keep the OFFICIAL frames    (mujoco 3.3.1)
#   B  ~/venvs/se3d         re-render the same qpos with MJWarp                 (mujoco 3.11.0)
#   C  robocasa conda env   compose video / gif / contact sheet, score fidelity (cv2 + ffmpeg)
#
# Stage A and stage C share an interpreter only because that is the one with robosuite AND
# imageio-ffmpeg; nothing in C needs robosuite. The MJCF stage A writes carries absolute asset paths
# and compiles standalone, which is what lets B run on a different MuJoCo.
#
#   bash examples/robocasa_visual/run.sh            # both kitchens, 200 steps, 256 px
#   bash examples/robocasa_visual/run.sh L52S03     # just the typical one
set -eu
REPO=/mnt/vast/home/randall.balestriero/code/stable-envs
WORK=/mnt/vast/home/randall.balestriero/robocasa_visual_work
RC=/mnt/vast/home/randall.balestriero/miniconda3/envs/robocasa/bin/python
SE3D=/mnt/vast/home/randall.balestriero/venvs/se3d/bin/python
if [[ $# -gt 0 ]]; then
  KITCHENS=("$@")
else
  KITCHENS=(L52S03 L45S33)
fi
STEPS=${STEPS:-200}
RES=${RES:-256}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WARP_CACHE_PATH=/mnt/vast/home/randall.balestriero/.cache/warp
mkdir -p "$WORK/logs"
cd "$REPO"

echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for K in "${KITCHENS[@]}"; do
  echo "--- A: trajectory + official frames, $K ---"
  "$RC" examples/robocasa_visual/dump_trajectory.py --kitchen "$K" --steps "$STEPS" --res "$RES" \
      --out "$WORK" 2>&1 | tail -5
  echo "--- B: MJWarp re-render, $K ---"
  "$SE3D" examples/robocasa_visual/render_mjwarp.py --kitchen "$K" --out "$WORK" \
      --max-worlds 64 --stages trajectory
done

echo "--- bridge: is a bare Renderer at group 1 the official renderer, on both MuJoCos? ---"
"$RC" examples/robocasa_visual/check_reference_bridge.py --kitchen "${KITCHENS[0]}" --tag mj331
"$SE3D" examples/robocasa_visual/check_reference_bridge.py --kitchen "${KITCHENS[0]}" --tag mj311

echo "--- B: throughput curve + grids (on ${KITCHENS[0]}) ---"
"$SE3D" examples/robocasa_visual/render_mjwarp.py --kitchen "${KITCHENS[0]}" --out "$WORK" \
    --stages throughput grids --batches 1 16 64 256 --grid 12 --kitchens 12

echo "--- C: video, gif, contact sheet, fidelity ---"
"$RC" examples/robocasa_visual/make_video.py --kitchens "${KITCHENS[@]}" --work "$WORK" \
    --out "$REPO/examples/robocasa_visual/media"
