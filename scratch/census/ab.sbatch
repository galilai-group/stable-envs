#!/bin/bash
#SBATCH --job-name=se_ab
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/census/ab_%j.log
R=/mnt/vast/home/randall.balestriero/code/stable-envs
cd $R
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
PY=/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python
T="tests/gpu/test_conformance.py::test_render_parity_with_official_renderer"
CASES="TwoRoomTarget TwoRoom Piecewise PiecewiseZones"

run10 () {
  for c in $CASES; do
    p=0; f=0
    for i in 1 2 3 4 5; do
      if $PY -m pytest "$T[$c]" -q --no-header -p no:cacheprovider >/tmp/ab_$$.txt 2>&1; then p=$((p+1)); else f=$((f+1)); fi
      grep -o "interior_exact_frac': [0-9.]*" /tmp/ab_$$.txt | tail -1
    done
    echo "  $c: $p passed / $f failed"
  done
}

echo "### WITH the device-guard edits (current working tree)"
run10
echo "### WITHOUT them"
$PY scratch/census/ab_device_guard.py off
run10
echo "### restoring"
$PY scratch/census/ab_device_guard.py on
