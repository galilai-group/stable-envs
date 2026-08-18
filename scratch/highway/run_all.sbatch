#!/bin/bash
#SBATCH --job-name=se-highway
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-070,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/se_logs/highway_%j.out
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
P=/mnt/vast/home/randall.balestriero/venvs/se-envs/bin/python

echo "===== primitive limit of the fused kernel ====="
for n in 300 320 340 350 400; do
  if out=$($P scratch/highway/probe_prim_limit.py $n 2>&1); then
    echo "$out" | grep -E "^N=" || echo "N=$n NO OUTPUT"
  else
    echo "N=$n ABORTED (exit $?)"
  fi
done

echo
echo "===== parity, 64 states each ====="
for e in Highway Merge; do
  echo "--- $e"
  $P scratch/highway/calib.py $e 64 2>&1 | grep -vE "Deprecation|logger"
done
echo "--- Roundabout (unregistered; torch backend, 16 states)"
SE_FORCE_TORCH=1 $P - <<'EOF' 2>&1 | grep -vE "Deprecation|logger"
import os, sys, time
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
import numpy as np, torch
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
from stable_envs.envs.highway import Roundabout
from stable_envs.testing import conformance as C
env = Roundabout(batch=16, device="cuda", backend="torch")
st = env.sample_states(16)
ref = Roundabout.reference_env(size=env.size)
off = Roundabout.reference_render(ref, st)
ours = env.render(st).cpu().numpy()
print("primitives:", env.scene(torch.as_tensor(st, device="cuda")).n_prims)
print(C.pixel_agreement(ours, off))
EOF

echo
echo "===== full conformance suite ====="
$P -u -m pytest tests/ -q -rs --tb=short 2>&1 | tail -30

echo
echo "===== throughput ====="
$P scratch/highway/bench.py Highway,Merge 256,1024,4096,16384 2>&1 | grep -vE "Deprecation|logger"
echo DONE
