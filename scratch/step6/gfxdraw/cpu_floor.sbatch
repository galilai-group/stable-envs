#!/bin/bash
#SBATCH --job-name=se6-gfxdraw-cpufloor
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# GFXDRAW-2-CPU: the candidate mechanisms, on the certification host, at one thread.
#
# The question is whether cores_cpu >= 1 (one core of JPEG-q90 decode of the env's own frame) is
# reachable at all for these frames, and by what. Four candidates on one frame: torch as shipped,
# the torch/numpy interleave that is the floor of any planar design, a compiled scalar C++ walk of
# the same op list, and the same walk under numba. The C++ and numba forms have their checksum
# compared against the torch reference before any number is quoted.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip NUMBA_CACHE_DIR=$SCR/.cache/numba
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=$SCR/venvs/se-step6/bin/python
NB=$SCR/venvs/se6-gfxdraw-box2d          # has pip; se-step6 does not
OUT=$SCR/se6/gfxdraw/examples/results/step6/gfxdraw/characterize
NODE=$(hostname -s)
mkdir -p "$OUT"
cd "$SCR/se6/gfxdraw"
echo "NODE $NODE JOB $SLURM_JOB_ID"; uptime; c++ --version | head -1; lscpu | grep -E "Model name|Flags" | cut -c1-200
set -x

# the torch/numpy interleave band, on THIS host
PYTHONPATH=$SCR/se6/gfxdraw $PY scratch/step6/gfxdraw/interleave_bench.py \
    --shape 64,400,600 --out "$OUT/interleave_${NODE}.json"
PYTHONPATH=$SCR/se6/gfxdraw $PY scratch/step6/gfxdraw/interleave_bench.py \
    --shape 64,500,500 --out "$OUT/interleave_500_${NODE}.json"

# the compiled floor, without numba
PYTHONPATH=$SCR/se6/gfxdraw $PY scratch/step6/gfxdraw/cost_floor.py \
    --env CartPole Acrobot Pendulum MountainCar --batch 8 --reps 5 \
    --out "$OUT/cost_floor_${NODE}.json"
echo "COST FLOOR rc=$?"

# and with it, if it installs under these pins at all -- which is itself the packaging answer
if [ -x "$NB/bin/pip" ]; then
  "$NB/bin/pip" install --quiet numba || echo "numba install rc=$?"
  PYTHONPATH=$SCR/se6/gfxdraw "$NB/bin/python" -c "import numba, torch; print('numba', numba.__version__, 'torch', torch.__version__)" \
    && PYTHONPATH=$SCR/se6/gfxdraw "$NB/bin/python" scratch/step6/gfxdraw/cost_floor.py \
        --env CartPole Acrobot --batch 8 --reps 5 \
        --out "$OUT/cost_floor_numba_${NODE}.json"
  echo "NUMBA FLOOR rc=$?"
fi

# the decode bar on this host, for the same frames, so the ratio is same-host
export OUT NODE
PYTHONPATH=$SCR/se6/gfxdraw $PY - <<'PYEOF'
import json, os, numpy as np, sys
sys.path.insert(0, "examples")
from bench_vs_decode import decode_baseline
import stable_envs as se
from pathlib import Path
out = {}
for name in ("CartPole", "Acrobot", "Pendulum", "MountainCar"):
    e = se.make(name, batch=1, device="cpu")
    st = np.load(f"examples/results/step4/states/{name}.npz")["states"][:1]
    f = np.asarray(e.render(st)[0].cpu(), np.uint8)
    out[name] = decode_baseline(f, 90, 40, Path(os.environ["SCR"]) / ".cache")
    print(name, round(out[name]["decode_fps_per_core"], 1), "decode f/s per core",
          "->", round(1000 / out[name]["decode_fps_per_core"], 3), "ms/frame budget at cores=1")
Path(os.environ["OUT"] + f"/decode_{os.environ['NODE']}.json").write_text(json.dumps(out, indent=1))
PYEOF
echo "CPU FLOOR JOB DONE"
