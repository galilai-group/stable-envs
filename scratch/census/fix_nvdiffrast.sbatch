#!/bin/bash
#SBATCH --job-name=se-nvdr2
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_nvdr2_%j.out
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1
UV=/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/uv
VENV=/mnt/vast/home/randall.balestriero/venvs/se-all
PY=$VENV/bin/python
SRC=/mnt/vast/home/randall.balestriero/src
echo "### node=$(hostname)"

# nvdiffrast builds a CUDA extension and needs THREE things this environment supplies separately:
#   --no-build-isolation   its setup.py does `from torch.utils.cpp_extension import CUDAExtension`,
#                          and under isolation torch is not in the build env.
#   nvcc                   from the conda env; the torch cu13 wheels ship libraries and headers but
#                          no compiler.
#   CPATH                  the headers. The build failed on `fatal error: cusparse.h: No such file or
#                          directory` -- nvcc came from one place and the CUDA headers live in the
#                          venv's own nvidia/cu13 wheel, which nothing put on the include path.
NV=$VENV/lib/python3.11/site-packages/nvidia/cu13
export CPATH="$NV/include:${CPATH:-}"
export LIBRARY_PATH="$NV/lib:${LIBRARY_PATH:-}"
export CUDA_HOME=/mnt/vast/home/randall.balestriero/miniconda3/envs/test
export PATH="$CUDA_HOME/bin:$PATH"
echo "nvcc: $(command -v nvcc)"; echo "cusparse.h: $([ -f $NV/include/cusparse.h ] && echo $NV/include/cusparse.h)"

echo; echo "-- build --"
"$UV" pip install --python "$PY" --no-deps --no-build-isolation "$SRC/nvdiffrast" 2>&1 | grep -vE "^\s+\[|^\s+/mnt.*nvcc" | tail -25

echo; echo "-- the check mujoco_locomotion.py:1325 makes --"
"$PY" -c "
try:
    import nvdiffrast.torch as d; print('  import nvdiffrast.torch OK')
except Exception as e: print('  FAILS:', type(e).__name__, e)" 2>&1 | tail -4

echo; echo "-- registry --"
PYTHONPATH=. "$PY" - <<'EOF' 2>&1 | grep -vE "^Gym has|^Please upgrade|^Users of|^See the migration|WARNING:jax"
import json, stable_envs as se
print("REGISTRY:", len(se.REGISTRY), "of 196")
print("modules that FAILED:", {k: str(v)[:80] for k, v in se.envs.import_errors().items()})
cover = json.load(open("scratch/census/assignment.json"))
missing = sorted({e for es in cover.values() for e in es} - set(se.REGISTRY))
print(f"unreachable: {len(missing)} -> {missing}")
EOF
echo "### done"
