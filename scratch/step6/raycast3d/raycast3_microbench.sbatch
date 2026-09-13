#!/bin/bash
#SBATCH --job-name=se6-raycast3d-mb
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# RAYCAST-3, design-first: the ns-per-ray-primitive-test floor of a fused CPU cast, on the
# certification host, at ONE thread. No renderer edits; the kernel is a scratch extension.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export TORCH_EXTENSIONS_DIR=$SCR/.cache/torch_extensions
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/raycast3d
export PYTHONPATH=$WS
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d
set -x
cd "$WS"
lscpu | grep -E "^Model name|^CPU\(s\)|Thread|MHz" | head -5
gcc --version | head -1

$PY scratch/step6/raycast3d/cast_microbench.py --threads 1 --rays 7056 --repeat 9 --warmup 3 \
    --out "$OUT/raycast3_cast_microbench.json" || echo "MB rc=$?"

# and at the OGBench window size, where the other two cells live
$PY scratch/step6/raycast3d/cast_microbench.py --threads 1 --rays 4096 --repeat 9 --warmup 3 \
    --cells AntmazeMediumExplore PointmazeLargeNavigate \
    --out "$OUT/raycast3_cast_microbench_224.json" || echo "MB224 rc=$?"
echo "MICROBENCH done"

# Second build, WITHOUT -march=native, to attribute the capsule-normal ulp: same source, same flags
# minus the ISA. If it becomes bit-exact the difference is codegen; if it does not, the transcription
# is wrong and no packaging fixes it.
MB_NO_MARCH=1 $PY scratch/step6/raycast3d/cast_microbench.py --threads 1 --rays 7056 --repeat 9 \
    --warmup 3 --out "$OUT/raycast3_cast_microbench_nomarch.json" || echo "MBNM rc=$?"
echo "MICROBENCH_NOMARCH done"
