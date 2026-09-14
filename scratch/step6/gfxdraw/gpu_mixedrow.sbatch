#!/bin/bash
#SBATCH --job-name=se6-gfxdraw-mixedrow
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The mixed-row CUDA guard for the fused kernel.
#
# Until now the kernel's per-ENVIRONMENT paths -- the colour of each primitive, the aa flag, and
# MountainCar's baked plate at PLATE_B = B -- were covered only by the corpus-style digest in the
# Step 4 replay, which is one style built one way. test_the_kernel_is_the_description now carries a
# style axis, and asserts the batch really is mixed before comparing, so the case cannot decay into
# the default one with more steps.
#
# Also dumps the Box2D venv's manifest, because findings.md quotes a numba version.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export TRITON_CACHE_DIR=$SCR/.cache/triton
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
PY=$SCR/venvs/se-step6/bin/python
BOX=$SCR/venvs/se6-gfxdraw-box2d/bin/python
OUT=$SCR/se6/gfxdraw/examples/results/step6/gfxdraw
NODE=$(hostname -s)
mkdir -p "$OUT/tests" "$OUT/characterize"
cd "$SCR/se6/gfxdraw"
echo "NODE $NODE JOB $SLURM_JOB_ID"; nvidia-smi --query-gpu=name --format=csv,noheader
set -x

PYTHONPATH=$SCR/se6/gfxdraw $PY -m pytest tests/gpu/test_gfxdraw_classic.py -v --tb=short \
    > "$OUT/tests/classic_kernel_mixedrow_${NODE}.txt" 2>&1
echo "MIXED-ROW KERNEL TESTS rc=$?"
grep -E "PASSED|FAILED|ERROR|passed|failed" "$OUT/tests/classic_kernel_mixedrow_${NODE}.txt" | tail -20

# the paired R script again, now that peak_gb is reset per arm, so the committed number means
# what its name says
PYTHONPATH=$SCR/se6/gfxdraw $PY scratch/step6/gfxdraw/paired_gpu.py \
    --batch 4096 --rounds 5 --out "$OUT/sweep2/paired_gpu_B4096_perarmpeak_${NODE}.json"
echo "PAIRED GPU (per-arm peak) rc=$?"

export OUT
"$BOX" - <<'PYEOF'
import json, os, sys
from pathlib import Path
m = {"python": sys.version.split()[0], "prefix": sys.prefix,
     "note": "a venv layered on $SCR/venvs/se-step6 through a .pth; se-step6 itself is never written to"}
for mod in ("numba", "llvmlite", "torch", "pygame", "gymnasium", "numpy"):
    try:
        m[mod] = __import__(mod).__version__
    except Exception as e:
        m[mod] = f"absent: {type(e).__name__}"
try:
    import Box2D; m["Box2D"] = Box2D.__version__
except Exception as e:
    m["Box2D"] = f"absent: {type(e).__name__}"
p = Path(os.environ["OUT"]) / "characterize" / "venv_box2d_manifest.json"
p.write_text(json.dumps(m, indent=1))
print(json.dumps(m, indent=1))
PYEOF
echo "MIXED-ROW JOB DONE"
