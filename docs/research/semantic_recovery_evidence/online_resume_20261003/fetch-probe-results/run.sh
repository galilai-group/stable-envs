#!/bin/bash
#SBATCH --job-name=fetch-sampling-probe
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=0-2%3
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/fetch-sampling-probe-v2-20261003/gate-%A_%a.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/fetch-sampling-probe-v2-20261003
FROZEN=/oscar/scratch/sboughan/se-semantic-recovery/fetch-source-pipeline-v1-20261003
PY=/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python
cd "$FROZEN/code"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
"$PY" - "$BASE" "$FROZEN" <<'VERIFY'
import hashlib,json,pathlib,sys
base,root=map(pathlib.Path,sys.argv[1:])
for rel,digest in json.loads((base/'SHA256SUMS.json').read_text()).items():assert hashlib.sha256((base/rel).read_bytes()).hexdigest()==digest,rel
for name,prefix in [('candidate_manifest.json','code'),('source_manifest.json','source')]:
    for rel,digest in json.loads((root/name).read_text())['files'].items():assert hashlib.sha256((root/prefix/rel).read_bytes()).hexdigest()==digest,rel
print('Frozen diagnostic and source verified',flush=True)
VERIFY
TASKS=(Reach Push Slide)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
nvidia-smi
set +e
env MUJOCO_GL=glfw PYOPENGL_PLATFORM=glx LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe CUDA_VISIBLE_DEVICES= xvfb-run -a -s '-screen 0 1024x768x24 -noreset' "$PY" "$BASE/probe.py" --task "$TASK" --device cpu --source-root "$FROZEN/source" --output "$BASE/results/$TASK/cpu"
CPU_RC=$?
env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$PY" "$BASE/probe.py" --task "$TASK" --device cuda --source-root "$FROZEN/source" --output "$BASE/results/$TASK/cuda"
GPU_RC=$?
test "$CPU_RC" -eq 0 && test "$GPU_RC" -eq 0
