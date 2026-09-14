#!/bin/bash
#SBATCH --job-name=se6-gfxdraw-gpusweep
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# GFXDRAW-2 GPU: the mandatory batch sweep with the fused kernel on and off, one card, one process.
# Plus the frozen Step 4 replay under the kernel, because a throughput number from a renderer whose
# pixels have not been replayed is not a result.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export TRITON_CACHE_DIR=$SCR/.cache/triton
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
PY=$SCR/venvs/se-step6/bin/python
OUT=$SCR/se6/gfxdraw/examples/results/step6/gfxdraw
NODE=$(hostname -s)
mkdir -p "$OUT/sweep2" "$OUT/replay"
cd "$SCR/se6/gfxdraw"
echo "NODE $NODE JOB $SLURM_JOB_ID"; nvidia-smi --query-gpu=name --format=csv,noheader
set -x

# the replay first: a sweep of a renderer whose pixels moved is not worth taking
PYTHONPATH=$SCR/se6/gfxdraw $PY examples/fidelity_step4.py \
    --configs Acrobot,CartPole,MountainCar,MountainCarContinuous,Pendulum --device cuda \
    --out "$OUT/replay/step4_replay_fused_cuda.json"
echo "REPLAY FUSED rc=$?"
PYTHONPATH=$SCR/se6/gfxdraw $PY scratch/step6/gfxdraw/replay_digests.py --device cuda \
    --out "$OUT/replay/digests_fused_cuda_${NODE}.json"
echo "DIGESTS FUSED rc=$?"

PYTHONPATH=$SCR/se6/gfxdraw $PY scratch/step6/gfxdraw/sweep_fused.py \
    --device cuda --out "$OUT/sweep2"
echo "SWEEP rc=$?"
echo "GPU SWEEP JOB DONE"
