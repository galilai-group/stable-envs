#!/bin/bash
#SBATCH --job-name=se6-raster2d-swmnav-fuzz
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The differential fuzz, which the first verification job reported as run and did not run: a cuda
# generator with no device= raises in torch.randint, every seed died in the same place, and the job
# carried on to report the sweep. Plus the AFTER stage profile, to say where the time now is.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export TRITON_CACHE_DIR=$SCR/.cache/triton WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=raster2d; AFT=$SCR/se6/$WS; RES=$AFT/examples/results/step6/$WS/swmnav2
mkdir -p "$RES"; cd $AFT
echo "node=$(hostname) job=$SLURM_JOB_ID tree $AFT at $(cat $AFT/GIT_COMMIT 2>/dev/null || echo unknown)"
P5=$SCR/venvs/se-step5/bin/python
set -x
FAILED=0
for S in 0 1 2 3 4 5; do
  PYTHONPATH=$AFT $P5 scratch/step6/raster2d/navdot_fuzz.py --cases 200 --seed $S \
    || FAILED=$((FAILED+1))
done
PYTHONPATH=$AFT $P5 scratch/step6/raster2d/swmnav_gpu_profile.py --device cuda \
    --batch 256 1024 4096 --out "$RES/gpu_profile_after.json"
echo "SWMNAV_FUZZ_DONE failed=$FAILED"
test "$FAILED" -eq 0
