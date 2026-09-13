#!/bin/bash
#SBATCH --job-name=se-s6-prof
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 6 profile-before for the CPU render paths, ON OSCAR.
#
# WHY RE-BASELINE HERE. Step 3 measured 74 of its 1-thread CPU rows on the laptop. A Step 6
# "after" taken on Oscar and compared against a laptop "before" is section 12's different-machines
# violation wearing a hostname. So this re-measures the PRE-STEP-6 code on Oscar; the optimised
# measurement will run on the same partition, and the tier ratio will be computed against the
# official renderer on the same node.
#
# CPU partition, not the GPU one: these paths need no CUDA, and the Step 5 campaign is holding the
# GPUs. --cpus-per-task=4 gives the job room while the certification budget itself stays at ONE
# torch thread, which is what --threads 1 pins.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export PYTHONPATH=$SCR/stable-envs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=$SCR/venvs/se-step6/bin/python
OUT=$SCR/step6
set -x
cd "$SCR/stable-envs"

# Absolute --states-dir. The first attempt passed it relative, every env died on the same
# ValueError, and SLURM still reported COMPLETED -- 89 seconds, seventeen tracebacks, zero files.
run () {  # env, workstream
  $PY scratch/step6/profile_note.py --env "$1" --device cpu --threads 1 \
      --batch 1 16 64 256 --repeat 10 --warmup 3 --workstream "step6/$2" \
      --states-dir "$SCR/stable-envs/examples/results/step4/states" \
      --out "$OUT/$2/profile_before" || { echo "PROFILE $1 rc=$?"; FAILED=$((FAILED+1)); }
  DONE=$((DONE+1))
}
DONE=0; FAILED=0

for E in PushT SimplePointMaze TwoRoom TwoRoomTarget Piecewise PiecewiseZones; do run $E raster2d; done
for E in Acrobot CartPole MountainCar MountainCarContinuous Pendulum;      do run $E gfxdraw;  done
run CraftaxClassic tilemap
for E in PointmazeLargeNavigate PointmazeMediumNavigate DMCReacherHard DMCFingerSpin DMCBallInCupCatch; do
  run $E raycast3d
done
echo "PROFILE_CPU envs=$DONE failed=$FAILED"
ls "$OUT"/*/profile_before/*.json 2>/dev/null | wc -l
test "$FAILED" -eq 0
