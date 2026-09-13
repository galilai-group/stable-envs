#!/bin/bash
#SBATCH --job-name=se6-gfxdraw-paired
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# S_cpu as a paired measurement, both arms, alternating with the reference. See paired_cpu.py for
# the reason: the batch node runs at a load average near its core count and the OFFICIAL renderer's
# own throughput moved 2.3x between two arms of one job.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=$SCR/venvs/se-step6/bin/python
OUT=$SCR/se6/gfxdraw/examples/results/step6/gfxdraw/sweep
NODE=$(hostname -s)
mkdir -p "$OUT"
echo "NODE $NODE JOB $SLURM_JOB_ID"; uptime
set -x
# The BEFORE arm runs the branch's copy of the harness against the baseline TREE. $SCR/se6/baseline
# is pristine at 89fa5ca, which still carries profile_note.py's NameError at the write step, so a
# profile-before launched from inside it measures everything and then writes nothing.
for E in Acrobot CartPole MountainCar MountainCarContinuous Pendulum; do
  cd "$SCR/se6/baseline"
  PYTHONPATH=$SCR/se6/baseline $PY $SCR/se6/gfxdraw/scratch/step6/profile_note.py \
      --env $E --device cpu --threads 1 --batch 1 16 64 256 --repeat 10 --warmup 3 \
      --workstream step6/gfxdraw \
      --states-dir "$SCR/se6/baseline/examples/results/step4/states" \
      --out "$SCR/se6/gfxdraw/examples/results/step6/gfxdraw/profile_before_samenode" \
      || echo "PROFILE BEFORE $E rc=$?"
  cd "$SCR/se6/gfxdraw"
  PYTHONPATH=$SCR/se6/gfxdraw $PY scratch/step6/profile_note.py \
      --env $E --device cpu --threads 1 --batch 1 16 64 256 --repeat 10 --warmup 3 \
      --workstream step6/gfxdraw \
      --states-dir "$SCR/se6/gfxdraw/examples/results/step4/states" \
      --out "$SCR/se6/gfxdraw/examples/results/step6/gfxdraw/profile_after_samenode" \
      || echo "PROFILE AFTER $E rc=$?"
done

for B in 16 64 256; do
  cd "$SCR/se6/baseline"
  PYTHONPATH=$SCR/se6/baseline $PY $SCR/se6/gfxdraw/scratch/step6/gfxdraw/paired_cpu.py \
      --arm before --batch $B --out "$OUT/paired_before_B${B}_${NODE}.json"
  cd "$SCR/se6/gfxdraw"
  PYTHONPATH=$SCR/se6/gfxdraw $PY scratch/step6/gfxdraw/paired_cpu.py \
      --arm after --batch $B --out "$OUT/paired_after_B${B}_${NODE}.json"
done
echo "PAIRED JOB DONE"
