#!/bin/bash
#SBATCH --job-name=se6-gfxdraw-cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# gfxdraw CPU certification, BEFORE and AFTER in one job on one node.
#
# Both arms here on purpose. Step 3 measured 74 of its 1-thread CPU rows on the laptop and the
# gfxdraw profile-before landed on node2314; a Step 6 "after" on a different node compared against
# either of those is the different-machines violation wearing a hostname. So this runs the frozen
# benchmark twice on THIS node -- $SCR/se6/baseline for the pre-Step-6 code, $SCR/se6/gfxdraw for
# the branch -- and the official renderer and the decode bar are taken in the same process as each
# arm, which is what makes S_cpu and cores_cpu ratios rather than two unrelated numbers.
#
# --max-cell-s 12 and the default reps are Step 3's own settings (scratch/step3/cpu_bench.sbatch);
# the certification budget is ONE torch thread, pinned by --threads 1 and the OMP variables.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=$SCR/venvs/se-step6/bin/python
ENVS=Acrobot,CartPole,MountainCar,MountainCarContinuous,Pendulum
OUT=$SCR/se6/gfxdraw/examples/results/step6/gfxdraw
NODE=$(hostname -s)
mkdir -p "$OUT/sweep" "$OUT/profile_after" "$OUT/profile_before_samenode"
echo "NODE $NODE  JOB $SLURM_JOB_ID"
set -x

# ---- the frozen Step 3 sweep, both arms, same node, same process per arm -------------------
cd "$SCR/se6/baseline"
PYTHONPATH=$SCR/se6/baseline $PY examples/bench_step3.py --device cpu --threads 1 \
    --envs "$ENVS" --max-cell-s 12 \
    --out "$OUT/sweep/cpu_1thread_before_${NODE}.json" || echo "SWEEP BEFORE rc=$?"

cd "$SCR/se6/gfxdraw"
PYTHONPATH=$SCR/se6/gfxdraw $PY examples/bench_step3.py --device cpu --threads 1 \
    --envs "$ENVS" --max-cell-s 12 \
    --out "$OUT/sweep/cpu_1thread_after_${NODE}.json" || echo "SWEEP AFTER rc=$?"

# ---- section-8 profile notes, both arms, same node ------------------------------------------
for E in Acrobot CartPole MountainCar MountainCarContinuous Pendulum; do
  cd "$SCR/se6/baseline"
  PYTHONPATH=$SCR/se6/baseline $PY scratch/step6/profile_note.py --env $E --device cpu --threads 1 \
      --batch 1 16 64 256 --repeat 10 --warmup 3 --workstream step6/gfxdraw \
      --states-dir "$SCR/se6/baseline/examples/results/step4/states" \
      --out "$OUT/profile_before_samenode" || echo "PROFILE BEFORE $E rc=$?"
  cd "$SCR/se6/gfxdraw"
  PYTHONPATH=$SCR/se6/gfxdraw $PY scratch/step6/profile_note.py --env $E --device cpu --threads 1 \
      --batch 1 16 64 256 --repeat 10 --warmup 3 --workstream step6/gfxdraw \
      --states-dir "$SCR/se6/gfxdraw/examples/results/step4/states" \
      --out "$OUT/profile_after" || echo "PROFILE AFTER $E rc=$?"
done

# ---- the CPU-runnable part of the regression suite -------------------------------------------
cd "$SCR/se6/gfxdraw"
PYTHONPATH=$SCR/se6/gfxdraw $PY -m pytest tests/unit -q 2>&1 | tail -20
PYTHONPATH=$SCR/se6/gfxdraw $PY scratch/step6/gfxdraw/conformance_cpu.py
PYTHONPATH=$SCR/se6/gfxdraw $PY scratch/step6/gfxdraw/replay_digests.py \
    --out "$OUT/replay/digests_cpu_${NODE}.json"
echo "CPU JOB DONE"
