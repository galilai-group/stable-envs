#!/bin/bash
#SBATCH --job-name=se6-tilemap-cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# TILEMAP — everything the CPU claim rests on, in ONE job on ONE node.
#
# The rule this script exists to obey: a before and an after taken on different nodes, or on the same
# node under different load, is not a before and an after. The batch partition carries load averages
# past 45, so the two legs are INTERLEAVED per batch size rather than run as two blocks, and every
# leg records the stable_envs it actually imported. PYTHONPATH is the only difference between them:
# $BASE is the pristine baseline checkout, $WS is this branch.
#
# The CPU certification budget is ONE torch thread (frozen Step 2, section 5.2); --cpus-per-task=4
# only gives the job room to breathe.
#
# Craftax's official renderer needs jax, which no Oscar Step-6 interpreter carries, so S_cpu is
# undefined here and the frozen NO_REFERENCE ladder applies (CPU_FAST at cores_cpu >= 3). That is not
# a choice made here: bench_step3 records the importer's error and certify.py selects the ladder.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/tilemap
BASE=$SCR/se6/baseline
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/tilemap
mkdir -p "$OUT/profile_before" "$OUT/profile_after" "$OUT/replay" "$OUT/sweep" "$SCR/logs"
cd "$WS"
echo "host=$(hostname) job=$SLURM_JOB_ID load=$(cat /proc/loadavg)"

echo "######## 1. phase attribution, interleaved (both legs run the BRANCH harness)"
for LEG in before after before after; do
  [ "$LEG" = before ] && TREE=$BASE DIR=profile_before || TREE=$WS DIR=profile_after
  PYTHONPATH=$TREE $PY scratch/step6/tilemap/attribute.py --device cpu --threads 1 \
    --batch 1 16 64 256 --repeat 15 --warmup 3 --label "$LEG" \
    --states "$WS/examples/results/step4/states/CraftaxClassic.npz" \
    --out "$OUT/$DIR" || echo "ATTR $LEG rc=$?"
done

echo "######## 2. operator-level profile note, interleaved"
for LEG in before after; do
  [ "$LEG" = before ] && TREE=$BASE DIR=profile_before || TREE=$WS DIR=profile_after
  PYTHONPATH=$TREE $PY scratch/step6/profile_note.py --env CraftaxClassic --device cpu \
    --threads 1 --batch 1 16 64 256 --repeat 15 --warmup 3 --workstream step6/tilemap \
    --states-dir "$WS/examples/results/step4/states" --out "$OUT/$DIR" || echo "NOTE $LEG rc=$?"
done

echo "######## 3. the frozen 1-thread CPU certification sweep, both legs, same node"
for LEG in before after; do
  [ "$LEG" = before ] && TREE=$BASE || TREE=$WS
  PYTHONPATH=$TREE $PY "$TREE/examples/bench_step3.py" --device cpu --threads 1 \
    --envs CraftaxClassic --out "$OUT/sweep/cpu_${LEG}.json" || echo "BENCH $LEG rc=$?"
done
$PY scratch/step6/tilemap/tier.py --device cpu --env CraftaxClassic \
  --before "$OUT/sweep/cpu_before.json" --after "$OUT/sweep/cpu_after.json" \
  --out "$OUT/sweep/tier_cpu.json" || echo "TIER rc=$?"

echo "######## 4. frozen Step-4 replay on the CPU, both legs (each tree runs its OWN harness:"
echo "########    the branch's fidelity_step4.py calls a conformance check the baseline lacks)"
PYTHONPATH=$BASE $PY "$BASE/examples/fidelity_step4.py" --configs CraftaxClassic --device cpu \
  --out "$OUT/replay/cpu_before.json" || echo "REPLAY before rc=$?"
PYTHONPATH=$WS $PY "$WS/examples/fidelity_step4.py" --configs CraftaxClassic --device cpu \
  --out "$OUT/replay/cpu_after.json" || echo "REPLAY after rc=$?"
$PY scratch/step6/tilemap/replay_check.py --replay "$OUT/replay/cpu_after.json" \
  --against "$OUT/replay/cpu_before.json" --out "$OUT/replay/cpu_before_vs_after.json"
echo "CPU before-vs-after rc=$?"
$PY scratch/step6/tilemap/replay_check.py --replay "$OUT/replay/cpu_after.json" \
  --out "$OUT/replay/cpu_vs_frozen.json"
echo "CPU vs frozen Step-4 oracle rc=$?"

echo "######## 5. the branch's unit suite"
PYTHONPATH=$WS $PY -m pytest tests/unit -q 2>&1 | tail -15

echo "TILEMAP_CPU_DONE"
