#!/bin/bash
#SBATCH --job-name=se6-raycast3d-2bcpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# RAYCAST-2b on the device it targets: the CPU.
#
# The frozen Step-4 oracle is CUDA-ONLY -- every digest in it was taken on a 3090 -- so the CPU path
# has no frozen contract to replay against. What it does have is two properties that can be checked
# here and are the ones that matter:
#
#   1. BRANCH == BASELINE on the same frozen states, on this node, in this job. That is the
#      regression check, and it is the strongest statement available without a CPU oracle.
#   2. COVER_COMPACTION on == off. The compaction is admissible only because it is the identity, and
#      the CPU is where it changes the most work, so it is asserted here as well as on the GPU.
#
# Plus the two OUT-OF-CORPUS callers of R3.composite (maze2d's PointMaze*, mujoco_locomotion's
# Mujoco*) which have no frozen states at all and are held to before-vs-after on a seeded draw.
#
# CPU certification budget is ONE torch thread; --cpus-per-task=4 only gives the job room.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/raycast3d
BASE=$SCR/se6/baseline
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d
D=$SCR/se6/_dumps/2bcpu
set -x
cd "$WS"
rm -rf "$D"; mkdir -p "$D/base" "$D/branch" "$D/compact_on" "$D/compact_off" "$D/oob_base" "$D/oob_branch"

OGB="AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore \
HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate \
PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate"
OOB="PointMazeLarge PointMazeMedium PointMazeOpen PointMazeUMaze \
MujocoReacher MujocoPusher MujocoInvertedPendulum MujocoInvertedDoublePendulum"

echo "######## 1. branch vs baseline on CPU, frozen states"
PYTHONPATH=$BASE $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $OGB \
    --dump-dir "$D/base" --out "$OUT/replay/2b_cpu_baseline.json" || echo "BASE rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $OGB \
    --dump-dir "$D/branch" --out "$OUT/replay/2b_cpu_branch.json" || echo "BRANCH rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/base" "$D/branch" \
    --out "$OUT/replay/2b_cpu_branch_vs_baseline.json" || echo "DIFF rc=$?"

echo "######## 2. COVER_COMPACTION on == off, on CPU"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $OGB \
    --cover-compaction on  --dump-dir "$D/compact_on"  --out "$OUT/replay/2b_cpu_compact_on.json"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $OGB \
    --cover-compaction off --dump-dir "$D/compact_off" --out "$OUT/replay/2b_cpu_compact_off.json"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py \
    --compare-dumps "$D/compact_off" "$D/compact_on" \
    --out "$OUT/replay/2b_cpu_compaction_identity.json" || echo "IDENT rc=$?"

echo "######## 3. the out-of-corpus callers of R3.composite, seeded draw, before vs after"
PYTHONPATH=$BASE $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $OOB \
    --sample-states 20260913 --dump-dir "$D/oob_base" \
    --out "$OUT/replay/2b_cpu_oob_baseline.json" || echo "OOB_B rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $OOB \
    --sample-states 20260913 --dump-dir "$D/oob_branch" \
    --out "$OUT/replay/2b_cpu_oob_branch.json" || echo "OOB_A rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py \
    --compare-dumps "$D/oob_base" "$D/oob_branch" \
    --out "$OUT/replay/2b_cpu_oob_diff.json" || echo "OOB_D rc=$?"

echo "######## 4. what the conformance suite's own backend-agreement check says here"
PYTHONPATH=$WS $PY - <<'PROBE' | tee "$OUT/replay/2b_cpu_backend_agreement.txt"
import json, os
os.environ.setdefault("MUJOCO_GL", "egl")
import stable_envs as se
from stable_envs.testing import conformance as C
out = {}
for n in ("PointmazeLargeNavigate", "AntmazeMediumExplore", "HumanoidmazeMediumNavigate"):
    e = se.make(n, batch=4, device="cpu")
    out[n] = C.check_backend_agreement(e)
print(json.dumps(out, indent=2, default=str))
PROBE

echo "######## 5. the 1-thread CPU certification sweep, before and after"
for E in PointmazeLargeNavigate PointmazeMediumNavigate AntmazeMediumExplore HumanoidmazeMediumNavigate; do
  PYTHONPATH=$BASE $PY scratch/step6/raycast3d/profile_phases.py --env $E --device cpu --threads 1 \
      --batch 1 16 64 256 --repeat 5 --warmup 2 \
      --states-dir "$WS/examples/results/step4/states" \
      --out "$OUT/profile_after_2b/before_baseline" || echo "SB $E rc=$?"
  PYTHONPATH=$WS   $PY scratch/step6/raycast3d/profile_phases.py --env $E --device cpu --threads 1 \
      --batch 1 16 64 256 --repeat 5 --warmup 2 \
      --states-dir "$WS/examples/results/step4/states" \
      --out "$OUT/profile_after_2b/after" || echo "SA $E rc=$?"
done
echo "2B_CPU done"
