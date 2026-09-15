#!/bin/bash
#SBATCH --job-name=se7-dmc-quadruped-cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 7 W4, the CPU side of DMCQuadrupedWalk on a batch node, 1 torch thread. Submitted with
# --dependency=afterany:<gpu job> so step 3 finds the cuda stack gpu_declare.sbatch writes.
#  1. CPU identity: the 21 raycast3d cells + 4 trackcom cells, branch vs baseline 76dc511 (se-step6,
#     the venv with ogbench; the local Apple-M5 run covered the 14 DMC cells only);
#  2. Gate A on CPU on the new corpus under the declared 3.11 pair (se-vis311) and 3.12 (se-step6),
#     through gateA_local.py (the frozen universe does not carry the cell);
#  3. cross-backend: CPU frames vs the cuda stack, per surface (cross_backend.py, se-step6);
#  4. the paired Step-3-shaped sweep against the same-node official renderer (certify_shard_cpu3d.py),
#     where the tier comes from; the local fps in findings.md are not certification numbers;
#  5. r5_kind_cost with the ELLIPSOID row -- the measurement that decides its _BOUNDED_KINDS entry;
#  6. the unit suite on the certification host.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export WS=$SCR/se7/dmc-quadruped
export BASE=$SCR/se7/baseline
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export STABLE_ENVS_GIT_COMMIT=${STABLE_ENVS_GIT_COMMIT:-$(cat "$WS/GIT_COMMIT" 2>/dev/null || true)}
PY6=$SCR/venvs/se-step6/bin/python            # mujoco 3.12 + ogbench (record arm)
PY311=$SCR/venvs/se-vis311/bin/python         # mujoco 3.11 (declared arm)
OUT=$WS/examples/results/step7/dmc-quadruped/phase1/cpu_oscar
STATES7=$WS/examples/results/step7/dmc-quadruped/phase1/corpus/states
D=$SCR/se7/_dumps/quad_cpu
NODE=$(hostname -s)
mkdir -p "$OUT" "$D"; cd "$WS"
echo "NODE $NODE JOB $SLURM_JOB_ID commit $STABLE_ENVS_GIT_COMMIT merged $(cat "$WS/MERGED_COMMIT" 2>/dev/null || echo ?) load $(cat /proc/loadavg)"; grep -m1 'model name' /proc/cpuinfo
{ echo "job $SLURM_JOB_ID node $NODE"; grep -m1 'model name' /proc/cpuinfo; echo "GIT_COMMIT $STABLE_ENVS_GIT_COMMIT"; echo "MERGED_COMMIT $(cat "$WS/MERGED_COMMIT" 2>/dev/null)"; date -u; } > "$OUT/provenance_${NODE}.txt"
set -x
# 1. CPU identity, branch vs baseline, 25 configs
ALL="AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCFingerSpin DMCFingerTurnEasy DMCFingerTurnHard DMCPendulumSwingup DMCReacherEasy DMCReacherHard DMCCheetahRun DMCHopperHop DMCWalkerWalk DMCHumanoidStand"
PYTHONPATH=$BASE $PY6 scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL --dump-dir "$D/base" --out "$OUT/cpu_baseline.json" || echo "BASE rc=$?"
PYTHONPATH=$WS   $PY6 scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL --dump-dir "$D/branch" --out "$OUT/cpu_branch.json" || echo "BRANCH rc=$?"
PYTHONPATH=$WS   $PY6 scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/base" "$D/branch" --out "$OUT/cpu_identity25_${NODE}.json" || echo "IDENT rc=$?"
export PYTHONPATH=$WS
# 2. Gate A on CPU, both arms (the 3.12 arm's reference-judged checks are recorded, not judged)
$PY311 scratch/step7/dmc-quadruped/gateA_local.py --config DMCQuadrupedWalk --device cpu --states-dir "$STATES7" --out "$OUT/gateA_cpu_mj3110.json" 2>&1 | tail -24
$PY6   scratch/step7/dmc-quadruped/gateA_local.py --config DMCQuadrupedWalk --device cpu --states-dir "$STATES7" --out "$OUT/gateA_cpu_mj3120.json" 2>&1 | tail -24
# 3. cross-backend vs the cuda stack (from gpu_declare.sbatch), per surface
for ARM in 311 312; do
  STACK=$WS/examples/results/step7/dmc-quadruped/phase1/gpu/localize_${ARM}/frames/DMCQuadrupedWalk.npz
  if [ -f "$STACK" ]; then
    $PY6 scratch/step7/dmc-quadruped/cross_backend.py --config DMCQuadrupedWalk --frames "$STACK" \
        --states "$STATES7/DMCQuadrupedWalk.npz" --out "$OUT/cross_backend_${ARM}_${NODE}.json" 2>&1 | tail -6 || echo "CROSS$ARM rc=$?"
  else echo "no cuda stack for arm $ARM: gpu_declare.sbatch did not write it"; fi
done
# 4. the paired sweep (tier)
$PY6 scratch/step6/cpu-3d/certify_shard_cpu3d.py --device cpu --threads 1 --reps 10 --max-cell-s 60 --max-load 12 \
    --envs DMCQuadrupedWalk --out "$OUT/cpu_1thread_paired_${NODE}_${SLURM_JOB_ID}.json" 2>&1 | tail -10 || echo "SWEEP rc=$?"
# 5. the kind-cost measurement with the ELLIPSOID row (three runs, as r6_kind_cost was built)
for R in 1 2 3; do
  $PY6 scratch/step6/raycast3d/r5_kind_cost.py --threads 1 --repeat 20 --out "$OUT/kind_cost_ellipsoid_${NODE}_run$R.json" 2>&1 | tail -3 || echo "KIND$R rc=$?"
done
# 6. the unit suite on the certification host
$PY6 -m pytest tests/unit -q -p no:randomly --deselect tests/unit/test_examples.py 2>&1 | tail -6 | tee "$OUT/unit_tests_${NODE}.txt"
set +x; echo "CPU CERTIFY DONE"
