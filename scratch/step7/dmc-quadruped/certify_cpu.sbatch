#!/bin/bash
#SBATCH --job-name=se7-dmc-quadruped-cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 7 W4, the CPU side of DMCQuadrupedWalk on a batch node, 1 torch thread (PENDING: written while
# Oscar SSH was denied; never run). Same prerequisites as gpu_declare.sbatch.
#  1. CPU identity: the 21 raycast3d cells + 4 trackcom cells, branch vs baseline 76dc511 (the local
#     Apple-M5 run is in phase1/replay/; this repeats it on the certification family);
#  2. Gate A on CPU on the new corpus under the declared 3.11 pair (se-vis311) and 3.12 (se-step6);
#  3. cross-backend: CPU frames vs the cuda stack gpu_declare.sbatch wrote, per surface;
#  4. the paired Step-3-shaped sweep against the same-node official renderer (certify_shard_cpu3d.py),
#     which is where the tier comes from; the local fps in findings.md are not certification numbers;
#  5. r5_kind_cost for the ELLIPSOID kind -- the measurement that decides its _BOUNDED_KINDS entry.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export WS=$SCR/se7/dmc-quadruped
export BASE=$SCR/se7/baseline
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export STABLE_ENVS_GIT_COMMIT=${STABLE_ENVS_GIT_COMMIT:-$(cat "$WS/GIT_COMMIT" 2>/dev/null || true)}
PY6=$SCR/venvs/se-step6/bin/python            # mujoco 3.12 (record arm)
PY311=$SCR/venvs/se-vis311/bin/python         # mujoco 3.11 (declared arm)
OUT=$WS/examples/results/step7/dmc-quadruped/phase1/cpu_oscar
STATES7=$WS/examples/results/step7/dmc-quadruped/phase1/corpus/states
D=$SCR/se7/_dumps/quad_cpu
NODE=$(hostname -s)
mkdir -p "$OUT" "$D"; cd "$WS"
echo "NODE $NODE JOB $SLURM_JOB_ID commit $STABLE_ENVS_GIT_COMMIT load $(cat /proc/loadavg)"; grep -m1 'model name' /proc/cpuinfo
set -x
# 1. CPU identity, branch vs baseline, 25 configs
ALL="AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCFingerSpin DMCFingerTurnEasy DMCFingerTurnHard DMCPendulumSwingup DMCReacherEasy DMCReacherHard DMCCheetahRun DMCHopperHop DMCWalkerWalk DMCHumanoidStand"
PYTHONPATH=$BASE $PY6 scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL --dump-dir "$D/base" --out "$OUT/cpu_baseline.json" || echo "BASE rc=$?"
PYTHONPATH=$WS   $PY6 scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL --dump-dir "$D/branch" --out "$OUT/cpu_branch.json" || echo "BRANCH rc=$?"
PYTHONPATH=$WS   $PY6 scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/base" "$D/branch" --out "$OUT/cpu_branch_vs_baseline_identity_${NODE}.json" || echo "IDENT rc=$?"
export PYTHONPATH=$WS
# 2. Gate A on CPU, both arms (the 3.12 arm's reference-judged checks BLOCK by design)
$PY311 examples/fidelity_step4.py --configs DMCQuadrupedWalk --device cpu --states-dir "$STATES7" --out "$OUT/gateA_cpu_mj3110.json" 2>&1 | tail -12
$PY6   examples/fidelity_step4.py --configs DMCQuadrupedWalk --device cpu --states-dir "$STATES7" --out "$OUT/gateA_cpu_mj3120.json" 2>&1 | tail -12
# 3. cross-backend vs the cuda stack (from gpu_declare.sbatch), per surface
STACK=$WS/examples/results/step7/dmc-quadruped/phase1/gpu/localize_311/frames/DMCQuadrupedWalk.npz
if [ -f "$STACK" ]; then
  $PY311 scratch/step7/dmc-quadruped/cross_backend.py --config DMCQuadrupedWalk --frames "$STACK" \
      --states "$STATES7/DMCQuadrupedWalk.npz" --out "$OUT/cross_backend_${NODE}.json" 2>&1 | tail -6 || echo "CROSS rc=$?"
else echo "no cuda stack yet: run gpu_declare.sbatch first"; fi
# 4. the paired sweep (tier)
$PY6 scratch/step6/cpu-3d/certify_shard_cpu3d.py --device cpu --threads 1 --reps 10 --max-cell-s 60 --max-load 12 \
    --envs DMCQuadrupedWalk --out "$OUT/cpu_1thread_paired_${NODE}_${SLURM_JOB_ID}.json" 2>&1 | tail -10 || echo "SWEEP rc=$?"
# 5. the kind-cost measurement for ELLIPSOID (r5_kind_cost.py's KINDS table gains the kind; see its header)
$PY6 scratch/step6/raycast3d/r5_kind_cost.py --threads 1 --repeat 20 --out "$OUT/kind_cost_ellipsoid_${NODE}.json" 2>&1 | tail -6 || echo "KIND rc=$?"
# 6. the unit suite on the certification host
$PY6 -m pytest tests/unit -q -p no:randomly 2>&1 | tail -6
set +x; echo "CPU CERTIFY DONE"
