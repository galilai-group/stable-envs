#!/bin/bash
#SBATCH --job-name=se7-dmc-quadruped-gpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 7 W4, the cuda side of DMCQuadrupedWalk, one card, one job (PENDING: written while Oscar SSH
# was denied; never run). Prerequisites the coordinator seeds: $SCR/se7/dmc-quadruped = rsync of the
# branch worktree (with GIT_COMMIT written by a sync.sh like scratch/step6/cpu-3d/sync.sh) and
# $SCR/se7/baseline = a pristine 76dc5116f432e61dd47fba8f9e879cf9212c8e46 checkout.
#  1. the RELEASE mujoco_warp draws ellipsoids / no sites / hfield path (grep, recorded);
#  2. cuda identity: the 21 raycast3d cells + 4 trackcom cells, branch vs baseline, one card
#     (replay_step4.py --dump-dir / --compare-dumps) -- the shared-file edit must move no cuda pixel;
#  3. the fused-vs-torch ellipsoid twin test and the fast-path suite on cuda;
#  4. the DECLARATION draws: visual_calib.py on DMCQuadrupedWalk, 3 draws x 64 states, under the
#     declared 3.11 pair (se-vis311) and, for the record, 3.12 (se-step5);
#  5. the cuda frame stack + Gate A (fidelity_step4.py) on the new frozen corpus under both arms;
#  6. the cross-backend oracle for the CPU job (a cuda stack on the corpus states).
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
WS=$SCR/se7/dmc-quadruped
BASE=$SCR/se7/baseline
PY5=$SCR/venvs/se-step5/bin/python            # mujoco 3.12 + mujoco_warp (record arm)
PY311=$SCR/venvs/se-vis311/bin/python         # mujoco 3.11 + mujoco-warp 3.11 (declared arm)
OUT=$WS/examples/results/step7/dmc-quadruped/phase1/gpu
STATES4=$WS/examples/results/step4/states
STATES7=$WS/examples/results/step7/dmc-quadruped/phase1/corpus/states
D=$SCR/se7/_dumps/quad_gpu
export STABLE_ENVS_GIT_COMMIT=${STABLE_ENVS_GIT_COMMIT:-$(cat "$WS/GIT_COMMIT" 2>/dev/null || true)}
mkdir -p "$OUT" "$D"; cd "$WS"
echo "NODE $(hostname -s) JOB $SLURM_JOB_ID commit $STABLE_ENVS_GIT_COMMIT"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee "$OUT/card.txt"
set -x
# 1. the release source
for V in se-step5 se-vis311; do
  SRC=$SCR/venvs/$V/lib/python3.12/site-packages/mujoco_warp/_src
  { echo "== $V $($SCR/venvs/$V/bin/python -c 'import importlib.metadata as m; print(m.version("mujoco-warp"))')";
    grep -n "ELLIPSOID\|HFIELD" "$SRC/render.py" | head -20; grep -n "def ray_ellipsoid\|def ray_hfield" "$SRC/ray.py";
    echo "site refs in render.py: $(grep -c 'site_' "$SRC/render.py")"; } > "$OUT/mujoco_warp_source_$V.txt" 2>&1
done
# 2. cuda identity, branch vs baseline, 25 configs
ALL="AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCFingerSpin DMCFingerTurnEasy DMCFingerTurnHard DMCPendulumSwingup DMCReacherEasy DMCReacherHard DMCCheetahRun DMCHopperHop DMCWalkerWalk DMCHumanoidStand"
PYTHONPATH=$BASE $PY5 scratch/step6/raycast3d/replay_step4.py --device cuda --configs $ALL --dump-dir "$D/base" --out "$OUT/cuda_baseline.json" || echo "BASE rc=$?"
PYTHONPATH=$WS   $PY5 scratch/step6/raycast3d/replay_step4.py --device cuda --configs $ALL --dump-dir "$D/branch" --out "$OUT/cuda_branch.json" || echo "BRANCH rc=$?"
PYTHONPATH=$WS   $PY5 scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/base" "$D/branch" --out "$OUT/cuda_branch_vs_baseline_identity.json" || echo "IDENT rc=$?"
# 3. the twins and the fast paths
export PYTHONPATH=$WS
$PY5 -m pytest tests/gpu/test_fast_paths.py -q -p no:randomly -k "ellipsoid or unknown_geom or raycast3d" 2>&1 | tail -8 | tee "$OUT/fast_paths.txt"
# 4. the declaration draws (3 x 64), both arms; the 3.11 arm is the one the entry is declared from
$PY311 scratch/dmcontrol/visual_calib.py --env DMCQuadrupedWalk --n 64 --draws 3 --size 224 --out "$OUT/visual_calib_311" 2>&1 | tail -6
$PY5   scratch/dmcontrol/visual_calib.py --env DMCQuadrupedWalk --n 64 --draws 3 --size 224 --out "$OUT/visual_calib_312" 2>&1 | tail -6
$PY311 scratch/dmcontrol/visual_calib.py --env DMCQuadrupedWalk --n 64 --draws 3 --size 84  --out "$OUT/visual_calib_311_84px" 2>&1 | tail -6
# 5. Gate A on cuda on the new corpus, both arms
$PY311 examples/fidelity_step4.py --configs DMCQuadrupedWalk --device cuda --states-dir "$STATES7" --out "$OUT/gateA_cuda_mj3110.json" 2>&1 | tail -12
$PY5   examples/fidelity_step4.py --configs DMCQuadrupedWalk --device cuda --states-dir "$STATES7" --out "$OUT/gateA_cuda_mj3120.json" 2>&1 | tail -12
# 6. the cuda stack on the corpus states, per surface vs official, both arms (texture-fidelity's
#    localize.py: writes frames/DMCQuadrupedWalk.npz = ours + official + segmentation, the
#    cross-backend oracle for certify_cpu.sbatch)
$PY311 scratch/step6/texture-fidelity/localize.py --configs DMCQuadrupedWalk --states-dir "$STATES7" --out "$OUT/localize_311" 2>&1 | tail -4 || echo "LOC311 rc=$?"
$PY5   scratch/step6/texture-fidelity/localize.py --configs DMCQuadrupedWalk --states-dir "$STATES7" --out "$OUT/localize_312" 2>&1 | tail -4 || echo "LOC312 rc=$?"
set +x; echo "GPU DECLARE DONE"
