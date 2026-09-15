#!/bin/bash
#SBATCH --job-name=se7-dmc-quadruped-gpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 7 W4, the cuda side of DMCQuadrupedWalk, one card, one job. $SCR/se7/dmc-quadruped is an rsync
# of the branch worktree (GIT_COMMIT written beside it), $SCR/se7/baseline a pristine 76dc511 tree.
#  1. the RELEASE mujoco_warp draws ellipsoids / no sites / hfield path (grep, recorded);
#  2. cuda identity, branch vs baseline, one card: the 21 raycast3d cells under se-step6 (the venv
#     with ogbench + triton) and the 4 trackcom cells under se-step5 (the venv with mujoco_warp) --
#     the shared-file edit must move no cuda pixel;
#  3. the fused-vs-torch ellipsoid twin test and the fast-path suite on cuda (se-step6);
#  4. the DECLARATION draws: visual_calib.py on DMCQuadrupedWalk, 3 draws x 64 states at 224 px under
#     the declared 3.11 pair (se-vis311) and, for the record, 3.12 (se-step5); 84 px supplementary;
#  5. Gate A on cuda on the new corpus, both arms, through gateA_local.py (the frozen Step-4 universe
#     does not carry the cell, so fidelity_step4.py's driver would run 0 configs);
#  6. localize.py stacks (ours + official + segmentation) on the corpus: the cross-backend oracle.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
WS=$SCR/se7/dmc-quadruped
BASE=$SCR/se7/baseline
PY5=$SCR/venvs/se-step5/bin/python            # mujoco 3.12 + mujoco_warp (record arm)
PY6=$SCR/venvs/se-step6/bin/python            # mujoco 3.12 + ogbench + triton, no mujoco_warp
PY311=$SCR/venvs/se-vis311/bin/python         # mujoco 3.11 + mujoco-warp 3.11 (declared arm)
OUT=$WS/examples/results/step7/dmc-quadruped/phase1/gpu
STATES7=$WS/examples/results/step7/dmc-quadruped/phase1/corpus/states
D=$SCR/se7/_dumps/quad_gpu
export STABLE_ENVS_GIT_COMMIT=${STABLE_ENVS_GIT_COMMIT:-$(cat "$WS/GIT_COMMIT" 2>/dev/null || true)}
mkdir -p "$OUT" "$D"; cd "$WS"
NODE=$(hostname -s)
echo "NODE $NODE JOB $SLURM_JOB_ID commit $STABLE_ENVS_GIT_COMMIT merged $(cat "$WS/MERGED_COMMIT" 2>/dev/null || echo ?)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee "$OUT/card.txt"
{ echo "job $SLURM_JOB_ID node $NODE"; echo "GIT_COMMIT $STABLE_ENVS_GIT_COMMIT"; echo "MERGED_COMMIT $(cat "$WS/MERGED_COMMIT" 2>/dev/null)"; date -u; } > "$OUT/provenance.txt"
set -x
# 1. the release source
for V in se-step5 se-vis311; do
  SRC=$SCR/venvs/$V/lib/python3.12/site-packages/mujoco_warp/_src
  { echo "== $V $($SCR/venvs/$V/bin/python -c 'import importlib.metadata as m; print(m.version("mujoco-warp"))')";
    grep -n "ELLIPSOID\|HFIELD" "$SRC/render.py" | head -20; grep -n "def ray_ellipsoid\|def ray_hfield" "$SRC/ray.py";
    echo "site refs in render.py: $(grep -c 'site_' "$SRC/render.py")"; } > "$OUT/mujoco_warp_source_$V.txt" 2>&1
done
# 2. cuda identity, branch vs baseline
RAY="AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCFingerSpin DMCFingerTurnEasy DMCFingerTurnHard DMCPendulumSwingup DMCReacherEasy DMCReacherHard"
TC="DMCCheetahRun DMCHopperHop DMCWalkerWalk DMCHumanoidStand"
PYTHONPATH=$BASE $PY6 scratch/step6/raycast3d/replay_step4.py --device cuda --configs $RAY --dump-dir "$D/base_ray" --out "$OUT/cuda_baseline_raycast.json" || echo "BASE-RAY rc=$?"
PYTHONPATH=$WS   $PY6 scratch/step6/raycast3d/replay_step4.py --device cuda --configs $RAY --dump-dir "$D/branch_ray" --out "$OUT/cuda_branch_raycast.json" || echo "BRANCH-RAY rc=$?"
PYTHONPATH=$WS   $PY6 scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/base_ray" "$D/branch_ray" --out "$OUT/cuda_identity_raycast21.json" || echo "IDENT-RAY rc=$?"
PYTHONPATH=$BASE $PY5 scratch/step6/raycast3d/replay_step4.py --device cuda --configs $TC --dump-dir "$D/base_tc" --out "$OUT/cuda_baseline_trackcom.json" || echo "BASE-TC rc=$?"
PYTHONPATH=$WS   $PY5 scratch/step6/raycast3d/replay_step4.py --device cuda --configs $TC --dump-dir "$D/branch_tc" --out "$OUT/cuda_branch_trackcom.json" || echo "BRANCH-TC rc=$?"
PYTHONPATH=$WS   $PY5 scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/base_tc" "$D/branch_tc" --out "$OUT/cuda_identity_trackcom4.json" || echo "IDENT-TC rc=$?"
export PYTHONPATH=$WS
# 3. the twins and the fast paths (triton lives in se-step6)
$PY6 -m pytest tests/gpu/test_fast_paths.py -q -p no:randomly -k "ellipsoid or unknown_geom" 2>&1 | tail -6 | tee "$OUT/fast_paths_ellipsoid.txt"
$PY6 -m pytest tests/gpu/test_fast_paths.py -q -p no:randomly -k "raycast3d" 2>&1 | tail -6 | tee "$OUT/fast_paths_raycast3d.txt"
# 4. the declaration draws (3 x 64), both arms; the 3.11 arm is the one the entry is declared from
$PY311 scratch/dmcontrol/visual_calib.py --env DMCQuadrupedWalk --n 64 --draws 3 --size 224 --out "$OUT/visual_calib_311" 2>&1 | tail -8
$PY5   scratch/dmcontrol/visual_calib.py --env DMCQuadrupedWalk --n 64 --draws 3 --size 224 --out "$OUT/visual_calib_312" 2>&1 | tail -8
$PY311 scratch/dmcontrol/visual_calib.py --env DMCQuadrupedWalk --n 64 --draws 3 --size 84  --out "$OUT/visual_calib_311_84px" 2>&1 | tail -8
# 5. Gate A on cuda on the new corpus, both arms
$PY311 scratch/step7/dmc-quadruped/gateA_local.py --config DMCQuadrupedWalk --device cuda --states-dir "$STATES7" --out "$OUT/gateA_cuda_mj3110.json" 2>&1 | tail -24
$PY5   scratch/step7/dmc-quadruped/gateA_local.py --config DMCQuadrupedWalk --device cuda --states-dir "$STATES7" --out "$OUT/gateA_cuda_mj3120.json" 2>&1 | tail -24
# 6. the cuda stack on the corpus states, per surface vs official, both arms
$PY311 scratch/step6/texture-fidelity/localize.py --configs DMCQuadrupedWalk --states-dir "$STATES7" --out "$OUT/localize_311" 2>&1 | tail -4 || echo "LOC311 rc=$?"
$PY5   scratch/step6/texture-fidelity/localize.py --configs DMCQuadrupedWalk --states-dir "$STATES7" --out "$OUT/localize_312" 2>&1 | tail -4 || echo "LOC312 rc=$?"
set +x; echo "GPU DECLARE DONE"
