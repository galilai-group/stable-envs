#!/bin/bash
#SBATCH --job-name=se7-rocket-refreeze
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116,gpu2109,gpu2104,gpu2264
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 7 / W6 M6 -- the CORRECTNESS RE-FREEZE measurement for the batch-invariant mvp.
#
# Sami's ruling (2026-09-15): mandatory batch-size invariance takes precedence over accidental float
# bit identity; the six locomotion nvdiff cells are re-baselined, PROVIDED this job's evidence holds.
# ONE GPU job (per-user cap gpu=2 shared with the other Step-7 agents), in order:
#
#   0. context probes. EGL is a HARD requirement here and not a nice-to-have: the six cells' official
#      reference is MuJoCo through EGL, and "fidelity unchanged or improved" cannot be measured
#      without it. gpu2109 (job 6408817) has a broken mujoco.GLContext, so it is excluded along with
#      the four nodes the coordinator excludes and gpu2104. If EGL fails anyway the job exits 4
#      IMMEDIATELY, naming the node, so the next submission can exclude it too rather than produce a
#      run with the one number that matters missing.
#   1. refreeze_locomotion.py: baseline (76dc511) and branch legs on the SAME card, then the branch's
#      full mandatory set, then the comparison. Frames, official-reference fidelity, downsample bound.
#   2. step5_surfaces.py under a venv that has pymunk + gym_pusht + dm_control + mujoco_warp: the two
#      Step-5 representative surfaces (PushT/LeWM and DMCReacher*/DINO-WM), both legs, frozen Step-4
#      corpus, digests, and the nvdiff call counter that proves the changed code is not on their path.
#   3. PFRocketLanding: the seven gates again (measure_tier.py) plus the row-0 prefix probe.
#   4. the GPU conformance subset for the six cells AND the rocket, on the card.
#
# SYNC RECIPE (the rsync excludes .git, so the measured SHA must travel as a file):
#   git -C <worktree> rev-parse HEAD > <worktree>/.git_sha
#   rsync -az --delete --exclude .git --exclude .venv --exclude __pycache__ --exclude "*.pyc" \
#     <worktree>/ sboughan@sshcampus.ccv.brown.edu:/oscar/scratch/sboughan/se7/rocket-landing/
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export NUMBA_CACHE_DIR=$SCR/.cache/numba TORCH_EXTENSIONS_DIR=$SCR/.cache/torch_extensions
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=4
module load cuda/12.9.0 2>/dev/null || true
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
PY=$SCR/venvs/se7-rocket/bin/python          # nvdiffrast + mujoco 3.12 + gymnasium + PyFlyt
PY5=$SCR/venvs/se6-verify/bin/python         # pymunk + gym_pusht + dm_control + mujoco_warp (no nvdiffrast)
WS=$SCR/se7/rocket-landing
BASE=$SCR/se7/baseline
OUT=$WS/examples/results/step7/rocket-landing
RF=$OUT/refreeze
mkdir -p "$RF" "$OUT/tier" "$SCR/logs"
[ -f "$WS/.git_sha" ] || { echo "MISSING $WS/.git_sha: write it with 'git rev-parse HEAD > .git_sha' before the rsync (see SYNC RECIPE); refusing to measure a tree of unknown commit"; exit 3; }
NODE=$(hostname -s)
echo "NODE $NODE JOB $SLURM_JOB_ID GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) SHA $(cat "$WS/.git_sha") BASE_SHA $(cat "$BASE/.git_sha" 2>/dev/null || echo 76dc511-archive) LOAD $(cat /proc/loadavg | cut -d" " -f1-3)"
which nvcc; nvcc --version | tail -1

# ---------------------------------------------------------------- 0. context probes
# torch.cuda.set_device(0) BEFORE the context: nvdiffrast's RasterizeCudaContext resolves the bare
# string 'cuda' through torch.cuda._exchange_device, which needs a current device. Job 6409703 died
# here on gpu2264 with "invalid argument to exchangeDevice" -- an initialisation-order failure of the
# probe, not necessarily of the node, and the probe has to test the node. Full output kept: that job's
# `tail -3` hid everything above the last traceback line.
PYTHONPATH=$WS $PY -c "
import torch, nvdiffrast.torch as dr
torch.cuda.init(); torch.cuda.set_device(0)
torch.zeros(1, device='cuda:0')
dr.RasterizeCudaContext(device=torch.device('cuda:0'))
print('nvdiffrast ok', torch.cuda.get_device_name(0), 'devices', torch.cuda.device_count())
" 2>&1 | tail -20 | tee "$RF/plugin_build.txt"
grep -q "nvdiffrast ok" "$RF/plugin_build.txt" || { echo "NVDIFFRAST CONTEXT FAILED on $NODE"; exit 2; }
$PY -c "import mujoco; c = mujoco.GLContext(64, 64); c.make_current(); print('egl ok')" 2>&1 | tail -20 | tee "$RF/egl_probe.txt"
grep -q "egl ok" "$RF/egl_probe.txt" || {
  echo "EGL CONTEXT FAILED on $NODE -- the six cells' official reference cannot be rendered here, so"
  echo "the 'fidelity unchanged or improved' half of the re-freeze evidence would be missing."
  echo "RESUBMIT with --exclude=...,$NODE ; refusing to produce a partial re-freeze."
  exit 4; }
echo "EGL_OK on $NODE"
set -x

# ---------------------------------------------------------------- 1. the six locomotion cells
PYTHONPATH=$BASE $PY "$WS/scratch/step7/rocket-landing/refreeze_locomotion.py" --phase render --tag baseline --out "$RF" \
  2>&1 | grep -v "UserWarning\|sparse_coo_tensor\|to_sparse_csr\|  self.S4\|  (self.V"
PYTHONPATH=$WS   $PY "$WS/scratch/step7/rocket-landing/refreeze_locomotion.py" --phase render --tag branch   --out "$RF" \
  2>&1 | grep -v "UserWarning\|sparse_coo_tensor\|to_sparse_csr\|  self.S4\|  (self.V"
PYTHONPATH=$WS   $PY "$WS/scratch/step7/rocket-landing/refreeze_locomotion.py" --phase gatea --out "$RF" \
  2>&1 | grep -v "UserWarning\|sparse_coo_tensor\|to_sparse_csr\|  self.S4\|  (self.V"
PYTHONPATH=$WS   $PY "$WS/scratch/step7/rocket-landing/refreeze_locomotion.py" --phase compare --out "$RF" \
  2>&1 | tail -30
rm -f "$RF"/baseline_Mujoco*.npz "$RF"/branch_Mujoco*.npz      # digests + the compare table stay

# ---------------------------------------------------------------- 2. the Step-5 representative surfaces
PYTHONPATH=$BASE $PY5 "$WS/scratch/step7/rocket-landing/step5_surfaces.py" --phase render --tag baseline --out "$RF" 2>&1 | tail -12
PYTHONPATH=$WS   $PY5 "$WS/scratch/step7/rocket-landing/step5_surfaces.py" --phase render --tag branch   --out "$RF" 2>&1 | tail -12
PYTHONPATH=$WS   $PY5 "$WS/scratch/step7/rocket-landing/step5_surfaces.py" --phase compare --out "$RF" 2>&1 | tail -12

# ---------------------------------------------------------------- 3. PFRocketLanding: the seven gates again
PYTHONPATH=$WS $PY "$WS/scratch/step7/rocket-landing/measure_tier.py" --draws 4 --n 64 2>&1 | tee "$OUT/tier/measure_tier_stdout.txt" | tail -40
PYTHONPATH=$WS $PY "$WS/scratch/step7/rocket-landing/probe_batch_size.py" 2>&1 | grep -v "UserWarning\|sparse_coo_tensor\|to_sparse_csr"

# ---------------------------------------------------------------- 4. the GPU conformance subset on the card
PYTHONPATH=$WS $PY -m pytest tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py -q -rA -p no:cacheprovider \
  -k "MujocoHalfCheetah or MujocoHopper or MujocoWalker2d or MujocoAnt or MujocoHumanoid or MujocoSwimmer or PFRocketLanding" \
  > "$OUT/tier/conformance_six_plus_rocket.txt" 2>&1; tail -30 "$OUT/tier/conformance_six_plus_rocket.txt"

set +x
echo "===== REFREEZE COMPARE ====="; cat "$RF/compare_refreeze.json"
echo "===== GATE A (branch, six cells) ====="; $PY -c "import json;d=json.load(open('$RF/gatea_branch.json'));print('all_pass',d['all_pass']);[print(k,v.get('pass')) for k,v in d['cells'].items()]"
echo "===== STEP-5 SURFACES ====="; cat "$RF/step5_surfaces_compare.json"
echo "===== ROCKET GATES ====="; cat "$OUT/tier/gates.txt" 2>/dev/null
echo "S7REFREEZE done on $NODE"
