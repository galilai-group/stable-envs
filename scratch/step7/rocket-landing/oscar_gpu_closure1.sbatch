#!/bin/bash
#SBATCH --job-name=se7-rocket-closure1
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116,gpu2109,gpu2104,gpu2264
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 7 / W6 -- CLOSURE-1: does the explicit pos contraction move any pixel on the card?
#
# The closure sweep found Scene.clip batch-size dependent on x86-64 CPU at widths 1 and 2 (one float32
# ulp on positions only). It is the sparse per-vertex stage, not the mvp: mvp_rows is bit-identical at
# every width on both architectures, and so are the normals and world positions. Scene._pos_rows now
# writes that four-term sum out in ascending column order, in the input dtype, which reproduces
# torch.sparse.mm's value bit for bit on arm64 while removing the kernel's freedom to reorder for a
# narrow batch.
#
# "Reproduces it on arm64" is not "reproduces it on cuda", and the six locomotion cells were re-frozen
# three days ago on exactly this arithmetic. So the question this job answers is one question:
#
#   do the six cells' 64-frame digests still equal the RE-FREEZE ARTIFACT (job 6410105, tree 726241e,
#   examples/results/step7/rocket-landing/refreeze/branch.json) byte for byte?
#
# If yes, the fix is free and CLOSURE-1 closes. If any digest moves, that is a SECOND re-freeze and
# needs Sami -- stop and report, do not re-baseline. Also re-measured here because they are cheap and
# they are what the cell certifies on: PFRocketLanding's check_batch_size_invariance and the whole
# conformance subset for the rocket, including test_visual_tier_matches_its_declared_fidelity.
#
# EGL is a hard gate: the render phase measures each cell's official-reference fidelity, which needs
# the MuJoCo reference. gpu2109 (EGL) and gpu2264 (nvdiffrast context) are excluded with the four the
# coordinator excludes and gpu2104.
#
# SYNC RECIPE: git -C <worktree> rev-parse HEAD > <worktree>/.git_sha, then the rsync that excludes .git.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export NUMBA_CACHE_DIR=$SCR/.cache/numba TORCH_EXTENSIONS_DIR=$SCR/.cache/torch_extensions
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=4
module load cuda/12.9.0 2>/dev/null || true
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
PY=$SCR/venvs/se7-rocket/bin/python
WS=$SCR/se7/rocket-landing
OUT=$WS/examples/results/step7/rocket-landing
RF=$OUT/refreeze
NEW=$OUT/closure1
mkdir -p "$NEW" "$OUT/tier" "$SCR/logs"
[ -f "$WS/.git_sha" ] || { echo "MISSING $WS/.git_sha (see SYNC RECIPE)"; exit 3; }
NODE=$(hostname -s)
echo "NODE $NODE JOB $SLURM_JOB_ID GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) SHA $(cat "$WS/.git_sha") LOAD $(cat /proc/loadavg | cut -d' ' -f1-3)"

PYTHONPATH=$WS $PY -c "
import torch, nvdiffrast.torch as dr
torch.cuda.init(); torch.cuda.set_device(0)
torch.zeros(1, device='cuda:0')
dr.RasterizeCudaContext(device=torch.device('cuda:0'))
print('nvdiffrast ok', torch.cuda.get_device_name(0))
" 2>&1 | tail -20 | grep -q "nvdiffrast ok" || { echo "NVDIFFRAST CONTEXT FAILED on $NODE"; exit 2; }
$PY -c "import mujoco; c = mujoco.GLContext(64, 64); c.make_current(); print('egl ok')" 2>&1 | tail -20 \
  | grep -q "egl ok" || { echo "EGL CONTEXT FAILED on $NODE; RESUBMIT with --exclude=...,$NODE"; exit 4; }
echo "EGL_OK on $NODE"
set -x

# 1. the CPU localisation probe, on this node's CPU as well (x86-64), for the record
PYTHONPATH=$WS OMP_NUM_THREADS=1 $PY "$WS/scratch/step7/rocket-landing/probe_clip_cpu_batch.py" \
  --out "$NEW/probe_clip_cpu_batch_gpunode.json" 2>&1 | tail -20

# 2. the six cells at the NEW tree, same protocol as the re-freeze render phase
PYTHONPATH=$WS $PY "$WS/scratch/step7/rocket-landing/refreeze_locomotion.py" --phase render --tag branch \
  --out "$NEW" 2>&1 | grep -v "UserWarning\|sparse_coo_tensor\|to_sparse_csr\|  self.S4\|  (self.V"
rm -f "$NEW"/branch_Mujoco*.npz

# 3. digests vs the re-freeze artifact -- the whole question
set +x
PYTHONPATH=$WS $PY - "$RF/branch.json" "$NEW/branch.json" "$NEW/digest_vs_refreeze.json" <<'PYEOF'
import json, sys
old, new, out = (json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3])
R = {"refreeze_job": old["provenance"].get("slurm_job"), "refreeze_sha": old["provenance"].get("sha"),
     "new_job": new["provenance"].get("slurm_job"), "new_sha": new["provenance"].get("sha"),
     "refreeze_gpu": old["provenance"]["gpu"], "new_gpu": new["provenance"]["gpu"], "cells": {}}
ok = True
for k, c in new["cells"].items():
    o = old["cells"].get(k, {})
    if "frames" not in c or "frames" not in o:
        R["cells"][k] = {"blocked": True}; ok = False; continue
    same = o["frames"]["sha256"] == c["frames"]["sha256"]
    row = {"identical": same, "sha256_refreeze": o["frames"]["sha256"],
           "sha256_new": c["frames"]["sha256"],
           "states_match": o["frames"]["states_sha256"] == c["frames"]["states_sha256"]}
    vo, vn = o.get("visual_fidelity", {}), c.get("visual_fidelity", {})
    if "mean_abs" in vo and "mean_abs" in vn:
        row["fidelity"] = {m: {"refreeze": vo[m], "new": vn[m], "delta": vn[m] - vo[m]}
                           for m in ("mean_abs", "within8_frac", "ssim")}
    R["cells"][k] = row
    ok &= same
    print(f"  {k:20s} identical={same} {c['frames']['sha256'][:16]}", flush=True)
R["all_identical"] = bool(ok)
json.dump(R, open(out, "w"), indent=1)
print("CLOSURE1_ALL_IDENTICAL", R["all_identical"], flush=True)
PYEOF
set -x

# 4. the rocket: the mandatory batch-size check and the full conformance subset at this tree
PYTHONPATH=$WS $PY -m pytest tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py -q -rA -p no:cacheprovider \
  -k "PFRocketLanding or MujocoHalfCheetah or MujocoHopper or MujocoWalker2d or MujocoAnt or MujocoHumanoid or MujocoSwimmer" \
  > "$OUT/tier/conformance_closure1.txt" 2>&1
set +x
echo "===== batch-size + declared fidelity ====="
grep -E "batch_size|visual_tier_matches" "$OUT/tier/conformance_closure1.txt" | grep -E "^(PASSED|FAILED)" | head -20
echo "===== failures ====="
grep -E "^FAILED" "$OUT/tier/conformance_closure1.txt" || echo "(none)"
echo "===== pytest line ====="
tail -3 "$OUT/tier/conformance_closure1.txt"
echo "===== digests ====="
cat "$NEW/digest_vs_refreeze.json"
echo "S7CLOSURE1 done on $NODE at $(cat "$WS/.git_sha")"
