#!/bin/bash
#SBATCH --job-name=se8-capability-plate-texels-4
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# Paired re-measure of the one default-style ratio outside noise in job 6460099 (Manipulator 0.72 at
# B=4096): three ALTERNATING baseline/branch rounds, same card, so the spread rule (<= 1.25x between
# rounds of the same arm) can say whether 0.72 was a shared-node artefact or a cost.
set -uo pipefail
SCR=/oscar/scratch/sboughan
export UV_CACHE_DIR=$SCR/cache/uv PIP_CACHE_DIR=$SCR/cache/pip XDG_CACHE_HOME=$SCR/cache/xdg
export TORCH_HOME=$SCR/cache/torch WARP_CACHE_PATH=$SCR/cache/warp HF_HOME=$SCR/cache/hf
export MUJOCO_GL=egl
WS=$SCR/se8/capability; BASE=$SCR/se8/baseline; OUT=$WS/examples/results/step8/capability/gpu; mkdir -p "$OUT"
PY=$SCR/venvs/se-step6/bin/python; $PY -c "import mujoco_warp" 2>/dev/null || PY=$SCR/venvs/se-vis311/bin/python
{ echo "host $(hostname)"; echo "job ${SLURM_JOB_ID:-none}"; echo "sha $(cat $WS/.git_sha)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; uptime; } | tee "$OUT/provenance4.txt"
for R in 1 2 3; do for T in "$BASE:before" "$WS:after"; do P=${T%%:*}; N=${T##*:}
( cd "$P" && PYTHONPATH="$P" $PY - <<'PY' >> "$OUT/throughput_rounds_$N.jsonl"
import json, time, numpy as np, torch
from stable_envs.core.env import make
for cell in ("DMCManipulatorBringBall", "DMCCartpoleBalance"):
    e = make(cell, batch=4096, size=84, device="cuda")
    st = e.sample_states(4096, np.random.default_rng(0))
    e.render(st); torch.cuda.synchronize()
    ts=[]
    for _ in range(3):
        t=time.perf_counter()
        for _ in range(5): e.render(st)
        torch.cuda.synchronize(); ts.append(5*4096/(time.perf_counter()-t))
    print(json.dumps({"cell": cell, "B": 4096, "style": "default", "fps_median": float(np.median(ts)), "fps": ts}))
    try: e.close()
    except Exception: pass
PY
) || echo "ROUND_${R}_${N} FAILED"; done; done
echo "ALLDONE4 $(date -Is)"
