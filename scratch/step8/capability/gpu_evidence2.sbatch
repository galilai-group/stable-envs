#!/bin/bash
#SBATCH --job-name=se8-capability-plate-texels-2
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# Second T3 evidence job (the first, 6459781 on gpu3002/L40S, produced declaration_{before,after}.json
# and the three GPU bake-vs-recompile probes; its Gate-A step was a placeholder and its throughput
# snippet passed floor.color with the wrong shape). This job runs the missing merge-gate items:
# (4) Gate A on cuda with the new style_dim through the closure2 driver, (5) frozen-corpus digest
# replay before/after on the same card, (7) paired throughput with a correctly shaped floor.color.
set -uo pipefail
SCR=/oscar/scratch/sboughan
export UV_CACHE_DIR=$SCR/cache/uv PIP_CACHE_DIR=$SCR/cache/pip XDG_CACHE_HOME=$SCR/cache/xdg
export TORCH_HOME=$SCR/cache/torch WARP_CACHE_PATH=$SCR/cache/warp HF_HOME=$SCR/cache/hf
export MUJOCO_GL=egl
WS=$SCR/se8/capability; BASE=$SCR/se8/baseline; OUT=$WS/examples/results/step8/capability/gpu; mkdir -p "$OUT"
PY=$SCR/venvs/se-step6/bin/python; $PY -c "import mujoco_warp" 2>/dev/null || PY=$SCR/venvs/se-vis311/bin/python
PARITY=DMCAcrobotSwingup,DMCBallInCupCatch,DMCCartpoleBalance,DMCCartpoleSwingup,DMCFingerTurnHard,DMCPendulumSwingup,DMCReacherHard
ALL=$PARITY,DMCManipulatorBringBall
{ echo "host $(hostname)"; echo "job ${SLURM_JOB_ID:-none}"; echo "sha $(cat $WS/.git_sha)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; $PY -c "import torch,mujoco,mujoco_warp;print('torch',torch.__version__,'mujoco',mujoco.__version__,'mujoco_warp',mujoco_warp.__version__)"; } | tee "$OUT/provenance2.txt"
echo "=== (4) Gate A cuda, parity-7 on the frozen Step-4 corpora"
( cd "$WS" && PYTHONPATH="$WS" $PY scratch/step7/closure/gate_a_new_cells.py --cells "$PARITY" --device cuda --states-dir "$WS/examples/results/step4/states" --n-states 64 --pool 256 --out "$OUT/gate_a_cuda_dmc_parity.json" ) || echo "GATE_A_PARITY FAILED"
echo "=== (4) Gate A cuda, Manipulator on the Step-7 closure corpus"
( cd "$WS" && PYTHONPATH="$WS" $PY scratch/step7/closure/gate_a_new_cells.py --cells DMCManipulatorBringBall --device cuda --states-dir "$SCR/se7/closure_states" --n-states 64 --pool 256 --out "$OUT/gate_a_cuda_dmc_manipulator.json" ) || echo "GATE_A_MANIP FAILED"
echo "=== (5) frozen-corpus digest replay, baseline arm then branch arm, same card"
( cd "$BASE" && PYTHONPATH="$BASE" $PY "$WS/scratch/step8/provenance/digest_replay.py" --cells "$ALL" --device cuda --out "$OUT/digest_replay_before.json" ) || echo "REPLAY_BEFORE FAILED"
( cd "$WS" && PYTHONPATH="$WS" $PY scratch/step8/provenance/digest_replay.py --cells "$ALL" --device cuda --out "$OUT/digest_replay_after.json" ) || echo "REPLAY_AFTER FAILED"
echo "=== (7) paired throughput, baseline arm then branch arm"
for T in "$BASE:before" "$WS:after"; do P=${T%%:*}; N=${T##*:}
( cd "$P" && CELLS="$ALL" PYTHONPATH="$P" $PY - <<'PY' > "$OUT/throughput_$N.json"
import json, os, time, numpy as np, torch
from stable_envs.core.env import make
rows=[]
for cell in os.environ["CELLS"].split(","):
    for B in (1, 4096):
        e = make(cell, batch=B, size=84, device="cuda")
        st = e.sample_states(B, np.random.default_rng(0))
        names = list(e.variation_space.names()) if hasattr(e.variation_space, "names") else []
        has_floor = any(n.startswith("floor.color") for n in names)
        cases = [("default", None)]
        if has_floor:
            rng = np.random.default_rng(1)
            cases.append(("shared_floor", {"floor.color": rng.random(6)}))
            cases.append(("per_env_floor_96", {"floor.color": np.repeat(rng.random((96, 6)), int(np.ceil(B/96)), axis=0)[:B]}))
        for label, var in cases:
            try:
                e.render(st, variation=var); torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(5): e.render(st, variation=var)
                torch.cuda.synchronize()
                rows.append({"cell": cell, "B": B, "style": label, "fps": 5*B/(time.perf_counter()-t)})
            except Exception as ex:
                rows.append({"cell": cell, "B": B, "style": label, "error": repr(ex)[:300]})
        try: e.close()
        except Exception: pass
print(json.dumps(rows, indent=1))
PY
) || echo "THROUGHPUT_$N FAILED"; done
echo "ALLDONE2 $(date -Is)"
