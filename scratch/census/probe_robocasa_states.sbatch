#!/bin/bash
#SBATCH --job-name=se-probe-rc
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_probe_rc_%j.out
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1
/mnt/vast/home/randall.balestriero/venvs/se-hb/bin/python - <<'EOF'
import numpy as np, stable_envs as se
from stable_envs.envs import robocasa as RC
for name in ("RoboCasaCounterToSink", "RoboCasaCounterToSinkCorrected"):
    if name not in se.REGISTRY: print(name, "NOT REGISTERED"); continue
    e = se.make(name, batch=8, device="cuda")
    pool = RC._yawed_pool(e.KITCHEN)
    live = e._live
    print(f"\n=== {name} ===")
    print(f"  state_dim   = {e.state_dim}")
    print(f"  pool        = shape {pool.shape} dtype {pool.dtype}")
    print(f"  _live       = type {type(live).__name__} "
          f"shape {getattr(live,'shape',None)} dtype {getattr(live,'dtype',None)}")
    print(f"  _live value = {np.asarray(live).ravel()[:12]} ... (n={np.asarray(live).size})")
    idx = np.random.default_rng(5).integers(0, len(pool), 4)
    print(f"  pool[idx]              -> {pool[idx].shape}")
    try: print(f"  pool[idx][:, _live]    -> {pool[idx][:, live].shape}")
    except Exception as ex: print(f"  pool[idx][:, _live]    -> RAISES {type(ex).__name__}: {ex}")
    s = e.sample_states(4, rng=np.random.default_rng(5))
    print(f"  sample_states(4)       -> {s.shape}   EXPECTED (4, {e.state_dim})")
    print(f"  get_state()            -> {tuple(e.get_state().shape)}")
    print(f"  VERDICT: {'BUG' if s.shape != (4, e.state_dim) else 'ok'}")
EOF
