#!/bin/bash
# Full tracebacks for the tests that had NEVER run before the matrix: the learned-decoder file (which
# only reaches a real checkpoint in an interpreter that has both mujoco_warp>=3.11 and a cached
# kitchen) and the distributed file (which needs 2 GPUs, and the single-GPU launcher never gave it any).
#SBATCH --job-name=se-diag
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_diag_%j.out
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1
PY=/mnt/vast/home/randall.balestriero/venvs/se-hb/bin/python

echo "### node=$(hostname) gpus=$(nvidia-smi -L | wc -l)"

echo; echo "######## what the fixture actually gets: shape of sample_states ########"
"$PY" - <<'EOF'
import numpy as np, stable_envs as se
learned=[n for n,c in se.REGISTRY.items() if getattr(c,"render_decoder","renderer")=="learned"]
print("learned classes:", learned)
for n in learned[:1]:
    e=se.make(n,batch=8,device="cuda")
    s=e.sample_states(4,rng=np.random.default_rng(5))
    print(f"{n}: state_dim={e.state_dim} size={e.size}")
    print(f"  sample_states(4) -> type={type(s).__name__} shape={getattr(s,'shape',None)} dtype={getattr(s,'dtype',None)}")
    print(f"  np.asarray(...).shape = {np.asarray(s).shape}")
EOF

echo; echo "######## learned-decoder file, FULL tracebacks ########"
"$PY" -u -m pytest tests/gpu/test_rendernet.py -rA -p no:randomly --tb=long -x --maxfail=8 2>&1 | tail -180

echo; echo "######## distributed file, FULL tracebacks ########"
"$PY" -u -m pytest tests/gpu/test_distributed.py -rA -p no:randomly --tb=long 2>&1 | tail -120

echo "### diag done"
