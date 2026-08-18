#!/bin/bash
#SBATCH --job-name=se-counts
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_counts_%j.out
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1
/mnt/vast/home/randall.balestriero/venvs/se-all/bin/python - <<'EOF' 2>&1 | grep -vE "^Gym|^Please|^Users|^See the|WARNING"
import collections, stable_envs as se
R = se.REGISTRY
par = {n: c for n, c in R.items() if getattr(c, "fidelity_tier", "parity") == "parity"}
vis = {n: c for n, c in R.items() if getattr(c, "fidelity_tier", "parity") == "visual"}
lo  = {n: c for n, c in par.items() if (getattr(c, "parity_interior_min", 1.0) or 1.0) < 1.0}
print(f"registry {len(R)} | parity {len(par)} | visual {len(vis)}")
print(f"parity envs with floor < 1.0: {len(lo)}")
print("  by module:", collections.Counter(c.__module__.split('.')[-1] for c in lo.values()).most_common())
print("  distinct floors:", sorted({float(getattr(c,'parity_interior_min',1.0)) for c in lo.values()}))
ss2 = {n for n, c in R.items() if int(getattr(c, "default_supersample", 1)) == 2}
print(f"\nenvs with default_supersample == 2: {len(ss2)}")
print("  by module:", collections.Counter(R[n].__module__.split('.')[-1] for n in ss2).most_common())
EOF
echo "### counts done"
