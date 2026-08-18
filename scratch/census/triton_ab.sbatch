#!/bin/bash
#SBATCH --job-name=se-triton-ab
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_triton_ab_%j.out
set -uo pipefail
R=/mnt/vast/home/randall.balestriero/code/stable-envs
cd $R
# ABSOLUTE. `PYTHONPATH=.` plus a `cd examples` in the runner pointed the path at examples/ and both
# halves of run 1 died on `No module named 'stable_envs'` before measuring anything.
export PYTHONPATH=$R
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1
export SCRATCH=/tmp/se_ab_$SLURM_JOB_ID; mkdir -p $SCRATCH
OUT=$R/examples/results/triton_ab
rm -rf $OUT; mkdir -p $OUT
echo "### node=$(hostname)  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# ONE NODE, BACK TO BACK, same envs, same seeds: across nodes this would compare two machines as much
# as two Tritons. Chosen to span the render BACKENDS, which is where the Triton kernels are.
ENVS=PushT,MiniGridEmpty,PointmazeMediumNavigate,MetaWorldAssembly,CarRacing,DMCWalkerWalk

run () {
  local tag=$1 py=$2
  echo; echo "================ $tag ================"
  [ -x "$py" ] || { echo "no interpreter at $py"; return; }
  "$py" -c "import torch,triton;print(f'  torch {torch.__version__}  triton {triton.__version__}')" 2>&1 | tail -1
  "$py" -u $R/examples/bench_vs_decode.py --tag "$tag" --scratch "$SCRATCH" \
      --envs "$ENVS" --batches 256,1024,4096 --reps 15 --decode-frames 0 --cpu-frames 0 \
      --out "$OUT" 2>&1 | tail -20
}

run t360_torch211 /mnt/vast/home/randall.balestriero/venvs/se-hb/bin/python
run t371_torch213 /mnt/vast/home/randall.balestriero/venvs/se-all/bin/python

echo; echo "================ side by side (f/s, higher is better) ================"
/mnt/vast/home/randall.balestriero/venvs/se-hb/bin/python - "$OUT" <<'EOF'
import json, sys, glob, os
root = sys.argv[1]
def load(tag):
    p = os.path.join(root, f"bench_vs_decode_{tag}.json")
    if not os.path.exists(p): return None, p
    d = json.load(open(p))
    out = {}
    for name, e in (d.get("envs") or {}).items():
        if not isinstance(e, dict): continue
        out[name] = {r["batch"]: r["fps"] for r in (e.get("gpu") or []) if "fps" in r and "batch" in r}
    return (d, out), p
A, pa = load("t360_torch211"); B, pb = load("t371_torch213")
if not A or not B:
    print("  missing result file(s):", [q for q, r in ((pa, A), (pb, B)) if not r])
    print("  files present:", sorted(glob.glob(os.path.join(root, "*.json"))))
    raise SystemExit(0)
(da, a), (db, b) = A, B
print(f"  A: torch {da['torch']}   B: torch {db['torch']}   device {da['device']}")
print(f"\n  {'env':26s} {'batch':>6s} {'t3.6.0':>11s} {'t3.7.1':>11s} {'ratio':>7s}")
rat = []
for env in sorted(set(a) & set(b)):
    for bt in sorted(set(a[env]) & set(b[env])):
        x, y = a[env][bt], b[env][bt]
        rat.append(y / x)
        print(f"  {env:26s} {bt:6d} {x:11,.0f} {y:11,.0f} {y/x:6.2f}x")
if rat:
    rat.sort()
    import math
    geo = math.exp(sum(math.log(r) for r in rat) / len(rat))
    print(f"\n  {len(rat)} cells | geomean {geo:.3f}x | min {rat[0]:.3f}x | median "
          f"{rat[len(rat)//2]:.3f}x | max {rat[-1]:.3f}x")
    print("  >1 means the LATEST stack is faster.")
EOF
rm -rf $SCRATCH
echo "### ab done"
