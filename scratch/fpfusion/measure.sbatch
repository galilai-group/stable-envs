#!/bin/bash
#SBATCH --job-name=se-fpfusion
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_fpfusion_%j.out
set -uo pipefail
R=/mnt/vast/home/randall.balestriero/code/stable-envs
cd $R; export PYTHONPATH=$R MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1
echo "### node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
/mnt/vast/home/randall.balestriero/venvs/se-all/bin/python - <<'EOF'
import importlib.util, time, torch, numpy as np, sys
import stable_envs as se
from stable_envs.render import raycast3d as EXACT

spec = importlib.util.spec_from_file_location("raycast3d_fast", "scratch/fpfusion/raycast3d_fast.py")
FAST = importlib.util.module_from_spec(spec); sys.modules["raycast3d_fast"] = FAST
spec.loader.exec_module(FAST)
print("exact triton?", EXACT.HAVE_TRITON, "| fast triton?", FAST.HAVE_TRITON)

# Real scene geometry, not synthetic: take it from the env whose parity claim this protects.
env = se.make("PointmazeMediumNavigate", batch=64, device="cuda")
print("env:", type(env).__name__, "size", env.size)

# Drive the cast kernel directly with the env's own scene + a real ray bundle.
rng = np.random.default_rng(7)
states = env.sample_states(64, rng=rng)
env.set_state(states)
scene = env._scene3d() if hasattr(env, "_scene3d") else None
if scene is None:
    print("no direct scene accessor; falling back to END-TO-END render timing only")

def bench(fn, *a, reps=30, warm=5):
    for _ in range(warm): fn(*a)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps): fn(*a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps

# END TO END through the env, swapping the module the env calls.
import stable_envs.render as RND
def render_with(mod):
    orig = RND.raycast3d
    RND.raycast3d = mod
    try:
        for name in ("cast_nearest", "any_blocking"):
            setattr(EXACT, "_swapped_" + name, getattr(mod, name))
        return None
    finally:
        RND.raycast3d = orig

print("\n=== end-to-end env.render, exact kernel (shipped) ===")
t_exact = bench(lambda: env.render(states))
print(f"  {1/t_exact*64:,.0f} f/s   ({t_exact*1e3:.3f} ms / batch 64)")

# Monkeypatch the two entry points the renderer calls, then re-time.
EXACT_cast, EXACT_any = EXACT.cast_nearest, EXACT.any_blocking
EXACT.cast_nearest, EXACT.any_blocking = FAST.cast_nearest, FAST.any_blocking
torch.cuda.synchronize()
print("=== end-to-end env.render, FAST kernel (fusion on, div/sqrt approx) ===")
frames_fast = env.render(states).clone()
t_fast = bench(lambda: env.render(states))
print(f"  {1/t_fast*64:,.0f} f/s   ({t_fast*1e3:.3f} ms / batch 64)")
EXACT.cast_nearest, EXACT.any_blocking = EXACT_cast, EXACT_any
frames_exact = env.render(states).clone()

print(f"\n  SPEEDUP from dropping exactness: {t_exact/t_fast:.3f}x")
d = (frames_fast.to(torch.int16) - frames_exact.to(torch.int16)).abs()
npix = int((d > 0).any(-1).sum())
print(f"  PIXELS CHANGED: {npix:,} of {d.shape[0]*d.shape[1]*d.shape[2]:,} "
      f"({100*npix/(d.shape[0]*d.shape[1]*d.shape[2]):.4f}%), max |diff| {int(d.max())}/255")
print(f"  parity_interior_min declared by this env: {getattr(type(env),'parity_interior_min',None)}")
EOF
echo "### done"
