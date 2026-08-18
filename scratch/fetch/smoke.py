"""Smallest end-to-end check: build one env of each scene, render, and score against the reference."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
import numpy as np, torch
import stable_envs as se
from stable_envs.testing import conformance as C

print("import errors:", se.envs.import_errors())
names = sorted(n for n in se.REGISTRY if n.startswith("Fetch"))
print(len(names), names)
for n in ("FetchReach", "FetchPush", "FetchSlide", "FetchPickAndPlace"):
    env = se.make(n, batch=8, device="cuda")
    st = env.sample_states(16, rng=np.random.default_rng(0))
    f = env.render(st)
    print(n, "state_dim", env.state_dim, "frame", tuple(f.shape), f.dtype)
    ref = C.reference_env(type(env), env.size)
    off = type(env).reference_render(ref, st)
    r = C.pixel_agreement(f.cpu().numpy(), off)
    print("   parity", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    print("   determinism", C.check_determinism(env))
    print("   batch inv", C.check_batch_invariance(env))
    print("   dead dims", C.check_state_dims_matter(env))
    print("   inert vars", C.check_variations_do_something(env))
    print("   per-env var", C.check_per_env_variation(env))
    print("   style is state", C.check_style_is_state(env))
    print("   default style", C.check_default_style_is_official(env))
    sys.stdout.flush()
