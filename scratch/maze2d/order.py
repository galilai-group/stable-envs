import sys, hashlib, numpy as np, torch
sys.path.insert(0, ".")
import stable_envs as se
from stable_envs.testing import conformance as C, render_worker as W

def dig(a):
    a = np.ascontiguousarray(a if isinstance(a, np.ndarray) else a.detach().cpu().numpy())
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]

env = se.make("PointMazeUMaze", batch=8, device="cuda")
states, style = W.sample_inputs(env, 8)
base = dig(env.render(states, style=style))
print("baseline           ", base)

# 1. build the parity reference env (a second EGL context on the same maze) and render through it
ref = C.reference_env(type(env), env.size)
type(env).reference_render(ref, env.sample_states(4))
print("after reference_env", dig(env.render(states, style=style)), "plate", dig(env._scene().plate))

# 2. churn the plate cache with many distinct camera styles
for k in range(40):
    st2 = env.sample_style(8, rng=np.random.default_rng(k))
    env.render(states, style=st2)
print("after 40 styles    ", dig(env.render(states, style=style)), "plate", dig(env._scene().plate))

# 3. a different size
env.render(states[:2], size=2 * env.size)
print("after a 2S render  ", dig(env.render(states, style=style)), "plate", dig(env._scene().plate))

# 4. a fresh env of the same class
fresh = se.make("PointMazeUMaze", batch=8, device="cuda")
print("fresh env          ", dig(fresh.render(states, style=style)), "plate", dig(fresh._scene().plate))

# 5. another maze built in between
other = se.make("PointMazeMedium", batch=8, device="cuda")
other.render(other.sample_states(4))
print("after another maze ", dig(env.render(states, style=style)), "plate", dig(env._scene().plate))
