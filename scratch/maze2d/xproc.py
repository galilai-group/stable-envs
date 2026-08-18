import sys, hashlib, numpy as np
sys.path.insert(0, ".")
import stable_envs as se
from stable_envs.testing import render_worker as W

def dig(a):
    a = np.ascontiguousarray(a if isinstance(a, np.ndarray) else a.detach().cpu().numpy())
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]

name = sys.argv[1]
env = se.make(name, batch=8, device="cuda")
states, style = W.sample_inputs(env, 8)
sc = env._scene()
print("plate      ", dig(sc.plate), "plate_t", dig(sc.plate_t.float()))
print("frames_def ", dig(env.render(states)))
print("frames_sty ", dig(env.render(states, style=style)))
u = np.unique(np.round(style.astype(np.float64), 6), axis=0)
print("style rows ", len(u), "cam", u[:, -6:].round(4).tolist())
for i, row in enumerate(u):
    over = {p: row[o:o+w] for p, (o, w) in zip(env.PLATE_PATHS,
            __import__("stable_envs.envs.maze2d", fromlist=["x"])._offsets(env.PLATE_PATHS, env._style))}
    s2 = env._scene(env.size, over)
    print(f"  plate[{i}]", dig(s2.plate))
