"""How does MuJoCo composite the translucent target site? Measure the law, do not read it."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import mujoco

gym.register_envs(gymnasium_robotics)

e = gym.make("PointMaze_Medium-v3", render_mode="rgb_array", width=224, height=224,
             continuing_task=False)
e.reset(seed=0)
u = e.unwrapped
m = u.model
tid = u.target_site_id
q = np.array([0.3, -0.4])
g = np.array([1.0, 2.0])


def render(alpha=None, offsamples=None):
    if alpha is not None:
        m.site_rgba[tid, 3] = alpha
    if offsamples is not None:
        m.vis.quality.offsamples = offsamples
    u.data.qpos[:] = q
    u.data.qvel[:] = 0.0
    u.goal = g
    u.update_target_site_pos()
    mujoco.mj_forward(m, u.data)
    return u.point_env.mujoco_renderer.render("rgb_array").copy().astype(np.float64)


base = render(0.0)
opaque = render(1.0)
obs = render(0.7)
mask = np.abs(opaque - base).max(-1) > 0
mask_soft = np.abs(obs - base).max(-1) > 0
print("opaque target pixels", int(mask.sum()), " translucent-affected pixels", int(mask_soft.sum()))

# erode so silhouette coverage is not in play
import scipy.ndimage as ndi
inner = ndi.binary_erosion(mask, np.ones((3, 3), bool))
print("eroded interior target pixels", int(inner.sum()))

for name, mk in (("all", mask), ("interior", inner)):
    if not mk.any():
        continue
    A, B, C = obs[mk], base[mk], opaque[mk]
    for al in (0.7,):
        pred = al * C + (1 - al) * B
        err = np.abs(pred - A)
        print(f"  {name}: src-over alpha {al}: mean {err.mean():.3f} max {err.max():.3f}")

# is the pixel AFFINE or QUADRATIC in alpha? render a sweep on interior pixels
alphas = np.array([0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 1.0])
vals = np.stack([render(float(a))[inner] for a in alphas])          # (A, P, 3)
m.site_rgba[tid, 3] = 0.7
p = 0
for p in range(min(3, vals.shape[1])):
    print(f"  pixel {p}: " + "  ".join(f"a={a:.1f}:{vals[i, p].astype(int).tolist()}"
                                       for i, a in enumerate(alphas)))
# fit linear and quadratic in alpha, per channel, over all interior pixels
X = alphas.reshape(-1, 1)
lin = np.stack([np.ones_like(alphas), alphas], 1)
quad = np.stack([np.ones_like(alphas), alphas, alphas ** 2], 1)
Y = vals.reshape(len(alphas), -1)
for nm, M in (("linear", lin), ("quadratic", quad)):
    coef, *_ = np.linalg.lstsq(M, Y, rcond=None)
    r = np.abs(M @ coef - Y)
    print(f"  fit {nm} in alpha: max residual {r.max():.4f} mean {r.mean():.5f}")

# implied back-surface colour under the two-pass model out = a*Cf + (1-a)*(a*Cb + (1-a)*bg)
a = 0.7
Cb = (obs[inner] - a * opaque[inner] - (1 - a) ** 2 * base[inner]) / (a * (1 - a))
print("  implied back-face colour: mean", Cb.mean(0).round(2).tolist(),
      "min", Cb.min(0).round(2).tolist(), "max", Cb.max(0).round(2).tolist())
print("  front colour: mean", opaque[inner].mean(0).round(2).tolist(),
      "  background: mean", base[inner].mean(0).round(2).tolist())
