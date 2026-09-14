import os
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np, stable_envs as se
from stable_envs.testing import conformance as C

e = se.make("DMCReacherHard", batch=8, device="cpu")
st = np.load("examples/results/step4/states/DMCReacherHard.npz")["states"][:8]
on = e.render(st).cpu().numpy()
off = e.render(st, variation={"visibility.target": 0}).cpu().numpy()

# per-row: alternate the knob down the batch
sty = np.tile(e.default_style(), (8, 1))
j = dict((p,o) for p,o,w in e.variation_space.flat_layout())["visibility.target"]
sty[1::2, j] = 0.0
mix = e.render(st, style=sty).cpu().numpy()
d_on = np.abs(mix[0::2].astype(int) - on[0::2].astype(int)).max()
d_off = np.abs(mix[1::2].astype(int) - off[1::2].astype(int)).max()
print("per-row on-rows match all-on :", d_on)
print("per-row off-rows match all-off:", d_off)
print("on vs off distinct frames     :", len(np.unique(mix.reshape(8, -1), axis=0)))

# order / permutation invariance with a per-row style
p = np.array([5, 0, 7, 2, 1, 6, 3, 4])
perm = e.render(st[p], style=sty[p]).cpu().numpy()
print("permutation max|d|            :", np.abs(perm.astype(int) - mix[p].astype(int)).max())

# batch-size invariance with a per-row style
sub = e.render(st[3:5], style=sty[3:5]).cpu().numpy()
print("sub-batch max|d|              :", np.abs(sub.astype(int) - mix[3:5].astype(int)).max())

# render purity: style must come back unchanged
before = e.get_style().clone()
e.render(st, style=sty)
print("style mutated by render       :", float((e.get_style() - before).abs().max()))
