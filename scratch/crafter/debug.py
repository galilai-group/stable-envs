"""Localise the Crafter GPU/official disagreement: which tiles, which rows, which values."""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, ".")
import stable_envs as se
from stable_envs.envs import crafter as CR

d = np.load("/mnt/vast/home/randall.balestriero/se_logs/crafter_pool.npz")
st, official = d["states"], d["frames"]
env = se.make("Crafter", batch=16, device="cuda")
ours = env.render(st[:8]).cpu().numpy()
diff = np.abs(ours.astype(np.int16) - official[:8].astype(np.int16)).max(-1)   # (8,64,64)

for i in range(3):
    dm = diff[i]
    print(f"frame {i}: differing px {int((dm > 0).sum())}/{64*64}, max {dm.max()}, "
          f"rows {np.flatnonzero(dm.max(1))[:6]}... cols {np.flatnonzero(dm.max(0))[:6]}...")
    tile = np.zeros((9, 9), np.int32)
    for r in range(9):
        for c in range(9):
            tile[r, c] = dm[r * 7:(r + 1) * 7, c * 7:(c + 1) * 7].max()
    print(tile)

i = 0
r0, c0 = 0, 0
print("ours  tile(0,0):\n", ours[i, r0:r0 + 7, c0:c0 + 7, 0])
print("offic tile(0,0):\n", official[i, r0:r0 + 7, c0:c0 + 7, 0])
code = int(round(float(st[i, 0])))
print("code[0,0] =", code, "mat", (("unknown",) + CR.MATERIALS)[code // 16], "obj", CR.OBJTEX[code % 16],
      "daylight", st[i, -1])
atlas = CR._Atlas.get(torch.device("cpu"), 7)
print("atlas tile:\n", atlas[code, ..., 0].numpy())

# is the world layer right BEFORE the light? render with daylight forced to 1
st1 = st[:8].copy()
st1[:, -1] = 1.0
o1 = env.render(st1).cpu().numpy()
from stable_envs.envs.crafter import Crafter
ref = Crafter.reference_env(64)
f1 = Crafter.reference_render(ref, st1)
d1 = np.abs(o1.astype(np.int16) - f1.astype(np.int16)).max(-1)
print("with daylight forced to 1.0: differing px", int((d1 > 0).sum()), "max", d1.max())
