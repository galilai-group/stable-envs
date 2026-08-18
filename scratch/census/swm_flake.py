"""How often does swm-nav parity land below 1.0, and by how much?

The parity test draws its states unseeded, so a failure is not reproducible and a pass is not evidence.
Seeding it is the obvious fix and is also the way to cheat: pick a seed, get a green suite, and never
learn that a fraction of draws disagree. So measure the RATE first over many independent draws, and let
that decide whether the right answer is a seed or a declared floor with a stated mechanism.
"""
from __future__ import annotations

import numpy as np
import torch

import stable_envs as se
from stable_envs.testing import conformance as C

ENVS = ["TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones"]
DRAWS = 40

for name in ENVS:
    if name not in se.REGISTRY:
        print(f"{name}: not registered")
        continue
    cls = se.REGISTRY[name]
    env = se.make(name, batch=64, device="cuda")
    ref = cls.reference_env(size=env.size)
    below, fracs, worst = 0, [], None
    for d in range(DRAWS):
        states = env.sample_states(64, rng=np.random.default_rng(d))
        official = cls.reference_render(ref, states)
        ours = env.render(states).cpu().numpy()
        r = C.pixel_agreement(ours, official)
        f = r["interior_exact_frac"]
        if f is None:
            continue
        fracs.append(f)
        if f < 1.0:
            below += 1
            n_bad = int(round((1 - f) * r["interior_frac"] * ours[..., 0].size))
            if worst is None or f < worst[1]:
                worst = (d, f, n_bad, r["max_abs"])
    print(f"{name:16s} {below}/{len(fracs)} draws below 1.0   min {min(fracs):.9f}   "
          f"worst seed {worst[0] if worst else '-'} "
          f"({worst[2] if worst else 0} px, max_abs {worst[3] if worst else 0})", flush=True)
    del env
    torch.cuda.empty_cache()
