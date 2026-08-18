"""Sokoban's transition, by ENUMERATION: all nine actions from every state in the pool.

A random rollout is a weak test of a decision table -- it walks into walls far more often than it
pushes a box onto a target, and the ``box_type = 3`` branch is exactly the one a port gets wrong. So
this drives every action from every sampled state and compares cell for cell against the official
``SokobanEnv._push`` / ``_move``.

    python scratch/sokoban/enumerate_step.py [n_states]
"""
import os
import sys
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from stable_envs.envs.sokoban import N_ACTIONS, Sokoban  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 512
env = Sokoban(batch=n, device="cuda")
states = env.sample_states(n, rng=np.random.default_rng(0))
ref = Sokoban.reference_env()

bad = 0
per_action = {}
outcomes = {"pushed_onto_target": 0, "pushed_off_target": 0, "pushed": 0, "walked": 0, "blocked": 0}
for a in range(N_ACTIONS):
    env.set_state(states)
    got = env.step(torch.full((n, 1), float(a), device="cuda")).cpu().numpy()
    want = np.empty_like(states)
    for i, st in enumerate(states):
        Sokoban.reference_set_state(ref, st)
        before = Sokoban.reference_sim_state(ref)
        Sokoban.reference_step(ref, [a])
        want[i] = Sokoban.reference_sim_state(ref)
        d = want[i] - before
        if not d.any():
            outcomes["blocked"] += 1
        elif (want[i] == 3).sum() > (before == 3).sum():
            outcomes["pushed_onto_target"] += 1
        elif (want[i] == 3).sum() < (before == 3).sum():
            outcomes["pushed_off_target"] += 1
        elif (before == 4).sum() and not np.array_equal(before == 4, want[i] == 4):
            outcomes["pushed"] += 1
        else:
            outcomes["walked"] += 1
    m = int((np.round(got) != np.round(want)).sum())
    per_action[a] = m
    bad += m
print(f"{n} states x {N_ACTIONS} actions = {n*N_ACTIONS} transitions, {n*N_ACTIONS*env.state_dim} cells")
print(f"mismatched cells: {bad}   per action: {per_action}")
print(f"outcomes exercised: {outcomes}")
print(f"exact_frac {1.0 - bad / (n*N_ACTIONS*env.state_dim):.9f}")
assert bad == 0, "the batched step disagrees with SokobanEnv"
