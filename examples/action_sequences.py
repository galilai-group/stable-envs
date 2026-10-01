"""Run with python examples/action_sequences.py from an installed checkout."""

import torch

import stable_envs as snv

with snv.CartPole(num_envs=8, device="cpu") as env:
    env.reset(seed=17)
    one = env.step(torch.zeros(8, 1, dtype=torch.int64))
    print("One frame:", tuple(one["frames"][:,0].shape))
    actions = torch.zeros(8, 4, 1, dtype=torch.int64)
    sequence = env.step(actions)
    print("State history:", sequence["states"].batch_size)
    print("Executed transitions:", sequence["valid"].sum().item())
    print("Final frame:", tuple(sequence["frames"][:,0].shape))
    physics = env.step(actions, frames=None)
    assert "frames" not in physics.keys()
