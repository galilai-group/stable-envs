"""The plan §11 Wave-0 demonstration, executed. CPU because this Mac has no CUDA device.

Run it with the worktree on the path::

    MUJOCO_GL=glfw PYTHONPATH=$PWD /Users/samib/se6-venv/bin/python \\
        scratch/step9/foundation/demo_wave0.py
"""
import torch
import stable_envs as snv

env = snv.PushT(num_envs=64, device="cpu")     # plan §11 uses "cuda"; this Mac has no CUDA device

state = env.reset(seed=0)
action = torch.full((64, 2), 300.0)
step = env.step(action)

next_state = env.transition(state, action)
action_sequence = torch.stack([action] * 3)
trajectory = env.unroll(state, action_sequence)

pixels = env.render(state)

print("state           ", dict(state["agent"].items()).keys(), list(state.keys()), list(state.batch_size))
print("step            ", list(step.keys()), list(step.batch_size))
print("next_state      ", torch.equal(env.pack(next_state), env.pack(step["state"])), "(== step, to the bit)")
print("trajectory      ", list(trajectory.batch_size))
print("pixels          ", tuple(pixels.shape), pixels.dtype)
print("specs           ", env.state_spec.keys(nested=True), env.action_spec.shape, env.observation_spec.shape)
print("registry        ", snv.spec("swm/PushT-v1")["name"], "| envs:", len(snv.list_envs()))
env.close()
print("closed          ", env.closed)
