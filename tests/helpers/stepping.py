"""Test-only branching through the public snapshot/step API."""


def branch_step(env, state, actions):
    held = env.get_state()
    try:
        env.set_state(state)
        return env.step(actions, frames=None)["state"]
    finally:
        env.set_state(held)


def branch_steps(env, state, actions):
    # Older trajectory fixtures are time-first. The public API is batch-first.
    held = env.get_state()
    try:
        env.set_state(state)
        import torch

        if actions.ndim != 3:
            raise ValueError("Test reference actions must be time first [T,B,A]")
        return torch.stack([env.step(action, frames=None)["state"] for action in actions])
    finally:
        env.set_state(held)
