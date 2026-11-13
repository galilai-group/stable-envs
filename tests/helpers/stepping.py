def branch_step(env, state, actions):
    held = env.get_state()
    try:
        env.set_state(state)
        return env.step(actions, frames=None)["state"]
    finally:
        env.set_state(held)


def branch_steps(env, state, actions):
    # Reference action arrays use [T, B, A]; the public chunk API uses [B, T, A].
    held = env.get_state()
    try:
        env.set_state(state)
        import torch

        if actions.ndim != 3:
            raise ValueError("Test reference actions must be time first [T,B,A]")
        return torch.stack([env.step(action, frames=None)["state"] for action in actions])
    finally:
        env.set_state(held)
