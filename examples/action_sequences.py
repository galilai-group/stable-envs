"""Single actions and known action chunks, with independent State and frame selection.

python examples/action_sequences.py
"""

import torch

import stable_envs as snv


def main():
    batch, horizon = 8, 8
    with snv.CartPole(num_envs=batch, device="cpu") as env:
        initial = env.reset(seed=17)
        # [B,A]: the default selects one State and the final frame.
        one = env.step(torch.zeros(batch, 1, dtype=torch.int64))
        print("Single action, frames:", tuple(one["frames"].shape))

        # [B,T,A]: actions must already be known; this is not a policy rollout.
        actions = torch.zeros(batch, horizon, 1, dtype=torch.int64)
        env.set_state(initial)
        sequence = env.step(actions, frames=slice(None))
        print("All States:", sequence["states"].batch_size)
        print("All frames:", tuple(sequence["frames"].shape))
        print("Valid transitions:", sequence["valid"].sum().item())
        final_state = sequence["state"]

        # Same actions from the same State, selecting the two histories separately.
        env.set_state(initial)
        sparse = env.step(actions, states=slice(None, None, 4), frames=slice(3, None, 4))
        print("States at indices 0,4:", sparse["states"].batch_size)
        print("Frames at indices 3,7:", tuple(sparse["frames"].shape))
        torch.testing.assert_close(sparse["frames"], sequence["frames"][:, 3::4], rtol=0, atol=0)
        assert torch.equal(sparse["frame_valid"], sequence["valid"][:, 3::4])

        env.set_state(final_state)
        endpoint_only = env.step(actions[:, 0], states=None, frames=None)
        assert "states" not in endpoint_only and "frames" not in endpoint_only
        print("Final State without histories:", endpoint_only["state"].batch_size)


if __name__ == "__main__":
    main()
