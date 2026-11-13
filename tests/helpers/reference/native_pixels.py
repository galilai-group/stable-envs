from contextlib import closing

import numpy as np
import torch

from tests.helpers.reference.native_classic_semantics import NativeClassicReference
from tests.helpers.reference.pixels import PixelTrace
from tests.helpers.reference.profiles import Registration
from tests.helpers.reference.semantics import compare_state


def native_trace(case: Registration, seed: int, dtype: str, horizon: int, scenario: str) -> PixelTrace:
    """Capture source-owned States and native-resolution pixels without restoring source lanes."""
    with closing(NativeClassicReference(case.domain, 2, render_mode="rgb_array")) as source:
        source.reset(seed)
        limit = source.envs[0].spec.max_episode_steps
        prefixes = {"reached": (0, 7), "horizon": (limit - 1, limit - 3)}[scenario]
        for env, prefix in zip(source.envs, prefixes, strict=True):
            for _ in range(prefix):
                if case.domain == "cartpole":
                    # Feedback keeps the pole upright until the real time limit.
                    x, velocity, angle, angular_velocity = env.unwrapped.state
                    action = int(angle + 0.15 * angular_velocity + 0.01 * x + 0.02 * velocity > 0)
                else:
                    action = 1 if dtype == "int64" else np.zeros(1, dtype=dtype)
                output = env.step(action)
                assert not (output[2] or output[3]), "Source prefix ended before its checkpoint"
        initial = source.get_state()
        rng = np.random.default_rng(seed + 1000)
        actions = (
            rng.integers(0, 2 if case.domain == "cartpole" else 3, (2, horizon, 1), dtype=np.int64)
            if dtype == "int64"
            else rng.uniform(-0.2, 0.2, (2, horizon, 1)).astype(dtype)
        )
        active = np.ones(2, dtype=bool)
        states, frames, validity, outputs = [], [], [], []
        previous = [None, None]
        for t in range(horizon):
            validity.append(active.copy())
            rows = []
            for lane, env in enumerate(source.envs):
                if active[lane]:
                    action = actions[lane, t, 0].item() if dtype == "int64" else actions[lane, t].copy()
                    observation, reward, terminated, truncated, info = env.step(action)
                    row = dict(
                        observation=observation.copy(),
                        info_state=info["state"].copy(),
                        reward=np.float64(reward),
                        terminated=np.bool_(terminated),
                        truncated=np.bool_(truncated),
                    )
                    previous[lane] = row
                    active[lane] = not (terminated or truncated)
                else:
                    row = {key: np.zeros_like(value) for key, value in previous[lane].items()}
                rows.append(row)
            states.append(source.get_state())
            frames.append(np.stack([env.render().copy() for env in source.envs]))
            compare_state(source.get_state(), states[-1])
            outputs.append({key: np.stack([row[key] for row in rows]) for key in rows[0]})
        if scenario == "horizon":
            np.testing.assert_array_equal(np.stack(validity, axis=1), np.arange(horizon)[None, :] < [[1], [3]])
        return PixelTrace(
            initial,
            states,
            {key: torch.from_numpy(np.stack([row[key] for row in outputs], axis=1)) for key in outputs[0]},
            np.stack(frames, axis=1),
            torch.from_numpy(np.stack(validity, axis=1)),
            actions,
        )
