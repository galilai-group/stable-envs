from contextlib import closing

import numpy as np
import torch

from tests.helpers.reference.native_navigation_semantics import NativeNavigationReference
from tests.helpers.reference.pixels import PixelTrace
from tests.helpers.reference.profiles import Registration
from tests.helpers.reference.semantics import compare_state


def navigation_trace(case: Registration, seed: int, dtype: str, horizon: int, scenario: str) -> PixelTrace:
    """Capture scalar CPU source rollouts at native resolution, without restoring source lanes."""
    with closing(NativeNavigationReference(case.domain, 2)) as source:
        if scenario == "terminal":
            for lane, env in enumerate(source.envs):
                if case.domain == "simple_point_maze":
                    # Source reset preserves these initial positions; all appearance stays at defaults.
                    env.variation_space["goal"]["position"].set_init_value(
                        np.asarray([0.8 + 0.4 * lane, 0.5], dtype=np.float32)
                    )
                    options = {"variation": []}
                else:
                    offset = (20, 34 if case.domain == "piecewise" else 30)[lane]
                    options = {"state": [50, 50], "target_state": [50 + offset, 50]}
                env.reset(seed=seed + lane, options=options)
            actions = np.zeros((2, horizon, 2), dtype=dtype)
            actions[..., 0] = 0.2 if case.domain == "simple_point_maze" else 1.0
        else:
            source.reset(seed)
            for env, prefix in zip(source.envs, (7, 0), strict=True):
                for t in range(prefix):
                    output = env.step(np.full(2, 0.01 * (-1) ** t, dtype=dtype))
                    assert not (output[2] or output[3]), "Source prefix ended before its checkpoint"
            actions = np.random.default_rng(seed + 1000).uniform(-0.01, 0.01, (2, horizon, 2)).astype(dtype)
        initial = source.get_state()
        active = np.ones(2, dtype=bool)
        states, frames, validity, outputs = [], [], [], []
        for t in range(horizon):
            validity.append(active.copy())
            output = source.step(actions[:, t], active=active)
            done = (output["terminated"] | output["truncated"]).numpy()
            tasks = {key: value.clone() for key, value in output.exclude("state").items(True, True)}
            for value in tasks.values():
                value[~active] = 0
            outputs.append(tasks)
            states.append(source.get_state())
            frames.append(np.stack([env.render().copy() for env in source.envs]))
            compare_state(source.get_state(), states[-1])
            active &= ~done
        if scenario == "terminal":
            np.testing.assert_array_equal(np.stack(validity, axis=1), np.arange(horizon)[None, :] < [[1], [3]])
        return PixelTrace(
            initial,
            states,
            {key: torch.stack([row[key] for row in outputs], dim=1) for key in outputs[0]},
            np.stack(frames, axis=1),
            torch.from_numpy(np.stack(validity, axis=1)),
            actions,
        )
