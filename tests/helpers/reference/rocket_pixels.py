from __future__ import annotations

import importlib.metadata
import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

import stable_envs as se
from stable_envs.testing.evidence import sha256
from tests.helpers.reference.pixels import PixelTrace
from tests.helpers.reference.profiles import PINS
from tests.helpers.reference.saved import ROCKET_SOURCE

SCENARIOS = {
    "flight": ({}, {}, (0, 3)),
    "contact": ({}, {}, (301, 303)),
    "success": ({"ceiling": 3.2}, {"accelerate_drop": False}, (0, 7)),
    "horizon": ({"ceiling": 500000.0}, {"accelerate_drop": False}, (1198, 1201)),
}


@contextmanager
def rocket_trace(
    scenario: str, dtype: type[np.float32] | type[np.float64], horizon: int
) -> Iterator[tuple[dict[str, Any], PixelTrace]]:
    """Witness candidate checkpoints against independent, never-restored source lanes."""
    from tests.helpers.reference.rocket_semantics import RocketLandingEnv as Oracle
    from tests.helpers.reference.rocket_semantics import capture, differences

    assert sha256(inspect.getfile(Oracle)) == ROCKET_SOURCE, "Unpinned Rocket source"
    for package, version in PINS["rocket"].items():
        assert importlib.metadata.version(package) == version, f"Unpinned {package}; use the Rocket source profile"
    config, options, prefixes = SCENARIOS[scenario]
    config = dict(config, render_mode="rgb_array", render_resolution=(96, 96))
    source = []
    # The witness steps the lanes of the in-process simulator.
    with se.tuning.override(workers=0):
        reference = se.make("PyFlytRocketLanding", num_envs=2, size=96, simulator_config=config)
    with reference:
        try:
            reference.reset(seed=17, options=options)
            for lane, prefix in enumerate(prefixes):
                oracle = Oracle(**config)
                source.append(oracle)
                oracle.reset(seed=17 + lane, options=options)
                for _ in range(prefix):
                    oracle.step(np.zeros(7, dtype=dtype))
                    reference._simulator.envs[lane].step(np.zeros(7, dtype=dtype))
                assert not differences(capture(oracle), capture(reference._simulator.envs[lane].env))
            initial = reference.get_state()
            actions = np.zeros((2, horizon, 7), dtype=dtype)
            if scenario == "flight":
                actions[:] = [0.15, -0.1, 0.2, 1, 0.5, 0.1, -0.1]
            active = np.ones(2, dtype=bool)
            states, tasks, validity, frames = [], [], [], []
            for t in range(horizon):
                validity.append(active.copy())
                row = np.zeros((2, 4), np.float64)
                next_active = active.copy()
                for lane, oracle in enumerate(source):
                    if active[lane]:
                        out = oracle.step(actions[lane, t].copy())
                        row[lane] = [out[1], out[2], out[3], out[4]["env_complete"]]
                        next_active[lane] = not (out[2] or out[3])
                reference._simulator.step(actions[:, t], active=torch.from_numpy(active))
                for lane, oracle in enumerate(source):
                    assert not differences(capture(oracle), capture(reference._simulator.envs[lane].env))
                states.append(reference.get_state())
                tasks.append(row)
                frames.append(np.stack([oracle.render() for oracle in source]))
                active = next_active
            columns = torch.from_numpy(np.stack(tasks, axis=1))
            tasks = {
                key: columns[:, :, i].to(torch.float64 if key == "reward" else torch.bool)
                for i, key in enumerate(("reward", "terminated", "truncated", "success"))
            }
            yield (
                config,
                PixelTrace(
                    initial,
                    states,
                    tasks,
                    np.stack(frames, axis=1),
                    torch.from_numpy(np.stack(validity, axis=1)),
                    actions,
                ),
            )
        finally:
            for oracle in source:
                oracle.close()
                if oracle.modified_pad_urdf:
                    Path(oracle.modified_pad_urdf).unlink(missing_ok=True)
