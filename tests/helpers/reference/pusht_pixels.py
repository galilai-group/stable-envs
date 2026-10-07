"""Forward source witnesses for replay-State PushT, including held terminal lanes."""

import importlib
from contextlib import ExitStack, closing

import numpy as np
import torch
from tensordict import TensorDict

import stable_envs as se
from tests.helpers.reference.pixels import PixelTrace
from tests.helpers.reference.profiles import Registration
from tests.helpers.reference.pusht_semantics import diagnostics, equal
from tests.helpers.reference.semantics import compare_state


def compare_padded_states(actual: TensorDict, expected: TensorDict) -> None:
    """Compare all fields exactly, allowing only additional zero history padding."""
    capacity = max(row["history", "valid"].shape[1] for row in (actual, expected))
    padded = []
    for state in (actual, expected):
        valid = state["history", "valid"]
        lengths = state["history", "lengths"].reshape(-1)
        assert valid.dtype == torch.bool and lengths.dtype == torch.int64
        assert bool(((lengths >= 0) & (lengths <= valid.shape[1])).all())
        torch.testing.assert_close(valid, torch.arange(valid.shape[1], device=valid.device)[None] < lengths[:, None])
        for field in ("actions", "dtypes"):
            assert bool((state["history", field][~valid] == 0).all()), "Nonzero invalid history padding"
        state = state.clone()
        for field in ("actions", "dtypes", "valid"):
            value = state["history", field]
            padding = value.new_zeros((len(state), capacity - value.shape[1], *value.shape[2:]))
            state["history", field] = torch.cat((value, padding), dim=1)
        padded.append(state)
    compare_state(*padded)


def pusht_trace(case: Registration, seed: int, dtype: str, horizon: int, scenario: str) -> PixelTrace:
    discrete = case.domain == "discrete"
    module = importlib.import_module("stable_worldmodel.envs.pusht." + ("env_discrete" if discrete else "env"))
    source_class = getattr(module, case.name)
    options = {}
    if scenario in ("contact", "terminal"):
        start = np.array([250, 220, 250, 250, 0, 0, 0], np.float64)
        options = {"variation": [], "state": start}
        if scenario == "terminal":
            options["goal_state"] = start.copy()
    prefixes = (0, 0) if scenario == "terminal" else (0, 7)
    rng = np.random.default_rng(seed + 1000)
    actions = (
        rng.integers(0, 16, (2, horizon, 2), dtype=np.int64)
        if discrete
        else rng.uniform(-0.3, 0.3, (2, horizon, 2)).astype(dtype)
    )
    if scenario == "terminal":
        actions[0] = 0
        actions[1] = [15, 0] if discrete else [1, 0]
    elif scenario == "contact":
        # Drive toward the block before the random continuation; small random
        # actions can move away and miss contact during the one-action case.
        actions[0, 0] = [15, 4] if discrete else [0, 1]
    with ExitStack() as stack:
        reference = stack.enter_context(se.make(case.name, num_envs=2, size=64, device="cpu"))
        reference.reset(seed=seed, options=options)
        sources = [stack.enter_context(closing(source_class(resolution=64))) for _ in range(2)]
        for lane, (source, prefix) in enumerate(zip(sources, prefixes, strict=True)):
            source.reset(seed=seed + lane, options=options)
            for _ in range(prefix):
                action = np.zeros(2, dtype=dtype)
                expected = source.step(action.copy())
                actual = reference._simulator.envs[lane].step(action.copy())
                equal(expected, actual)
            equal(diagnostics(source), diagnostics(reference._simulator.envs[lane].env))
        initial = reference.get_state()
        states, frames, tasks, validity = [], [], [], []
        active = np.ones(2, bool)
        for step in range(horizon):
            validity.append(active.copy())
            row = np.zeros((2, 4), np.float64)
            images = []
            for lane, source in enumerate(sources):
                if active[lane]:
                    output = source.step(actions[lane, step].copy())
                    row[lane] = output[1], output[2], output[3], output[2]
                before = diagnostics(source)
                if scenario == "contact" and step == 0 and lane == 0:
                    assert source.n_contact_points > 0, "Contact scenario did not exercise a collision"
                frame = source.render().copy()
                assert frame.shape == (64, 64, 3) and frame.dtype == np.uint8
                equal(before, diagnostics(source))
                images.append(frame)
            reference._simulator.step(actions[:, step], active=torch.from_numpy(active))
            for lane, source in enumerate(sources):
                equal(diagnostics(source), diagnostics(reference._simulator.envs[lane].env))
            states.append(reference.get_state())
            frames.append(np.stack(images))
            tasks.append(row)
            active &= ~(row[:, 1].astype(bool) | row[:, 2].astype(bool))
        columns = torch.from_numpy(np.stack(tasks, axis=1))
        if scenario == "terminal":
            assert not np.stack(validity, axis=1)[0, 1:].any()
            assert np.stack(validity, axis=1)[1].all()
        for lane, source in enumerate(sources):
            action = actions[lane, -1].copy()
            equal(source.step(action.copy()), reference._simulator.envs[lane].step(action.copy()))
            equal(diagnostics(source), diagnostics(reference._simulator.envs[lane].env))
        return PixelTrace(
            initial,
            states,
            {
                key: columns[:, :, i].to(torch.float64 if key == "reward" else torch.bool)
                for i, key in enumerate(("reward", "terminated", "truncated", "success"))
            },
            np.stack(frames, axis=1),
            torch.from_numpy(np.stack(validity, axis=1)),
            actions,
        )
