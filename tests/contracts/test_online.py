"""Forward-only source control for the public single/chunk State capture path.

Frame/process combinations retain their existing family acceptance gates;
this small profile does not imply rendered online or multiprocessing acceptance.
"""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from tests.helpers.reference.bridges import assert_tasks
from tests.helpers.reference.semantics import compare_state

pytestmark = [pytest.mark.integration, pytest.mark.reference]


def test_selected_states_follow_source(worlds, source_case):
    candidate, destination, source, actions = worlds
    _, seed, _ = source_case
    for horizon in (1, 8):
        for selection in (None, slice(None), slice(-1, None), slice(None, None, -2), slice(99, None)):
            candidate.reset(seed=seed)
            source.reset(seed)
            chunk = actions[:horizon].transpose(1, 0, 2).copy()
            result = candidate.step(chunk[:, 0] if horizon == 1 else chunk, states=selection, frames=None)
            assert all(value.device.type == candidate.device.type for value in result.values(True, True))
            frozen = result.clone()
            indices = [] if selection is None else list(range(horizon))[selection]
            active = np.ones(2, dtype=bool)
            for t in range(horizon):
                expected = source.step(actions[t]) if active.all() else source.step(actions[t], active=active)
                expected = TensorDict({key: value for key, value in expected.items() if key != "state"}, [2]).clone()
                done = (expected["terminated"] | expected["truncated"]).cpu().numpy().copy()
                expected[~torch.tensor(active)] = 0
                actual = {key: result[key] if horizon == 1 else result[key][:, t] for key in expected.keys()}
                assert_tasks(actual, expected, candidate.device)
                torch.testing.assert_close(result["valid"][:, t].cpu(), torch.tensor(active))
                active &= ~done
                if t in indices:
                    selected = result["states"][:, indices.index(t)]
                    destination.set_state(selected)
                    source.compare(destination)
            source.compare(candidate)
            if selection is not None:
                assert result["states"].batch_size == torch.Size([2, len(indices)])
            candidate.step(actions[0], states=None, frames=None)
            compare_state(result, frozen)
