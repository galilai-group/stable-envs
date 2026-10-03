"""Compare shared routing/padding against the frozen pre-optimization dispatcher."""

import importlib.util
from pathlib import Path

import pytest
import torch

from stable_envs._stepping import step
from tests.unit.test_step_capture import Counter, PipelinedCounter

_path = Path(__file__).parents[1] / "helpers/reference/step_capture.py"
_spec = importlib.util.spec_from_file_location("stable_envs._reference_stepping", _path)
_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference)


def equal(actual, expected):
    assert actual.batch_size == expected.batch_size
    assert set(actual.keys(True, True)) == set(expected.keys(True, True))
    for key in actual.keys(True, True):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("limits", [[20, 20, 20], [1, 3, 20], [1, 1, 1]])
@pytest.mark.parametrize("length,sequence", [(1, False), (1, True), (8, True)])
@pytest.mark.parametrize("selection", [None, slice(None), slice(None, None, -1), slice(5, 5)])
def test_owned_nested_outputs_and_terminal_routing(device, pipeline, limits, length, sequence, selection):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA requires Oscar")

    class PayloadCounter(PipelinedCounter if pipeline else Counter):
        def __init__(self):
            super().__init__(device)
            self.image = torch.zeros(3, 16, 16, 3, dtype=torch.uint8, device=device)

        def _step(self, actions, *, active=None):
            result = super()._step(actions, active=active)
            self.image.copy_(self.position.reshape(3, 1, 1, 1))
            # Serial task buffers may be borrowed. Pipeline receive owns its
            # output before it allows the next worker transition to start.
            result["info", "goal"] = self.image.clone() if pipeline else self.image
            result["info", "flag"] = self.position[:, 0] > 1
            result["observation"] = self.position.float()
            return result

    with PayloadCounter() as candidate, PayloadCounter() as reference:
        candidate.limit.copy_(torch.tensor(limits, device=device))
        reference.limit.copy_(candidate.limit)
        shape = (3, length, 1) if sequence else (3, 1)
        actions = torch.ones(shape, dtype=torch.int64, device=device)
        expected = _reference.step(reference, actions, states=selection, frames=selection)
        actual = step(candidate, actions, states=selection, frames=selection)
        equal(actual, expected)
        pixels = actual["info", "goal"][..., 0, 0, 0]
        wanted = torch.arange(1, length + 1, device=device).expand(3, length)
        wanted = torch.where(actual["valid"], wanted, 0).to(torch.uint8)
        torch.testing.assert_close(pixels, wanted if sequence else wanted[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(candidate.calls, reference.calls, rtol=0, atol=0)
        held = actual.clone()
        candidate.image.zero_()
        candidate.position.fill_(19)
        step(candidate, torch.ones(3, 1, dtype=torch.int64, device=device), frames=None)
        equal(actual, held)
        if pipeline:
            assert candidate.pending is None
