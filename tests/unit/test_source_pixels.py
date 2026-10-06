import numpy as np
import pytest
import torch
from tensordict import TensorDict

from stable_envs.testing.evidence import verify_seal
from tests.helpers.reference.pixels import PixelTrace, check_online_pixels


@pytest.fixture
def trace():
    positions = torch.tensor([[1, 1, 1], [11, 12, 13]])
    states = [TensorDict({"position": positions[:, t : t + 1].clone()}, [2]) for t in range(3)]
    return PixelTrace(
        TensorDict({"position": torch.tensor([[0], [10]])}, [2]),
        states,
        {"reward": torch.tensor([[1.0, 0.0, 0.0], [11.0, 12.0, 13.0]])},
        (positions.numpy() * 16).astype(np.uint8)[:, :, None, None, None].repeat(3, axis=-1),
        torch.tensor([[True, False, False], [True, True, True]]),
        np.ones((2, 3, 1), np.int64),
    )


class CapturedCandidate:
    device = torch.device("cpu")

    def __init__(self, trace, defect):
        self.trace, self.defect = trace, defect
        self.calls = 0

    def set_state(self, state):
        self.state = state.clone()

    def get_state(self):
        return self.state.clone()

    def step(self, actions, *, states, frames):
        self.calls += 1
        if self.calls > 1:
            if self.defect == "aliased":
                self.result["reward"].zero_()
            return
        self.state = self.trace.states[-1].clone()
        result = TensorDict(
            {
                "state": self.state.clone(),
                "valid": self.trace.valid.clone(),
                **{k: v.clone() for k, v in self.trace.tasks.items()},
            },
            [2],
        )
        for name, selection, validity in (("states", states, "state_valid"), ("frames", frames, "frame_valid")):
            if selection is not None:
                indices = list(range(3))[selection]
                result[name] = (
                    torch.stack(self.trace.states, 1)[:, indices].clone()
                    if name == "states"
                    else torch.from_numpy(self.trace.frames[:, indices].copy())
                )
                result[validity] = self.trace.valid[:, indices].clone()
        if self.defect == "pixels":
            result["frames"][0, 0] = 255
        elif self.defect == "order":
            result["frames"] = result["frames"].flip(1)
        elif self.defect == "mask":
            result["frame_valid"].fill_(True)
        elif self.defect == "state":
            result["states"]["position"][1, 0] += 1
        elif self.defect == "task":
            result["reward"][0, 1] = 1
        self.result = result
        return result


@pytest.mark.parametrize(
    "states,frames",
    [
        (None, slice(-1, None)),
        (slice(None), None),
        (slice(None, None, -1), slice(None, None, -2)),
        (slice(9, 10), slice(9, 10)),
    ],
)
def test_independent_selections_and_empty_frames(trace, states, frames, tmp_path):
    errors = check_online_pixels(
        CapturedCandidate(trace, None),
        trace,
        states=states,
        frames=frames,
        mse_max=0,
        failure_dir=tmp_path / "failure",
    )
    assert np.all(errors == 0)
    if frames is None or frames.start == 9:
        assert errors.size == 0
    assert not (tmp_path / "failure").exists()


@pytest.mark.parametrize("defect", ["pixels", "order", "mask", "state", "task", "aliased"])
def test_corrupt_capture_fails_and_preserves_a_sealed_witness(trace, defect, tmp_path):
    with pytest.raises(AssertionError):
        check_online_pixels(
            CapturedCandidate(trace, defect),
            trace,
            states=slice(None),
            frames=slice(None),
            mse_max=0.001,
            failure_dir=tmp_path / "failure",
        )
    assert set(verify_seal(tmp_path / "failure")) == {"failure.json", "failure.npz"}
    with np.load(tmp_path / "failure/failure.npz", allow_pickle=False) as witness:
        np.testing.assert_array_equal(witness["source_frames"], trace.frames)
        assert {"ours", "actions", "initial/position", "state/2/position"} <= set(witness.files)
