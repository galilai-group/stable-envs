import numpy as np
import pytest
import torch
from tensordict import TensorDict

from tests.helpers.reference.oracle import assert_outputs, state_at


def test_state_copy_preserves_rng_precision_dtype_and_ownership():
    arrays = {"state/rng/word": np.array([[[2**62 + 1]], [[2**62 + 3]]], dtype=np.int64)}
    state = state_at(arrays, 1, batch=1)
    assert state["rng", "word"].dtype == torch.int64
    assert state["rng", "word"].item() == 2**62 + 3
    state["rng", "word"].zero_()
    assert arrays["state/rng/word"][1, 0, 0] == 2**62 + 3
    with pytest.raises(AssertionError, match="no State"):
        state_at({"actions": np.zeros((1, 1, 1))}, 0, batch=1)


@pytest.mark.parametrize("defect", [None, "missing", "extra", "dtype", "value", "shape", "empty_oracle"])
def test_full_output_comparison_rejects_schema_and_numeric_changes(defect):
    arrays = {
        "output/reward": np.array([[0.25]], np.float64),
        "output/observation/position": np.array([[[2.0, 3.0]]], np.float32),
    }
    actual = TensorDict(
        {
            "reward": torch.tensor([0.25], dtype=torch.float64),
            ("observation", "position"): torch.tensor([[2.0, 3.0]]),
            ("state", "qpos"): torch.zeros(1, 2),
        },
        [1],
    )
    if defect == "missing":
        del actual["observation", "position"]
    elif defect == "extra":
        actual["unexpected"] = torch.zeros(1)
    elif defect == "dtype":
        actual["reward"] = actual["reward"].float()
    elif defect == "value":
        actual["reward"] += 1e-9
    elif defect == "shape":
        actual["reward"] = actual["reward"][:, None]
    elif defect == "empty_oracle":
        arrays = {}
    if defect is None:
        assert_outputs(actual, arrays, 0, equal_nan=True)
    else:
        with pytest.raises(AssertionError):
            assert_outputs(actual, arrays, 0, equal_nan=True)


@pytest.mark.parametrize("prefixed", [False, True])
def test_nan_policy_is_explicit_for_both_corpus_layouts(prefixed):
    actual = TensorDict({"score": torch.tensor([float("nan")])}, [1])
    arrays = {"output/score" if prefixed else "score": np.array([[np.nan]], np.float32)}
    keys = None if prefixed else ("score",)
    assert_outputs(actual, arrays, 0, keys=keys, equal_nan=True)
    with pytest.raises(AssertionError):
        assert_outputs(actual, arrays, 0, keys=keys, equal_nan=False)


def test_explicit_output_selection_still_requires_every_named_field():
    actual = TensorDict({"reward": torch.tensor([0.0])}, [1])
    arrays = {key: np.zeros((1, 1), np.float32) for key in ("reward", "score")}
    with pytest.raises(KeyError):
        assert_outputs(actual, arrays, 0, keys=("reward", "score"), equal_nan=False)
