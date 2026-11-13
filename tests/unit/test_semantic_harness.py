import pytest
import torch
from tensordict import TensorDict

from tests.helpers.reference.semantics import check_reward_done_matches_reference, compare_state


def test_hidden_rng_and_lifecycle_mismatches_fail():
    state = TensorDict({"qpos": torch.zeros(1, 2), "rng": torch.tensor([[41]]), "elapsed": torch.tensor([[3]])}, [1])
    for key in ("qpos", "rng", "elapsed"):
        wrong = state.clone()
        wrong[key] += 1
        with pytest.raises(AssertionError, match=key):
            compare_state(wrong, state)


def test_missing_termination_is_not_a_false_termination():
    result = TensorDict({"reward": torch.zeros(1), "terminated": torch.zeros(1, dtype=torch.bool)}, [1])
    with pytest.raises(AssertionError, match="truncated"):
        check_reward_done_matches_reference(result, result)
