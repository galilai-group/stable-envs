from collections.abc import Mapping, Sequence

import torch
from numpy.typing import NDArray
from tensordict import TensorDict


def state_at(arrays: Mapping[str, NDArray], index: int, *, batch: int) -> TensorDict:
    """Copy a full State from the legacy reference-only NPZ layout."""
    state = TensorDict({}, [batch])
    for key in arrays:
        if key.startswith("state/"):
            state[tuple(key[6:].split("/"))] = torch.from_numpy(arrays[key][index].copy())
    assert len(state.keys(True, True)), "Oracle has no State fields"
    return state


def assert_outputs(
    actual: TensorDict,
    arrays: Mapping[str, NDArray],
    index: int,
    *,
    keys: Sequence[str] | None = None,
    equal_nan: bool,
) -> None:
    """Compare exact outputs; explicit keys select older unprefixed corpora."""
    if keys is None:
        expected = {key[7:]: arrays[key][index] for key in arrays if key.startswith("output/")}
        observed = {
            key if isinstance(key, str) else "/".join(key): actual[key]
            for key in actual.keys(True, True)
            if key != "state" and not (isinstance(key, tuple) and key[0] == "state")
        }
        assert expected, "Oracle has no output fields"
        assert observed.keys() == expected.keys(), f"Output keys differ: {observed.keys() ^ expected.keys()}"
    else:
        expected = {key: arrays[key][index] for key in keys}
        observed = {key: actual[key] for key in keys}
    for key, value in expected.items():
        torch.testing.assert_close(
            observed[key].cpu(),
            torch.from_numpy(value),
            atol=0,
            rtol=0,
            equal_nan=equal_nan,
            msg=lambda message, key=key: f"Output {key}: {message}",
        )
