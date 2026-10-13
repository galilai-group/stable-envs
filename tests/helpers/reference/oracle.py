from collections.abc import Mapping, Sequence

import torch
from numpy.typing import NDArray
from tensordict import TensorDict

from tests.helpers.reference.saved import assert_frozen_equal


def state_at(arrays: Mapping[str, NDArray], index: int, *, batch: int, physics=None) -> TensorDict:
    """Copy a full State from the legacy reference-only NPZ layout.

    The corpora predate State physics fields and were recorded at the registered physics;
    ``physics`` is a ``PhysicsLayout`` whose nominal values are added."""
    state = TensorDict({}, [batch])
    for key in arrays:
        if key.startswith("state/"):
            state[tuple(key[6:].split("/"))] = torch.from_numpy(arrays[key][index].copy())
    assert len(state.keys(True, True)), "Oracle has no State fields"
    if physics is not None:
        for leaf, offset, width in zip(physics.leaves, physics.offsets, physics.widths, strict=True):
            value = torch.tensor(physics.nominal[offset : offset + width], dtype=torch.float64)
            state[("physics", *leaf.split("."))] = value.expand(batch, width).clone()
    return state


def assert_outputs(
    actual: TensorDict,
    arrays: Mapping[str, NDArray],
    index: int,
    *,
    keys: Sequence[str] | None = None,
    equal_nan: bool,
) -> None:
    """Compare outputs, floats within the frozen-corpus tolerance; explicit keys select older unprefixed corpora."""
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
        assert_frozen_equal(
            observed[key],
            value,
            equal_nan=equal_nan,
            msg=lambda message, key=key: f"Output {key}: {message}",
        )
