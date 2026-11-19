"""The packed copies of the PushT batch fields keep every value, dtype and shape."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stable_envs.envs.pusht.simulation.batch import _packed_copies

pytestmark = pytest.mark.unit

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))]


@pytest.mark.parametrize("device", DEVICES)
def test_packed_copies_keep_values_and_own_their_memory(device):
    rng = np.random.default_rng(0)
    fields = {
        "a": rng.standard_normal((3, 5)),
        "b": rng.integers(0, 9, (3,), dtype=np.int64),
        "c": rng.standard_normal((3, 0)),
        "d": rng.integers(0, 2, (3, 7)).astype(bool),
        "e": rng.standard_normal((3, 2, 3))[:, :, ::2],
        "f": rng.integers(0, 255, (3, 3), dtype=np.uint8),
        "g": np.array(4, np.int64),
    }
    out = _packed_copies(fields, torch.device(device))
    assert list(out) == list(fields)
    for key, value in fields.items():
        tensor = out[key]
        assert tensor.device.type == device and tensor.is_contiguous()
        assert tensor.data_ptr() % 16 == 0 or tensor.numel() == 0
        np.testing.assert_array_equal(tensor.cpu().numpy(), value, err_msg=key)
        assert tensor.cpu().numpy().dtype == value.dtype
    out["a"].fill_(7)
    out["b"].fill_(-1)
    assert not (fields["a"] == 7).any()
    np.testing.assert_array_equal(out["e"].cpu().numpy(), fields["e"])
