from types import SimpleNamespace

import pytest

from stable_envs.envs import rocket_render as R


@pytest.mark.parametrize("chunk", [0, -1, 1.5, True, "4"])
def test_invalid_chunk_is_rejected_before_loading_a_backend(chunk):
    with pytest.raises(ValueError, match="positive integer"):
        R.render(SimpleNamespace(), None, 96, 1, chunk_size=chunk)
