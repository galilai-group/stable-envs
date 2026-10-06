import pytest
import torch

from stable_envs.render import tilemap


@pytest.mark.parametrize("triton_available", [False, True])
def test_cpu_tiles_and_visibility_ignore_cuda_backend(monkeypatch, triton_available):
    monkeypatch.setattr(tilemap, "HAVE_TRITON", triton_available)
    atlas = torch.arange(24, dtype=torch.uint8).reshape(2, 2, 2, 3)
    indices = torch.tensor([[[1, 0]]], dtype=torch.int32)
    expected = torch.cat([atlas[1], atlas[0]], dim=1).unsqueeze(0)
    out = torch.empty_like(expected)
    assert tilemap.blit(atlas, indices, 2, out=out) is out
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    transparent = torch.zeros(2, 3, 3, dtype=torch.bool)
    transparent[1] = True
    visible = tilemap.visibility_scan(transparent, 3)
    expected_visible = torch.zeros_like(transparent)
    expected_visible[0, 1, 2] = True
    expected_visible[1] = True
    assert torch.equal(visible, expected_visible)
