from types import SimpleNamespace

import pytest
import torch

from stable_envs.render.nvdiff import BatchRenderer

pytestmark = pytest.mark.gpu


def test_anisotropic_texture_matches_known_footprint():
    if not torch.cuda.is_available():
        pytest.skip("nvdiffrast requires CUDA")
    dr = pytest.importorskip("nvdiffrast.torch")
    texture = torch.rand(16, 16, 3, generator=torch.Generator().manual_seed(17)).cuda()
    uv = torch.tensor([[[[0.5, 0.5]]]], device="cuda")
    derivatives = torch.tensor([[[[0.4, 0.05, 0.3, 0.02]]]], device="cuda")
    options = dict(filter_mode="linear-mipmap-linear", boundary_mode="wrap")
    first = dr.texture(texture[None], torch.tensor([[[[0.4, 0.425]]]], device="cuda"), derivatives * 0.5, **options)
    second = dr.texture(texture[None], torch.tensor([[[[0.6, 0.575]]]], device="cuda"), derivatives * 0.5, **options)
    actual = BatchRenderer._sample(SimpleNamespace(aniso=2), texture, uv, derivatives)
    torch.testing.assert_close(actual, (first + second) / 2, rtol=0, atol=1e-6)
    actual = BatchRenderer._sample(SimpleNamespace(aniso=1), texture, uv, derivatives)
    torch.testing.assert_close(actual, dr.texture(texture[None], uv, derivatives, **options), rtol=0, atol=0)
