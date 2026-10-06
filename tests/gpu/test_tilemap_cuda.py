import pytest
import torch

from stable_envs.render import tilemap

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("tile_size,view_size", [(2, 3), (8, 7)])
def test_cuda_tile_paths_match_torch(tile_size, view_size):
    if not torch.cuda.is_available() or not tilemap.HAVE_TRITON:
        pytest.skip("Tile kernels require CUDA and Triton")
    rng = torch.Generator().manual_seed(17)
    atlas = torch.randint(256, (5, tile_size, tile_size, 3), generator=rng, dtype=torch.uint8)
    indices = torch.randint(5, (3, 2, 4), generator=rng, dtype=torch.int32)
    expected = tilemap.blit_torch(atlas, indices)
    actual = tilemap.blit(atlas.cuda(), indices.cuda(), tile_size).cpu()
    assert torch.equal(actual, expected)
    transparent = torch.rand((3, view_size, view_size), generator=rng) > 0.3
    expected = tilemap.visibility_scan_torch(transparent, view_size)
    actual = tilemap.visibility_scan(transparent.cuda(), view_size).cpu()
    assert torch.equal(actual, expected)
