"""Precision coverage must recover absent thin faces and stable occlusion."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("coverage requires CUDA", allow_module_level=True)
pytest.importorskip("triton")
dr = pytest.importorskip("nvdiffrast.torch")

from stable_envs.render.nvdiff_coverage import rasterize


def test_thin_triangle_missing_from_native_coverage_is_recovered_with_correct_weights():
    h, w = 32, 48
    # A real nonzero-area face straddles one center but collapses under the
    # native 1/16-pixel quantization. A larger face sits behind it.
    xy = np.array([[20.5001, 12.49], [20.5001, 12.51], [20.4999, 12.5], [2, 2], [45, 2], [20, 30]])
    vertices = np.column_stack([xy * [2 / w, 2 / h] - 1, [-0.5] * 3 + [0.2] * 3, np.ones(6)])
    pos = torch.tensor(vertices[None], dtype=torch.float32, device="cuda").contiguous()
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device="cuda")
    context = dr.RasterizeCudaContext()
    native, _ = dr.rasterize(context, pos, faces, resolution=[h, w], grad_db=False)
    assert native[0, 12, 20, 3] != 1
    precise = rasterize(pos, faces, (h, w))
    assert precise[0, 12, 20, 3] == 1
    p, q = precise[0, 12, 20, :2].double()
    reconstructed = (pos[0, 0] * p + pos[0, 1] * q + pos[0, 2] * (1 - p - q)).double()
    torch.testing.assert_close(
        reconstructed[:2],
        torch.tensor([2 * 20.5 / w - 1, 2 * 12.5 / h - 1], device="cuda"),
        rtol=0,
        atol=2e-8,
        check_dtype=False,
    )
    assert precise[0, 12, 20, 2] == -0.5
    repeated = rasterize(pos.expand(3, -1, -1).contiguous(), faces, (h, w))
    assert torch.equal(repeated, precise.expand_as(repeated))


def test_equal_depth_tie_is_deterministic_and_clipping_requests_fallback():
    positions = torch.tensor(
        [[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [0, 0.8, 0, 1]]], dtype=torch.float32, device="cuda"
    )
    faces = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int32, device="cuda")
    out = rasterize(positions, faces, (16, 24))
    assert out[0, 5, 12, 3] == 2
    positions[0, 0, 2] = -2
    assert rasterize(positions, faces, (16, 24)) is None


def test_far_offscreen_triangles_do_not_overflow_pixel_bounds():
    positions = torch.tensor([[[1e20, 0, 0, 1], [2e20, 1, 0, 1], [2e20, -1, 0, 1]]], device="cuda")
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    assert not rasterize(positions, faces, (16, 24)).any()
