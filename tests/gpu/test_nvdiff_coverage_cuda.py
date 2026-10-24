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


@pytest.mark.parametrize("partition", [[True, False, True, False], [False] * 4, [True] * 4])
def test_static_visibility_preserves_closer_movers_depth_ties_and_multiple_movers(partition):
    from stable_envs.render.nvdiff_coverage import StaticCoverage

    triangle = [[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [0, 0.8, 0, 1]]
    positions = torch.tensor([triangle * 4] * 4, dtype=torch.float32, device="cuda")
    faces = torch.arange(12, device="cuda", dtype=torch.int32).reshape(4, 3)
    mask = torch.tensor(partition, device="cuda")
    # Include front/behind, interpenetration and exact-depth ties. Static faces
    # interleave movers in original order; a nearer lower ID must replace them.
    for face in range(4):
        if not partition[face]:
            positions[:, face * 3 : face * 3 + 3, 2] = torch.tensor([-0.3, 0.3, 0, -0.1], device="cuda")[:, None]
    cache = StaticCoverage(faces, mask)
    for pos in (positions, positions.flip(0).contiguous(), positions[:1]):
        expected = rasterize(pos, faces, (16, 24))
        actual = rasterize(pos, faces, (16, 24), cache=cache)
        assert torch.equal(actual, expected)
        # The raster is caller-owned; changing it must not poison the seed.
        actual.zero_()
        assert torch.equal(rasterize(pos, faces, (16, 24), cache=cache), expected)
    assert cache.builds == int(any(partition))


def test_static_cache_rebuilds_on_camera_geometry_resolution_changes_and_falls_back_safely():
    from stable_envs.render.nvdiff_coverage import StaticCoverage

    pos = torch.tensor(
        [
            [
                [-0.8, -0.8, 0, 1],
                [0.8, -0.8, 0, 1],
                [0, 0.8, 0, 1],
                [-0.5, -0.5, -0.1, 1],
                [0.5, -0.5, -0.1, 1],
                [0, 0.5, -0.1, 1],
            ]
        ],
        device="cuda",
    )
    faces = torch.arange(6, device="cuda", dtype=torch.int32).reshape(2, 3)
    cache = StaticCoverage(faces, torch.tensor([True, False], device="cuda"))
    for size, offset in (((16, 24), 0), ((16, 24), 0.2), ((32, 48), 0.2)):
        changed = pos.clone()
        changed[..., 0] += offset
        assert torch.equal(rasterize(changed, faces, size, cache=cache), rasterize(changed, faces, size))
    assert cache.builds == 3
    moving_background = pos.expand(2, -1, -1).clone()
    moving_background[1, :3, 0] += 0.3
    assert torch.equal(
        rasterize(moving_background, faces, (16, 24), cache=cache),
        rasterize(moving_background, faces, (16, 24)),
    )
    assert cache.builds == 3
    clipped = pos.clone()
    clipped[0, 0, 2] = -2
    assert rasterize(clipped, faces, (16, 24), cache=cache) is None
    assert torch.equal(rasterize(pos, faces, (16, 24), cache=cache), rasterize(pos, faces, (16, 24)))
    with pytest.raises(ValueError, match="different face"):
        rasterize(pos, faces.clone(), (16, 24), cache=cache)


def test_double_clip_input_recovers_information_already_lost_in_float32():
    from stable_envs.render.nvdiff_coverage import StaticCoverage

    pos = torch.tensor(
        [[[0.0625 - 1e-10, 0.0615, 0, 1], [0.0625 + 1e-10, 0.0615, 0, 1], [0.0625, 0.0635, 0, 1]]],
        dtype=torch.float64,
        device="cuda",
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    assert rasterize(pos.float(), faces, (16, 16))[0, 8, 8, 3] == 0
    exact = rasterize(pos, faces, (16, 16))
    assert exact.dtype == torch.float32 and exact[0, 8, 8, 3] == 1
    cache = StaticCoverage(faces, torch.tensor([True], device="cuda"))
    assert torch.equal(rasterize(pos, faces, (16, 16), cache=cache), exact)
    # torch.equal ignores dtype: equal represented coordinates still need a seed
    # rebuild when switching projection precision, then reuse within that mode.
    rounded = pos.float()
    rasterize(rounded, faces, (16, 16), cache=cache)
    assert cache.builds == 2
    rasterize(rounded.double(), faces, (16, 16), cache=cache)
    assert cache.builds == 3
    rasterize(rounded.double(), faces, (16, 16), cache=cache)
    assert cache.builds == 3


def test_double_projection_avoids_transform_cancellation_and_releases_cached_storage():
    import weakref

    from stable_envs.render.nvdiff import Scene

    vertices = np.array([[65536, 65536, -3], [65538, 65536, -3], [65536, 65538, -3]], np.float32)
    scene = Scene.from_triangles(vertices, [[0, 0, 1]] * 3, [[0, 1, 2]], [0, 0, 0], [[1, 0, 0, 1]])
    pos = torch.tensor([[[-65536, -65536, 0]]], device="cuda", dtype=torch.float32)
    rot = torch.eye(3, device="cuda")[None, None]
    view = torch.eye(4, dtype=torch.float64, device="cuda")[None]
    view[0, :2, 3] = torch.tensor([0.123456789, 0.234567891], device="cuda", dtype=torch.float64)
    proj = torch.diag(torch.tensor([1.123456789, 1.234567891, 1, 1], dtype=torch.float64, device="cuda"))
    old, normals, world = scene.clip(pos, rot, view.float(), proj.float())
    precise, normals64, world64 = scene.clip(pos, rot, view, proj, coverage_dtype=torch.float64)
    points = np.column_stack((vertices.astype(np.float64) + [-65536, -65536, 0], np.ones(3)))
    expected = points @ view[0].cpu().numpy().T @ proj.cpu().numpy().T
    np.testing.assert_allclose(precise[0].cpu().numpy(), expected, atol=1e-8, rtol=0)
    assert np.max(abs(old[0].cpu().numpy() - expected)) > 1e-3
    assert torch.equal(normals64, normals) and torch.equal(world64, world)
    resource = weakref.ref(scene.S4_64)
    repeated, _, _ = scene.clip(
        pos.expand(2, -1, -1), rot.expand(2, -1, -1, -1), view, proj, coverage_dtype=torch.float64
    )
    assert torch.equal(repeated, precise.expand_as(repeated))
    scene.close()
    assert resource() is None and scene.S4_64 is None
