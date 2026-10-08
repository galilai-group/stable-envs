import pytest
import torch

from stable_envs.render import raycast3d as R3
from tests.helpers.raycast_reference import composite_window
from tests.unit.test_raycast3d_composite_compact import _scene, _split
from tests.unit.test_raycast3d_primitive_bounds import _rand_scene

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def cuda():
    if not torch.cuda.is_available() or not R3.HAVE_TRITON:
        pytest.skip("CUDA and Triton are required for raycast dispatch tests")


def _cuda_scene(kinds=R3.PRIMITIVE_KINDS):
    kinds, *args = _rand_scene(rng=521, B=3, K=len(kinds), N=513, kinds=kinds)
    return kinds, *(t.cuda() for t in args)


@pytest.mark.parametrize("kind", R3.PRIMITIVE_KINDS)
def test_each_primitive_matches_torch(kind):
    args = _cuda_scene((kind,) * 4)
    t, n, idx = R3.cast_nearest(*args, backend="triton")
    rt, rn, ridx = R3.cast_nearest(*args, backend="torch")
    assert torch.equal(t, rt) and torch.equal(idx, ridx)
    hit = t < R3.INF
    assert hit.any(), "the scene did not exercise a hit"
    assert torch.equal(n[hit], rn[hit])
    lim = torch.full_like(t, 3.0)
    for skip in (None, idx):
        actual = R3.any_blocking(*args, lim, skip, backend="triton")
        expected = R3.any_blocking(*args, lim, skip, backend="torch")
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("blocking", [False, True])
@pytest.mark.parametrize("layout", ["float64", "strided", "transposed"])
def test_unsupported_explicit_triton_inputs_fail_and_auto_matches_torch(blocking, layout):
    args = list(_cuda_scene())
    if layout == "float64":
        args[1:] = [t.double() for t in args[1:]]
        reason = "float32"
    elif layout == "strided":
        origins = args[-2]
        padded = torch.empty(*origins.shape[:-1], 6, device="cuda")
        padded[..., ::2] = origins
        args[-2] = padded[..., ::2]
        reason = "packed"
    else:
        args[3] = args[3].transpose(-1, -2)
        reason = "packed"
    fn = R3.any_blocking if blocking else R3.cast_nearest
    if blocking:
        args.append(torch.full(args[-1].shape[:2], 3.0, dtype=args[-1].dtype, device="cuda"))
    with pytest.raises(ValueError, match=reason):
        fn(*args, backend="triton")
    actual, expected = fn(*args), fn(*args, backend="torch")
    if blocking:
        assert torch.equal(actual, expected)
    else:
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize("supersample", [1, 2])
@pytest.mark.parametrize("tile", [None, 16])
def test_gpu_composite_matches_both_window_references(supersample, tile):
    sc, _ = _split(_scene(seed=81, B=5, K=1, kinds=(R3.SPHERE,)))
    sc = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in sc.items()}
    sc["supersample"] = supersample
    actual = R3.composite(**sc, tile=tile)
    for compact in (False, True):
        expected = composite_window(**sc, tile=tile, cover_compaction=compact)
        assert torch.equal(actual, expected)
