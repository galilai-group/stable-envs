"""Exact ray seeds retain nearer movers, semantic ties and invalidation semantics."""

import weakref

import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("ray cache requires CUDA", allow_module_level=True)
pytest.importorskip("nvdiffrast.torch")

from stable_envs.envs.object_permanence.rendering.visibility_cache import StaticRayCoverage
from stable_envs.envs.object_permanence.rendering.visibility_cuda import _visibility
from tests.unit.test_object_permanence_ray_contract import camera


def inputs(partition, size=(16, 24), batch=4):
    h, w = size
    face = torch.tensor([[-1.0, -1.0, -3.0], [1.0, -1.0, -3.0], [0.0, 1.0, -3.0]], device="cuda", dtype=torch.float64)
    vertices = face.repeat(4, 1)[None].repeat(batch, 1, 1)
    planes = torch.tensor([[0.0, 0.0, 1.0, 3.0]] * 4, device="cuda", dtype=torch.float64)[None].repeat(batch, 1, 1)
    for f in range(4):
        if not partition[f]:
            shift = torch.tensor([0.5, -0.5, 0.0, 0.25], device="cuda", dtype=torch.float64)[:batch]
            vertices[:, f * 3 : f * 3 + 3, 2] += shift[:, None]
            planes[:, f, 3] -= shift
    p = torch.tensor(camera()["projection"], device="cuda", dtype=torch.float64)
    pos = torch.cat((vertices, torch.ones_like(vertices[..., :1])), -1) @ p.T
    y, x = torch.meshgrid(
        torch.arange(h, device="cuda", dtype=torch.float64),
        torch.arange(w, device="cuda", dtype=torch.float64),
        indexing="ij",
    )
    rays = torch.stack((2 * (x + 0.5) / w - 1, 2 * (y + 0.5) / h - 1, -torch.ones_like(x)), -1)
    return {
        "positions": pos.contiguous(),
        "vertices": vertices,
        "planes": planes,
        "rays": rays,
        "origin": torch.zeros(3, device="cuda", dtype=torch.float64),
        "labels": torch.tensor([40, 3, 70, 2], device="cuda", dtype=torch.int32),
        "resolution": size,
        "near": 0.1,
        "far": 10.0,
    }


def visibility(data, faces, cache=None):
    def draw(pos, vertices, planes, depth, winner, active=None, background=None):
        _visibility(
            pos,
            vertices,
            planes,
            faces,
            data["rays"],
            data["origin"],
            data["labels"],
            depth,
            winner,
            data["resolution"],
            data["near"],
            data["far"],
            active,
            background,
        )

    seed = None if cache is None else cache.seed(faces, **data, draw=draw)
    if seed is None:
        depth = torch.full(
            (len(data["positions"]), *data["resolution"]),
            torch.iinfo(torch.int64).max,
            device="cuda",
            dtype=torch.int64,
        )
        winner = torch.full_like(depth, torch.iinfo(torch.int64).max)
        active = background = None
    else:
        depth, winner, active, background = seed
    draw(data["positions"], data["vertices"], data["planes"], depth, winner, active, background)
    return depth, winner


@pytest.mark.parametrize(
    "partition", [[True, False, True, False], [False, True, False, True], [True] * 4, [False] * 4]
)
def test_interleaved_static_dynamic_nearer_semantic_ties_and_owned_seeds(partition):
    data = inputs(partition)
    faces = torch.arange(12, device="cuda", dtype=torch.int32).reshape(4, 3)
    cache = StaticRayCoverage(faces, torch.tensor(partition, device="cuda"))
    expected = visibility(data, faces)
    actual = visibility(data, faces, cache)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
    for value in actual:
        value.zero_()
    again = visibility(data, faces, cache)
    assert all(torch.equal(a, b) for a, b in zip(expected, again, strict=True))
    assert cache.builds == int(any(partition))
    storage = weakref.ref(cache._depth) if any(partition) else None
    cache.clear()
    assert storage is None or storage() is None
    assert cache._snapshot is None and cache._winner is None


def test_static_seed_invalidates_on_geometry_planes_ray_origin_labels_clipping_and_resolution():
    partition = [True, False, True, False]
    data = inputs(partition)
    faces = torch.arange(12, device="cuda", dtype=torch.int32).reshape(4, 3)
    cache = StaticRayCoverage(faces, torch.tensor(partition, device="cuda"))

    def check():
        expected = visibility(data, faces)
        actual = visibility(data, faces, cache)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))

    check()
    for key in ("positions", "vertices", "planes", "rays", "origin", "labels", "near", "resolution"):
        if key == "positions":
            data[key][:, 0, 0] += 0.01
        elif key == "vertices":
            data[key][:, 0, 0] += 0.01
        elif key == "planes":
            data[key][:, 0, 3] += 0.001
        elif key == "rays":
            data[key][..., 0] += 0.001
        elif key == "origin":
            data[key][0] += 0.001
        elif key == "labels":
            data[key][0] = 1
        elif key == "near":
            data[key] = 0.2
        else:
            data.update(inputs(partition, size=(17, 25)))
        check()
    assert cache.builds == 9
    # A supposedly static vertex varying within a batch uses complete coverage.
    data["vertices"][1, 0, 0] += 0.01
    check()
    assert cache.fallbacks == 1 and cache.builds == 9
    with pytest.raises(ValueError, match="immutable face"):
        visibility(data, faces.clone(), cache)
    faces[0, 0] = faces[0, 0]
    with pytest.raises(ValueError, match="immutable face"):
        visibility(data, faces, cache)
