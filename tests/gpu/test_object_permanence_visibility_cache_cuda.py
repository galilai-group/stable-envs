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


def test_one_ulp_plane_change_still_invalidates_the_exact_static_seed():
    data = inputs([True, False, True, False])
    faces = torch.arange(12, device="cuda", dtype=torch.int32).reshape(4, 3)
    cache = StaticRayCoverage(faces, torch.tensor([True, False, True, False], device="cuda"))
    visibility(data, faces, cache)
    offset = data["planes"][:, 0, 3]
    offset.copy_(torch.nextafter(offset, torch.full_like(offset, float("inf"))))
    actual = visibility(data, faces, cache)
    expected = visibility(data, faces)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
    assert cache.builds == 2 and cache.fallbacks == 0


@pytest.mark.parametrize("count", [7, 41])
@pytest.mark.parametrize("moving", [False, True])
def test_rotated_static_geometry_reuses_exact_seed_across_batch_shapes_and_real_changes(count, moving):
    import numpy as np

    from stable_envs.envs.object_permanence.rendering.transforms import rotations
    from stable_envs.envs.object_permanence.rendering.visibility_cuda import ContractRenderer
    from stable_envs.render.nvdiff import Camera, Scene

    # No source scene or frozen data: arbitrary triangle planes and rotations
    # expose shape-dependent three-term GEMM rounding even with seven objects.
    rng = np.random.default_rng(913)
    vertices = rng.uniform(-0.35, 0.35, (count * 3, 3)).astype(np.float32)
    faces = np.arange(count * 3, dtype=np.int32).reshape(count, 3)
    positions = torch.tensor(
        np.column_stack((rng.uniform(-0.5, 0.5, (count, 2)), np.full(count, -4.0))),
        device="cuda",
        dtype=torch.float64,
    )
    matrices = rotations(torch.tensor(rng.normal(size=(count, 4)), device="cuda", dtype=torch.float64))
    static = torch.ones(count, device="cuda", dtype=torch.bool)
    static[-1] = not moving
    world = np.eye(4)
    c, s = np.cos(0.23), np.sin(0.23)
    world[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    world[:3, 3] = [0.15, -0.2, 0.1]
    metadata = camera()
    metadata["world"] = world.tolist()
    with Scene.from_triangles(
        vertices,
        np.tile([0, 0, 1], (count * 3, 1)),
        faces,
        np.repeat(np.arange(count), 3),
        np.tile([1, 0, 0, 1], (count, 1)),
    ) as scene:
        options = dict(
            scene=scene,
            res=(19, 31),
            textures=False,
            mipmap=False,
            skybox=False,
            shading="tiny",
            coverage_mode="refined",
            geometry_precision="float64",
            camera_metadata=metadata,
            semantic_ids=torch.arange(1, count + 1, device="cuda"),
        )
        with ContractRenderer(**options, static_geoms=static) as cached, ContractRenderer(**options) as full:
            cam = Camera(world[:3, 3], world[:3, :3], 90, 0.1, 10)
            cam.matrices = lambda **kwargs: (np.linalg.inv(world)[None], np.asarray(metadata["projection"]))
            cache = cached._ray_coverage
            first = None

            def compare(batch, *, shift=0.0, vary=False):
                nonlocal first
                pos = positions[None].expand(batch, -1, -1).clone()
                rot = matrices[None].expand(batch, -1, -1, -1).contiguous()
                pos[:, 0, 0] += shift
                if moving:
                    pos[:, -1, 0] += torch.arange(batch, device="cuda") * 0.01
                if vary:
                    pos[1, 0, 0] += 0.02
                expected = full.render_many(pos, rot, cam)
                actual = cached.render_many(pos, rot, cam)
                assert all(torch.equal(actual[key], expected[key]) for key in actual)
                if shift == 0 and not vary:
                    if first is None:
                        first = {key: value[:1].clone() for key, value in actual.items()}
                    else:
                        assert all(torch.equal(actual[key][:1], first[key]) for key in actual)

            for batch in (4, 1, 4, 2, 3, 5, 1):
                compare(batch)
                assert cache.builds == 1 and cache.fallbacks == 0
            compare(3, shift=0.03)
            assert cache.builds == 2
            compare(1)
            assert cache.builds == 3
            compare(3, vary=True)
            assert cache.fallbacks == 1 and cache.builds == 3
            compare(1)
            assert cache.builds == 3
