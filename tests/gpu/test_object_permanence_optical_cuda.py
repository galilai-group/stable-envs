"""Arbitrary-ray BVH queries, optical laws and full/cached recorded paths."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("optical replay requires CUDA", allow_module_level=True)

from scripts.testing.object_permanence_optical_reference import WorldTree, uniform
from stable_envs.envs.object_permanence.rendering.optical_scene import dielectric, random_event
from stable_envs.envs.object_permanence.rendering.optics import snell
from stable_envs.envs.object_permanence.rendering.optics_cuda import TriangleQuery
from stable_envs.envs.object_permanence.rendering.visibility import local_planes


def test_counter_stream_matches_independent_uint32_reference_and_order():
    ray = np.array([0, 1, 8, 999, 460799, 10000000], np.int64)
    tensor = torch.tensor(ray, device="cuda")
    for event in (0, 1, 31, 127):
        np.testing.assert_array_equal(random_event(tensor, event).cpu(), uniform(ray, event))
        np.testing.assert_array_equal(random_event(tensor.flip(0), event).flip(0).cpu(), uniform(ray, event))


@pytest.mark.parametrize("n1,n2,angle", [(1.0, 1.45, 0.0), (1.0, 1.45, 60.0), (1.45, 1.0, 60.0), (1.45, 1.45, 90.0)])
def test_cuda_snell_and_fresnel_choices_match_analytic_equations(n1, n2, angle):
    a = np.deg2rad(angle)
    d = np.array([[np.sin(a), 0, -np.cos(a)]], np.float64)
    n = np.array([[0, 0, 1]], np.float64)
    reflected, refracted, f = snell(d, n, n1, n2)
    x = torch.tensor(d, device="cuda")
    normal = torch.tensor(n, device="cuda")
    for choice in (0.0, 0.999999):
        actual, reflection = dielectric(
            x,
            normal,
            torch.tensor([n1], dtype=torch.float64, device="cuda"),
            torch.tensor([n2], dtype=torch.float64, device="cuda"),
            torch.tensor([choice], device="cuda"),
        )
        expected = reflected if choice < f[0] else refracted
        np.testing.assert_allclose(actual.cpu(), expected, atol=2e-15, rtol=0)
        assert reflection.item() == bool(choice < f[0])


def test_bvh_arbitrary_parallel_inside_miss_and_shared_edge_rays():
    square = np.array([[[-1, -1, -2], [1, -1, -2], [1, 1, -2]], [[-1, -1, -2], [1, 1, -2], [-1, 1, -2]]], np.float64)
    tri = [square, square.copy()]
    query = TriangleQuery(tri, semantic_ids=[20, 10])
    points = np.array([[0, 0, 0], [0, 0, -3], [2, 0, 0], [-1, -1, 0], [0, 0, 0], [0, 0, -2]], np.float64)
    directions = np.array([[0, 0, -1], [0, 0, 1], [0, 0, -1], [0, 0, -1], [1, 0, 0], [0, 0, -1]], np.float64)
    planes = np.concatenate([local_planes(t.reshape(-1, 3), np.arange(t.size // 3).reshape(-1, 3)) for t in tri])
    t, face, normal = query(
        torch.tensor(points, device="cuda"),
        torch.tensor(directions, device="cuda"),
        torch.zeros(2, 3, device="cuda", dtype=torch.float64),
        torch.eye(3, device="cuda", dtype=torch.float64).expand(2, 3, 3).contiguous(),
        torch.ones(2, device="cuda", dtype=torch.bool),
        torch.tensor(planes, device="cuda"),
    )
    np.testing.assert_array_equal(face.cpu(), [2, 2, -1, 2, -1, -1])
    np.testing.assert_array_equal(t.cpu(), [2, 1, np.inf, 2, np.inf, np.inf])
    np.testing.assert_array_equal(normal[[0, 1, 3]].cpu(), [[0, 0, 1]] * 3)


def test_bvh_random_rays_agree_with_independent_world_space_tree():
    rng = np.random.default_rng(9)
    mesh = rng.uniform(-2, 2, (33, 3, 3))
    mesh[:, :, 2] -= 5
    tree = WorldTree({"a": mesh}, ["a"])
    q = TriangleQuery([mesh])
    origin = rng.uniform(-0.5, 0.5, (117, 3))
    direction = rng.normal(size=(117, 3))
    direction[:, 2] = -abs(direction[:, 2])
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    expected_t, expected_f, _, _ = tree.hit(origin, direction)
    planes = local_planes(mesh.reshape(-1, 3), np.arange(mesh.size // 3).reshape(-1, 3))
    actual_t, actual_f, _ = q(
        torch.tensor(origin, device="cuda"),
        torch.tensor(direction, device="cuda"),
        torch.zeros(1, 3, device="cuda", dtype=torch.float64),
        torch.eye(3, device="cuda", dtype=torch.float64)[None],
        torch.ones(1, device="cuda", dtype=torch.bool),
        torch.tensor(planes, device="cuda"),
    )
    np.testing.assert_array_equal(actual_f.cpu(), expected_f)
    hit = expected_f >= 0
    np.testing.assert_allclose(actual_t.cpu().numpy()[hit], expected_t[hit], atol=1e-13, rtol=0)


@pytest.mark.parametrize("clip_name", ["g01-nominal", "g01-member1", "g03-member1"])
@pytest.mark.parametrize("size", [90, 180])
def test_optical_all_frame_cache_parity_and_cross_batch_reuse(clip_name, size):
    from stable_envs.envs.object_permanence import RigidTrajectory
    from stable_envs.envs.object_permanence.rendering.optics import CONTRACT
    from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
    from tests.helpers.reference.object_permanence_recordings import RECORDINGS

    modes = ("rgb", "depth", "ids", "masks", "path_depth")
    with RigidTrajectory(RECORDINGS, clip_name) as clip:
        with (
            RigidSceneRenderer(clip, size=size, visibility_contract=CONTRACT) as cached,
            RigidSceneRenderer(clip, size=size, visibility_contract=CONTRACT, cache_static=False) as full,
        ):
            for start in range(0, len(clip), 4):
                frames = np.arange(start, min(start + 4, len(clip)))
                a = cached.render_many(frames, modes=modes)
                b = full.render_many(frames, modes=modes)
                assert all(torch.equal(a[k], b[k]) for k in modes), (
                    clip_name,
                    size,
                    start,
                    [k for k in modes if not torch.equal(a[k], b[k])],
                )
            frames = np.array([119, 44, 0, 44, 80])
            a = full.render_many(frames, modes=modes)
            cached.renderer.fuse_cache = False
            reference = cached.render_many(frames, modes=modes)
            assert all(torch.equal(a[k], reference[k]) for k in modes)
            cached.renderer.fuse_cache = True
            for chunk in (1, 3, 5):
                cached.chunk_size = chunk
                b = cached.render_many(frames, modes=modes)
                assert all(torch.equal(a[k], b[k]) for k in modes)
            for mode in modes:
                assert torch.equal(a[mode], cached.render(frames, mode=mode))
            out = {k: torch.empty_like(v) for k, v in a.items()}
            assert cached.render_many(frames, modes=modes, out=out) is out
            assert all(torch.equal(a[k], out[k]) for k in modes)
            assert cached.renderer.builds == 1 and cached.renderer.fallbacks == 0


@pytest.mark.parametrize("task,member", [("G01", 0), ("G01", 1), ("G03", 1)])
@pytest.mark.parametrize("cached", [False, True])
def test_optical_all_frame_independent_oracle(task, member, cached):
    from scripts.testing.check_object_permanence_optical import check
    from tests.helpers.reference.object_permanence_recordings import FIXTURES, RECORDINGS

    directory = (
        FIXTURES
        / f"semantics/object_permanence/{task.lower()}/sampled_optical_rays_v1"
        / ("nominal" if member == 0 else "member1")
    )
    result = check(RECORDINGS, directory, cache_static=cached)
    assert result["passed"] and result["frames"] == 120, result["failures"]


@pytest.mark.parametrize("moving_glass", [False, True])
def test_static_transport_supports_multiple_opaque_movers_and_falls_back_for_moving_glass(moving_glass):
    from copy import deepcopy

    from stable_envs.envs.object_permanence import RigidTrajectory
    from stable_envs.envs.object_permanence.rendering.optics import CONTRACT
    from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
    from tests.helpers.reference.object_permanence_recordings import RECORDINGS

    with RigidTrajectory(RECORDINGS, "g01-nominal") as clip:
        metadata = clip.manifest
        ball = deepcopy(next(o for o in metadata["objects"] if o["node"] == 0))
        ball.update(name="second-mover", id=99, node=1)
        metadata["objects"].append(ball)
        metadata["dynamic_nodes"].append("second-mover")
        if moving_glass:
            ball["rgba"][3] = 0.28
            ball["optics"]["transmission"] = 0.25
            ball["optics"]["ior"] = next(
                o["optics"]["ior"] for o in metadata["objects"] if o["optics"]["transmission"] > 0
            )

        class Scene:
            manifest = metadata

            def mesh(self, key):
                return clip.mesh(key)

            def transforms(self, frames):
                original = clip.transforms(frames)
                second = original.copy()
                second[:, :, 0, 3] += 0.7
                return np.concatenate((original, second), axis=1)

        modes = ("rgb", "depth", "ids", "masks", "path_depth")
        with (
            RigidSceneRenderer(Scene(), size=90, visibility_contract=CONTRACT) as cached,
            RigidSceneRenderer(Scene(), size=90, visibility_contract=CONTRACT, cache_static=False) as full,
        ):
            frames = [119, 44, 0, 119]
            a = cached.render_many(frames, modes=modes)
            b = full.render_many(frames, modes=modes)
            assert all(torch.equal(a[k], b[k]) for k in modes)
            assert cached.renderer.builds == (0 if moving_glass else 1)
            assert cached.renderer.fallbacks == (1 if moving_glass else 0)


def test_cold_optical_replay_imports_no_source_engine_or_rasterizer():
    import subprocess
    import sys

    code = """
import sys
sys.modules['bpy']=None;sys.modules['mujoco']=None;sys.modules['nvdiffrast']=None
from stable_envs.envs.object_permanence import RigidTrajectory
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
from tests.helpers.reference.object_permanence_recordings import RECORDINGS
with RigidTrajectory(RECORDINGS,'g01-nominal') as clip:
 with RigidSceneRenderer(clip,size=90,visibility_contract=clip.manifest['visibility_contract']) as renderer:
  out=renderer.render_many([44],modes=('rgb','masks'))
  assert all(v.is_cuda for v in out.values())
assert all(sys.modules[k] is None for k in ('bpy','mujoco','nvdiffrast'))
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, timeout=90)


def test_optical_mixed_ior_is_rejected_before_building_paths():
    from stable_envs.envs.object_permanence import RigidTrajectory
    from stable_envs.envs.object_permanence.rendering.optics import CONTRACT
    from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
    from tests.helpers.reference.object_permanence_recordings import RECORDINGS

    with RigidTrajectory(RECORDINGS, "g01-nominal") as clip:
        metadata = clip.manifest
        next(o for o in metadata["objects"] if o["optics"]["transmission"] > 0)["optics"]["ior"] = 2.0

        class Scene:
            manifest = metadata

        with pytest.raises(ValueError, match="common dielectric IOR"):
            RigidSceneRenderer(Scene(), size=90, visibility_contract=CONTRACT)


def test_optical_cached_resolve_keeps_semantic_ties_and_blocked_paths():
    import triton

    from stable_envs.envs.object_permanence.rendering.optical_cache_cuda import composite

    device = "cuda"
    n = 5
    ray = torch.arange(n, device=device)
    travel = torch.full((n,), 0.25, device=device, dtype=torch.float64)
    st = torch.tensor([2.0, 2.0, float("inf"), float("inf"), 2.0], device=device, dtype=torch.float64)
    sf = torch.tensor([0, 1, -1, -1, 0], device=device)
    dt = torch.tensor([2.0, 2.0, 1.0, float("inf"), 1.0], device=device, dtype=torch.float64)
    df = torch.tensor([1, 0, 1, -1, 1], device=device)
    normal = torch.tensor([[0.0, 0.0, 1.0]] * n, device=device, dtype=torch.float64)
    geometry = torch.tensor([0, 1], device=device)
    semantics = torch.tensor([20, 10], device=device, dtype=torch.int32)
    labels = semantics.to(torch.int16)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=torch.float64)
    axis = torch.ones(n, device=device, dtype=torch.float64)
    alive = torch.tensor([True, True, True, True, False], device=device)
    rgb = torch.full((n, 3), 0.125, device=device, dtype=torch.float64)
    depth = torch.zeros(n, device=device, dtype=torch.float64)
    ids = torch.zeros(n, device=device, dtype=torch.int16)
    paths = torch.full((2 * n,), float("inf"), device=device, dtype=torch.float64)
    composite[(triton.cdiv(n, 128),)](
        ray,
        travel,
        st,
        sf,
        dt,
        df,
        normal,
        geometry,
        semantics,
        labels,
        colors,
        axis,
        alive,
        rgb,
        depth,
        ids,
        paths,
        n,
        n,
        1,
        True,
        True,
        128,
        enable_fp_fusion=False,
    )
    np.testing.assert_array_equal(ids.cpu(), [10, 10, 10, 0, 0])
    np.testing.assert_array_equal(alive.cpu(), [False, True, False, True, False])
    np.testing.assert_array_equal(depth.cpu(), [2, 2, 1, 0, 0])
    np.testing.assert_array_equal(paths.reshape(2, n)[1, :3].cpu(), [2.25, 2.25, 1.25])
    np.testing.assert_array_equal(rgb[[1, 3, 4]].cpu(), [[0.125] * 3] * 3)


@pytest.mark.parametrize(
    "option", [{"geometry_precision": "invalid"}, {"coverage_mode": "native"}, {"camera_invalid": True}]
)
def test_optical_rejects_incompatible_raster_options_and_invalid_projection(option):
    from stable_envs.envs.object_permanence import RigidTrajectory
    from stable_envs.envs.object_permanence.rendering.optics import CONTRACT
    from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
    from tests.helpers.reference.object_permanence_recordings import RECORDINGS

    with RigidTrajectory(RECORDINGS, "g01-nominal") as clip:
        metadata = clip.manifest
        if option.get("camera_invalid"):
            metadata["camera"]["projection"][0][0] = 0

        class Scene:
            manifest = metadata

        kwargs = {k: v for k, v in option.items() if k != "camera_invalid"}
        with pytest.raises(ValueError):
            RigidSceneRenderer(Scene(), size=90, visibility_contract=CONTRACT, **kwargs)


@pytest.mark.parametrize("clip_name", ["g01-nominal", "g01-member1", "g03-member1"])
@pytest.mark.parametrize("size", [90, 180])
def test_static_hit_ray_limits_preserve_all_fields_and_every_frame(clip_name, size):
    from stable_envs.envs.object_permanence import RigidTrajectory
    from stable_envs.envs.object_permanence.rendering.optics import CONTRACT
    from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
    from tests.helpers.reference.object_permanence_recordings import RECORDINGS

    with RigidTrajectory(RECORDINGS, clip_name) as clip:
        with RigidSceneRenderer(clip, size=size, visibility_contract=CONTRACT) as renderer:
            modes = ("rgb", "depth", "ids", "masks", "path_depth")
            for start in range(0, len(clip), 4):
                frames = np.arange(start, min(start + 4, len(clip)))
                renderer.renderer.use_ray_limits = False
                reference = renderer.render_many(frames, modes=modes)
                renderer.renderer.use_ray_limits = True
                bounded = renderer.render_many(frames, modes=modes)
                assert all(torch.equal(reference[k], bounded[k]) for k in modes), (clip_name, size, start)
            assert renderer.renderer.builds == 1


def test_query_distance_limit_keeps_exact_boundary_hits_and_miss_convention():
    mesh = np.array([[[-1, -1, -2], [1, -1, -2], [0, 1, -2]]], np.float64)
    query = TriangleQuery([mesh])
    n = 3
    origin = torch.zeros(n, 3, device="cuda", dtype=torch.float64)
    direction = torch.tensor([[0.0, 0.0, -1.0]] * n, device="cuda", dtype=torch.float64)
    planes = torch.tensor(local_planes(mesh.reshape(-1, 3), np.arange(3).reshape(-1, 3)), device="cuda")
    t, face, _ = query(
        origin,
        direction,
        torch.zeros(1, 3, device="cuda", dtype=torch.float64),
        torch.eye(3, device="cuda", dtype=torch.float64)[None],
        torch.ones(1, device="cuda", dtype=torch.bool),
        planes,
        limit=torch.tensor([1.0, 2.0, float("inf")], device="cuda", dtype=torch.float64),
    )
    np.testing.assert_array_equal(face.cpu(), [-1, 0, 0])
    np.testing.assert_array_equal(t.cpu(), [np.inf, 2.0, 2.0])


@pytest.mark.parametrize("layout", ["broadcast", "strided", "transposed"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_query_ray_views_match_contiguous_inputs(layout, dtype):
    mesh = np.array([[[-1, -1, -2], [1, -1, -2], [0, 1, -2]]], np.float64)
    query = TriangleQuery([mesh])
    n = 5
    origin = torch.zeros(n, 3, device="cuda", dtype=dtype)
    direction = torch.tensor([[0.0, 0.0, -1.0]] * n, device="cuda", dtype=dtype)
    if layout == "broadcast":
        origin = origin[:1].expand(n, 3)
        direction = direction[:1].expand(n, 3)
    elif layout == "strided":
        origin = origin.repeat_interleave(2, 0)[::2]
        direction = direction.repeat_interleave(2, 0)[::2]
    else:
        origin = origin.T.contiguous().T
        direction = direction.T.contiguous().T
    args = (
        torch.zeros(1, 3, device="cuda", dtype=torch.float64),
        torch.eye(3, device="cuda", dtype=torch.float64)[None],
        torch.ones(1, device="cuda", dtype=torch.bool),
        torch.tensor(local_planes(mesh.reshape(-1, 3), np.arange(3).reshape(-1, 3)), device="cuda"),
    )
    actual = query(origin, direction, *args)
    expected = query(origin.contiguous(), direction.contiguous(), *args)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert actual[2].shape == (n, 3) and actual[2].is_contiguous()
    assert actual[2].dtype == dtype


def test_reconstructed_normals_match_query_for_affine_transforms_and_misses():
    mesh = np.array([[[-1, -1, -2], [1, -1, -2], [0, 1, -1]]], np.float64)
    query = TriangleQuery([mesh])
    matrix = np.array([[1.3, -0.2, 0.1], [0.4, 0.8, 0.2], [-0.1, 0.3, 1.1]], np.float64)
    world = mesh @ matrix.T
    centre = world.mean(1)
    normal = np.cross(world[:, 1] - world[:, 0], world[:, 2] - world[:, 0])
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    origin = np.concatenate((centre + 2 * normal, centre - 2 * normal, centre + 100))
    direction = np.concatenate((-normal, normal, normal))
    inverse = torch.tensor(np.linalg.inv(matrix)[None], device="cuda")
    _, face, expected = query(
        torch.tensor(origin, device="cuda"),
        torch.tensor(direction, device="cuda"),
        torch.zeros(1, 3, device="cuda", dtype=torch.float64),
        inverse,
        torch.ones(1, device="cuda", dtype=torch.bool),
        torch.tensor(local_planes(world.reshape(-1, 3), np.arange(3).reshape(-1, 3)), device="cuda"),
    )
    assert face.tolist() == [0, 0, -1]
    actual = query.normals(face, inverse)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert query.normals(face[:0], inverse).shape == (0, 3)


@pytest.mark.parametrize("clip_name", ["g01-nominal", "g01-member1"])
@pytest.mark.parametrize("size", [90, 180])
def test_retained_optical_storage_stays_compact_across_diagnostic_and_fused_calls(clip_name, size):
    from benchmarks.object_permanence_optical import storage_bytes
    from stable_envs.envs.object_permanence import RigidTrajectory
    from stable_envs.envs.object_permanence.rendering.optics import CONTRACT
    from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
    from tests.helpers.reference.object_permanence_recordings import RECORDINGS

    with RigidTrajectory(RECORDINGS, clip_name) as clip:
        with RigidSceneRenderer(clip, size=size, visibility_contract=CONTRACT) as renderer:
            modes = ("rgb", "depth", "ids", "masks", "path_depth")
            reference = renderer.render_many([44, 119], modes=modes)
            baseline = storage_bytes(renderer.renderer)
            # Previous duplicate primary rays and retained static normals use
            # over 28 MiB here. Leave headroom for scene metadata, not duplication.
            assert sum(baseline.values()) < 20 * (size / 90) ** 2 * 2**20
            renderer.renderer.fuse_cache = False
            diagnostic = renderer.render_many([44, 119], modes=modes)
            assert all(torch.equal(reference[k], diagnostic[k]) for k in modes)
            assert storage_bytes(renderer.renderer) == baseline
            renderer.renderer.fuse_cache = True
            repeated = renderer.render_many([119, 44], modes=modes)
            assert all(torch.equal(reference[k].flip(0), repeated[k]) for k in modes)
            assert storage_bytes(renderer.renderer) == baseline
            for start in range(0, len(clip), 16):
                frames = np.arange(start, min(start + 16, len(clip)))
                fused = renderer.render_many(frames, modes=modes)
                renderer.renderer.fuse_cache = False
                diagnostic = renderer.render_many(frames, modes=modes)
                assert all(torch.equal(fused[k], diagnostic[k]) for k in modes), (clip_name, size, start)
                renderer.renderer.fuse_cache = True
                assert storage_bytes(renderer.renderer) == baseline
            assert renderer.renderer.builds == 1
