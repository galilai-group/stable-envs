"""CUDA uses the unchanged Blender gates, independent triangles and CPU State."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu
pytest.importorskip("mujoco")
if not torch.cuda.is_available():
    pytest.skip("G34 CUDA checks require an available GPU", allow_module_level=True)

import stable_envs as se
from tests.helpers.reference.semantics import assert_same_state
from tests.regression import test_occlusion_ball_geometry as geometry_checks
from tests.unit import test_object_permanence_uv_sphere as sphere_checks


@pytest.mark.parametrize("iterations", [0, 8])
@pytest.mark.parametrize("slices,stacks", [(48, 24), (12, 7)])
def test_cuda_facets_against_independent_triangles(iterations, slices, stacks):
    sphere_checks.test_uv_sphere_matches_triangles_for_grazing_polar_and_inside_rays(
        iterations, slices, stacks, device="cuda"
    )


def test_cuda_against_frozen_blender_geometry(record_property):
    geometry_checks.test_source_depth_masks_and_occlusion_boundaries(record_property, "raycast", device="cuda")


def test_cuda_geometry_negative_controls():
    geometry_checks.test_geometry_gate_detects_shifted_motion_and_wrong_depth("raycast", device="cuda")


@pytest.mark.parametrize("size", [90, 180])
def test_cpu_replay_upload_batch_independence_cache_and_restore(size, monkeypatch):
    from stable_envs.envs.object_permanence.rendering.g34_native import _bundle

    with (
        se.MuJoCoOcclusionBall(backend="raycast", size=size) as cpu,
        se.MuJoCoOcclusionBall(backend="raycast", size=size, device="cuda") as gpu,
    ):
        initial = cpu.reset()
        rollout = cpu.step(np.zeros((1, 119, 1)), frames=None)["states"]
        bank = torch.cat([initial, rollout.squeeze(0)], 0)
        held, live = bank.clone(), gpu.get_state()
        expected, actual = cpu.geometry(bank), gpu.geometry(bank)
        assert torch.equal(expected["object_ids"], actual["object_ids"].cpu())
        torch.testing.assert_close(actual["depth"].cpu(), expected["depth"], rtol=0, atol=1e-6)
        images = gpu.render(bank)
        assert images.device.type == "cuda" and images.dtype == torch.uint8
        assert torch.equal(cpu.render(bank), images.cpu())
        for frame in (0, 44, 59, 75, 119):
            assert torch.equal(gpu.render(bank[frame : frame + 1])[0], images[frame])
        output = torch.empty_like(images)
        renderer = _bundle(size, size * 16 // 9)[2]

        def forbidden_draw(*args, **kwargs):
            raise AssertionError("warm CUDA rendering must reuse the static bake")

        monkeypatch.setattr(renderer, "render", forbidden_draw)
        assert gpu.render(bank, out=output) is output
        assert torch.equal(output, images)
        images.zero_()
        assert torch.equal(gpu.render(bank), output)
        with pytest.raises(ValueError, match="out"):
            gpu.render(bank, out=output.cpu())
        assert_same_state(bank, held)
        assert_same_state(gpu.get_state(), live)
        gpu.set_state(bank[44:45])
        cpu.set_state(bank[44:45])
        assert_same_state(gpu.get_state().cpu(), cpu.get_state())
        assert_same_state(gpu.step(np.zeros((1, 7, 1)), frames=None).cpu(), cpu.step(np.zeros((1, 7, 1)), frames=None))


def test_cuda_torch_fallback_uses_same_geometry(monkeypatch):
    from stable_envs.envs.object_permanence.rendering import g34_raycast
    from stable_envs.render import uv_sphere

    with se.MuJoCoOcclusionBall(device="cuda", backend="raycast") as env:
        state = env.step(np.zeros((1, 44, 1)), frames=None)["state"]
        expected = env.geometry(state)
        rgb = env.render(state)
        monkeypatch.setattr(g34_raycast, "HAVE_TRITON", False)
        monkeypatch.setattr(uv_sphere, "triton", None)
        actual = env.geometry(state)
        assert torch.equal(actual["object_ids"], expected["object_ids"])
        torch.testing.assert_close(actual["depth"], expected["depth"], rtol=0, atol=1e-6)
        assert torch.equal(env.render(state), rgb)


def test_cuda_handles_non_nominal_poses_and_camera_inside_sphere():
    from stable_envs.envs.object_permanence.rendering.g34_raycast import _plate

    with (
        se.MuJoCoOcclusionBall(backend="raycast", num_envs=4) as cpu,
        se.MuJoCoOcclusionBall(backend="raycast", num_envs=4, device="cuda") as gpu,
    ):
        state = cpu.reset()
        offset = cpu._simulator.pose_offset
        rows = state["physics", "integration"]
        camera = torch.tensor(_plate(90, 160)["camera"].pos)
        # Arbitrary poses are still valid State. Exercise a camera-plane crossing,
        # inside rays, a ball behind the camera and a non-Y rotation.
        rows[:, offset : offset + 3] = torch.stack(
            (
                camera,
                camera + camera.new_tensor([0, 0.04, 0]),
                camera + camera.new_tensor([0, -1, 0]),
                camera.new_tensor([-1, 0, 0.194]),
            )
        )
        quaternion = rows.new_tensor([0.3, -0.4, 0.2, 0.5])
        rows[:, offset + 3 : offset + 7] = quaternion / torch.linalg.vector_norm(quaternion)
        expected, actual = cpu.geometry(state), gpu.geometry(state)
        assert torch.equal(actual["object_ids"].cpu(), expected["object_ids"])
        torch.testing.assert_close(actual["depth"].cpu(), expected["depth"], rtol=0, atol=1e-6)
        assert torch.equal(gpu.render(state).cpu(), cpu.render(state))
