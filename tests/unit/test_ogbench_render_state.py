"""CPU validation of the named renderer inputs and configured model-provider boundary."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("ogbench")
from stable_envs._adapters.ogbench import render_state as pixels
from stable_envs.envs.ogbench import OGBCube, OGBScene
from stable_envs.envs.ogbench import manipulation as core
from stable_envs.envs.ogbench import manipulation_renderer as renderer


def test_scene_samples_match_opengl_positions_and_plate():
    from stable_envs.envs.ogbench.manipulation_fidelity import rotate_four_samples, sample_plate

    # NVIDIA GL_SAMPLE_POSITION, converted from bottom-up to top-down rows.
    positions = torch.tensor([[[0.125, 0.375], [0.625, 0.125]], [[0.375, 0.875], [0.875, 0.625]]])
    y, x = torch.meshgrid(torch.tensor([0.25, 0.75]), torch.tensor([0.25, 0.75]), indexing="ij")
    rays = torch.stack((x, y, -torch.ones_like(x)), dim=-1)
    rays /= torch.linalg.vector_norm(rays, dim=-1, keepdim=True)
    rotate_four_samples(rays)
    torch.testing.assert_close((rays / -rays[..., 2:3])[..., :2], positions)
    y, x = torch.meshgrid(torch.arange(4) / 4 + 0.125, torch.arange(4) / 4 + 0.125, indexing="ij")
    plate = torch.stack((x, y, torch.zeros_like(x)), dim=-1)
    torch.testing.assert_close(sample_plate(plate)[..., :2], positions)


def test_scene_sampling_has_separate_model_cache():
    from tests.helpers.ogbench_golden import FIXTURES, load

    state, _ = load(FIXTURES / "Scene-mjwarp.npz")
    regular_key, regular = renderer.configured_bundle("Scene", state[0])
    sampled_key, sampled = renderer.configured_bundle("Scene", state[0], "gl4")
    assert regular_key != sampled_key and regular is not sampled
    assert regular.sample_pattern == "grid" and sampled.sample_pattern == "gl4"
    assert renderer.configured_bundle("Scene", state[0]) == (regular_key, regular)


@pytest.mark.parametrize("family", ["Cube", "Scene"])
def test_configuration_groups_preserve_exact_static_keys(family):
    from tests.helpers.ogbench_golden import FIXTURES, load

    state, _ = load(FIXTURES / f"{family}-mjwarp.npz")
    keys = (*pixels.STATIC, *(("lock_color",) if family == "Scene" else ()))
    for index, key in enumerate(keys):
        state["appearance", key][index].reshape(-1)[0] += 0.001
    # These per-world inputs must not create new compiled-model groups.
    state["pose", "qpos"][-1, 0] += 0.1
    state["appearance", "cube.color"][-1] = 0.2
    expected = {}
    for index, row in enumerate(state):
        expected.setdefault(renderer.configuration_key(family, row), []).append(index)
    assert list(renderer._configuration_groups(family, state)) == list(expected.values())
    assert len(expected) == len(keys) + 1


@pytest.mark.parametrize("family,cls", [("Cube", OGBCube), ("Scene", OGBScene)])
def test_named_render_state_boundary(family, cls):
    with cls(num_envs=3, seed=17) as env:
        full = env.get_state()
        saved = full.clone()
        image_state = env.to_render_state(full)
        pixels.validate(family, image_state)
        assert image_state.batch_size == torch.Size([3])
        assert set(image_state.keys(True, True)) == set(pixels.fields(family))
        assert not any(
            k[0] in ["physics", "rng", "task", "lifecycle", "controller", "output"]
            for k in image_state.keys(True, True)
        )
        image_state["pose", "qpos"][0, 0] += 0.1
        assert torch.equal(full["physics", "qpos"], saved["physics", "qpos"])
        with pytest.raises(ValueError):
            env.set_state(image_state)
        with pytest.raises(NotImplementedError, match="CPU mesh"):
            env.render(image_state, backend="mjwarp")
        with pytest.raises(ValueError, match="backend"):
            env.render(image_state, backend="osmesa")
        for key in image_state.keys(True, True):
            bad = image_state.clone()
            del bad[key]
            with pytest.raises(ValueError):
                pixels.validate(family, bad)
            bad = image_state.clone()
            bad[key] = bad[key].float()
            if image_state[key].dtype != torch.float32:
                with pytest.raises(ValueError):
                    pixels.validate(family, bad)
        for key, value in [
            (("renderer", "signature"), 0),
            (("renderer", "camera"), 1),
            (("renderer", "size"), 0),
            (("renderer", "supersample"), 0),
            (("appearance", "cube.size"), 0.3),
            (("appearance", "target_alpha"), 0),
            (("appearance", "camera.angle_delta"), 30),
            (("appearance", "light.intensity"), 2),
            (("pose", "mocap_quat"), 0),
        ]:
            bad = image_state.clone()
            bad[key][-1] = value
            with pytest.raises(ValueError):
                pixels.validate(family, bad)
        bad = image_state.clone()
        bad["extra"] = TensorDict({}, [3])
        with pytest.raises(ValueError):
            pixels.validate(family, bad)
        assert all(torch.equal(env.get_state()[key], value) for key, value in saved.items(True, True))
        assert image_state["renderer", "size"][0, 0] == (224 if family == "Cube" else 200)


@pytest.mark.parametrize("family,cls", [("Cube", OGBCube), ("Scene", OGBScene)])
def test_render_model_configuration_no_steps_or_rng_mutation(family, cls):
    before = np.random.get_state()
    with cls(seed=17) as env:
        env.reset(
            17,
            {
                "variation": [],
                "variation_values": {
                    "agent.color": np.array([0.1, 0.3, 0.7]),
                    "cube.size": np.array([0.03]),
                    "camera.angle_delta": np.array([[2.0, 1.0]]),
                    "light.intensity": np.array([0.9]),
                },
            },
        )
        full = env.get_state()
        rs = env.to_render_state(full)
        candidate_cls = env._simulator.cls
        with (
            patch.object(candidate_cls, "step", side_effect=AssertionError("renderer stepped")),
            patch.object(candidate_cls, "reset", side_effect=AssertionError("renderer reset")),
        ):
            token, b = renderer.configured_bundle(family, rs[0])
        assert renderer.configured_bundle(family, rs[0])[1] is b
        source = env._simulator.envs[0]
        for name in ["geom_size", "body_mass", "body_inertia", "cam_pos", "cam_quat", "light_diffuse"]:
            assert np.array_equal(getattr(source.model, name), getattr(b.model, name)), name
        # Scene button tables intentionally place the private model at a known base.
        # The render-only Scene model adds two button materials. Compare named
        # materials; insertion can renumber unrelated compiled material IDs.
        for name in ["black", "pad_gray"]:
            assert np.array_equal(
                source.model.material("ur5e/robotiq/" + name).rgba,
                b.model.material("ur5e/robotiq/" + name).rgba,
            )
        assert b.cam >= 0 and b.nbuttons == (2 if family == "Scene" else 0)
        # Exercise the inherited style API without claiming a GPU render.
        obj = object.__new__(renderer._ConfiguredRenderer)
        obj.requires_cuda = False
        renderer._ConfiguredRenderer.__init__(obj, token, b, "cpu", 224, 1)
        styles = obj.resolve_style(
            None,
            {
                "cube.color": rs["appearance", "cube.color"].reshape(1, 3),
                "floor.color": rs["appearance", "floor.color"].reshape(1, 6),
            },
            batch=1,
        )
        assert styles.values.shape == (1, 9)
        obj.close()
        # A source agent-only pending MJCF change must not become a rendered compiled change.
        env.reset(17, {"variation": [], "variation_values": {"agent.color": np.array([0.8, 0.2, 0.1])}})
        current = env.to_render_state()
        assert torch.equal(current["appearance", "agent.color"], env.get_state()["model", "compiled", "agent.color"])
        assert renderer._ConfiguredRenderer._render is core.OGBManipBase._render
        assert renderer._ConfiguredRenderer._bake_plate is core.OGBManipBase._bake_plate
        assert renderer._ConfiguredRenderer._upload_cube_colours is core.OGBManipBase._upload_cube_colours
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


@pytest.mark.parametrize("family,cls", [("Cube", OGBCube), ("Scene", OGBScene)])
def test_renderstate_configuration_grouping(family, cls):
    with cls(seed=0) as env:
        rs = env.to_render_state()
        key = renderer.configuration_key(family, rs[0])
        for leaf in ["cube.color", "floor.color"]:
            other = rs.clone()
            other["appearance", leaf].fill_(0.2)
            assert renderer.configuration_key(family, other[0]) == key
        for leaf in ["cube.size", "agent.color", "camera.angle_delta", "light.intensity"]:
            other = rs.clone()
            other["appearance", leaf].add_(0.001)
            assert renderer.configuration_key(family, other[0]) != key
        with pytest.raises(ValueError):
            env.to_render_state(size=224.5)
        with pytest.raises(ValueError, match="integers"):
            env.render(rs, size=224.5)
        with pytest.raises(TypeError):
            pixels.validate(family, np.zeros(30))


@pytest.mark.parametrize("family,cls", [("Cube", OGBCube), ("Scene", OGBScene)])
def test_cpu_default_render_uses_existing_opengl_backend(family, cls):
    from stable_envs.envs.ogbench import manipulation_opengl

    with cls(num_envs=2, size=64, seed=17) as env:
        state = env.get_state()
        image = torch.zeros((2, 64, 64, 3), dtype=torch.uint8)
        with patch.object(manipulation_opengl, "render", return_value=image) as draw:
            assert env.render(state) is image
            args = draw.call_args.args
            assert args[0] == family and args[2] == torch.device("cpu")
            pixels.validate(family, args[1])
            draw.reset_mock()
            with pytest.raises(NotImplementedError, match="CPU mesh"):
                env.render(state, backend="mjwarp")
            draw.assert_not_called()
        assert env.cpu_render_support == "SUPPORTED_SOFTWARE_OPENGL"
