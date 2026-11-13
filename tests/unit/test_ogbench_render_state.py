"""CPU validation of the named renderer inputs and configured model-provider boundary."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("ogbench")
from stable_envs.envs.ogbench import OGBCube, OGBScene
from stable_envs.envs.ogbench import manipulation as core
from stable_envs.envs.ogbench import manipulation_renderer as renderer
from stable_envs.envs.ogbench.simulation import render_state as pixels


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
    regular_key, regular = renderer.configured_bundle("Scene")
    sampled_key, sampled = renderer.configured_bundle("Scene", "gl4")
    assert regular_key != sampled_key and regular is not sampled
    assert regular.sample_pattern == "grid" and sampled.sample_pattern == "gl4"
    assert renderer.configured_bundle("Scene") == (regular_key, regular)


@pytest.mark.parametrize("family", ["Cube", "Scene"])
def test_per_world_arrays_match_each_rows_own_compile(family):
    from tests.helpers.ogbench_golden import FIXTURES, load

    state, _ = load(FIXTURES / f"{family}-mjwarp.npz")
    state = state[:6].clone()
    state["appearance", "cube.size"][1] = 0.013
    state["appearance", "light.intensity"][2] = 0.35
    state["appearance", "camera.angle_delta"][3] = state["appearance", "camera.angle_delta"].new_tensor([[4.0, -3.0]])
    if family == "Scene":
        state["appearance", "lock_color"][4] = state["appearance", "lock_color"].new_tensor([[0.1, 0.8, 0.2]] * 2)
    # Colours and poses are uploaded per world and never split a configuration.
    state["pose", "qpos"][5, 0] += 0.1
    state["appearance", "cube.color"][5] = 0.2
    token, bundle = renderer.configured_bundle(family)
    worlds = renderer._worlds(family, token, bundle, state, "cpu")
    assert worlds.view.tolist() == [0, 0, 1, 2, 0, 0]
    assert worlds.lock is None if family == "Cube" else worlds.lock.tolist() == [0, 0, 0, 0, 1, 0]
    for index, row in enumerate(state):
        env = pixels.build_model(family, row)
        try:
            for name in core.WORLD_FIELDS:
                want = torch.as_tensor(getattr(env.model, name), dtype=torch.float32)
                got = worlds.fields[name][index if len(worlds.fields[name]) > 1 else 0]
                assert torch.equal(got.reshape(want.shape), want), (index, name)
            if family == "Scene":
                want = core._button_tables(env.unwrapped, env.model, 2)
                got = worlds.locks[worlds.lock[index]].tables
                for bits, (g, m) in want.items():
                    assert np.array_equal(got[bits][0], g) and np.array_equal(got[bits][1], m), (index, bits)
        finally:
            env.close()
    rows = slice(2, 5)
    sliced = worlds[rows]
    assert sliced.view.tolist() == worlds.view[rows].tolist()
    for name, value in worlds.fields.items():
        assert torch.equal(sliced.fields[name], value[rows])


@pytest.mark.parametrize("family", ["Cube", "Scene"])
def test_uniform_batch_keeps_one_world_entry(family):
    from tests.helpers.ogbench_golden import FIXTURES, load

    state, _ = load(FIXTURES / f"{family}-mjwarp.npz")
    state = state[:4].clone()
    for key in state["appearance"].keys():
        if key not in ("agent.color", "cube.color"):
            state["appearance", key][:] = state["appearance", key][0]
    token, bundle = renderer.configured_bundle(family)
    worlds = renderer._worlds(family, token, bundle, state, "cpu")
    assert worlds.view.tolist() == [0] * 4 and len(worlds.views) == 1
    assert all(len(value) == 1 for value in worlds.fields.values())
    assert worlds.lock is None if family == "Cube" else len(worlds.locks) == 1


@pytest.mark.parametrize("family,cls", [("Cube", OGBCube), ("Scene", OGBScene)])
def test_render_of_state_checks_only_the_leaves_the_frame_reads(family, cls):
    with cls(num_envs=3, seed=17) as env:
        full = env.get_state()
        frames = env.render(full)
        unread = full.clone()
        unread["physics", "qvel"][0, 0] = float("nan")
        unread["rng", "task", "position"].fill_(10**6)
        assert torch.equal(env.render(unread), frames)
        assert torch.equal(env.render(full[:2].clone()), frames[:2])
        with pytest.raises(ValueError):
            env.set_state(unread)
        for key in pixels.projected_keys(family):
            bad = full.clone()
            del bad[key]
            with pytest.raises(ValueError, match="missing"):
                env.render(bad)
        bad = full.clone()
        bad["physics", "qpos"][0, 0] = float("nan")
        with pytest.raises(ValueError, match=r"\('physics', 'qpos'\)"):
            env.render(bad)


@pytest.mark.parametrize("family,cls", [("Cube", OGBCube), ("Scene", OGBScene)])
def test_render_state_checks_agree_on_host_and_device(family, cls):
    with cls(num_envs=3, seed=17) as env:
        image_state = env.to_render_state()
    pixels.validate(family, image_state.clone().clear_device_())
    for key, value in [
        (("pose", "qpos"), float("nan")),
        (("renderer", "signature"), 0),
        (("renderer", "camera"), 1),
        (("renderer", "size"), 0),
        (("appearance", "cube.size"), 0.3),
        (("appearance", "target_alpha"), 0.2),
        (("appearance", "camera.angle_delta"), 30),
        (("appearance", "cube.color"), 2),
        (("pose", "mocap_quat"), 0),
    ]:
        bad = image_state.clone()
        bad[key][-1] = value
        messages = []
        # A TensorDict without a device takes the device branch of validate.
        for candidate in (bad, bad.clone().clear_device_()):
            with pytest.raises(ValueError) as error:
                pixels.validate(family, candidate)
            messages.append(str(error.value))
        assert messages[0] == messages[1]


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
            (("appearance", "target_alpha"), 0.2),
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
def test_building_worlds_neither_steps_nor_resets_nor_touches_global_rng(family, cls):
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
        rs = env.to_render_state(env.get_state())
        source_cls = env._simulator.cls
        with (
            patch.object(source_cls, "step", side_effect=AssertionError("renderer stepped")),
            patch.object(source_cls, "reset", side_effect=AssertionError("renderer reset")),
        ):
            token, b = renderer.configured_bundle(family)
            worlds = renderer._worlds(family, token, b, rs.cpu(), "cpu")
        assert renderer.configured_bundle(family)[1] is b
        source = env._simulator.envs[0]
        for name in core.WORLD_FIELDS:
            want = torch.as_tensor(getattr(source.model, name), dtype=torch.float32)
            assert torch.equal(worlds.fields[name][0].reshape(want.shape), want), name
        assert [b.model.material(int(i)).name for i in b.agent_mats] == list(pixels.AGENT_MATERIALS)
        assert b.cam >= 0 and b.nbuttons == (2 if family == "Scene" else 0)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


@pytest.mark.parametrize("family,cls", [("Cube", OGBCube), ("Scene", OGBScene)])
def test_pending_agent_color_is_not_a_compiled_render_change(family, cls):
    with cls(seed=17) as env:
        env.reset(17, {"variation": [], "variation_values": {"agent.color": np.array([0.8, 0.2, 0.1])}})
        projected = env.to_render_state()["appearance", "agent.color"]
        assert torch.equal(projected, env.get_state()["model", "compiled", "agent.color"])


@pytest.mark.parametrize("family,cls", [("Cube", OGBCube), ("Scene", OGBScene)])
def test_renderstate_size_validation(family, cls):
    with cls(seed=0) as env:
        rs = env.to_render_state()
        with pytest.raises(ValueError):
            env.to_render_state(size=224.5)
        with pytest.raises(ValueError, match="integers"):
            env.render(rs, size=224.5)
        with pytest.raises(TypeError):
            pixels.validate(family, np.zeros(30))


@pytest.mark.parametrize("family,cls", [("Cube", OGBCube), ("Scene", OGBScene)])
def test_cpu_default_render_uses_the_rasterizer(family, cls):
    pytest.importorskip("numba")
    from stable_envs.envs.ogbench import manipulation_raster

    with cls(num_envs=2, size=64, seed=17) as env:
        state = env.get_state()
        image = torch.zeros((2, 64, 64, 3), dtype=torch.uint8)
        with patch.object(manipulation_raster, "frames", return_value=image) as draw:
            assert env.render(state) is image
            args = draw.call_args.args
            assert args[0] == family and args[2] == torch.device("cpu")
            assert args[1]["renderer", "size"].shape == (2, 1)
            draw.reset_mock()
            with pytest.raises(NotImplementedError, match="CPU mesh"):
                env.render(state, backend="mjwarp")
            draw.assert_not_called()
        assert env.cpu_render_support == "SUPPORTED_SOFTWARE_OPENGL"
