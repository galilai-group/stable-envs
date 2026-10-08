"""Source-backed configuration checks, independent of image error thresholds."""

import itertools

import numpy as np
import pytest

_RETAINED_REFERENCES = []  # Keep GL contexts alive until process exit.


@pytest.mark.parametrize(
    "lock_colors",
    [
        [[0.15, 0.7, 0.3], [0.2, 0.4, 0.8]],
        [[0.5, 0.5, 0.5], [0.50001, 0.50001, 0.50001]],
    ],
)
def test_scene_button_materials_preserve_shading_and_independent_colors(lock_colors):
    pytest.importorskip("ogbench")
    import mujoco

    from stable_envs.envs.ogbench import OGBScene
    from stable_envs.envs.ogbench.simulation.render_state import build_model

    with OGBScene(seed=17) as env:
        row = env.to_render_state()[0].clone()
        row["appearance", "lock_color"][:] = row["appearance", "lock_color"].new_tensor(lock_colors)
        source = env._simulator.envs[0]
        source.variation_space._walk("lock_color").set_value(np.array(lock_colors))
        renderer = build_model("Scene", row)
        try:
            model = renderer.model
            old = model.material("btn_red").id
            ids = [model.material(f"se_button_{i}").id for i in range(2)]
            assert len(set([old, *ids])) == 3
            for attr in [
                "mat_specular",
                "mat_shininess",
                "mat_reflectance",
                "mat_texid",
                "mat_texrepeat",
                "mat_emission",
            ]:
                for mid in ids:
                    np.testing.assert_array_equal(getattr(model, attr)[mid], getattr(model, attr)[old])
            for bits in itertools.product([0, 1], repeat=2):
                renderer._cur_button_states = np.array(bits)
                renderer._apply_button_states()
                source._cur_button_states = np.array(bits)
                source._apply_button_states()
                # Read resolved MuJoCo colors, independently of our material
                # translation and without needing an OpenGL render context.
                resolved = []
                for owner in [source, renderer]:
                    scene = mujoco.MjvScene(owner.model, maxgeom=1000)
                    mujoco.mjv_updateScene(
                        owner.model,
                        owner.data,
                        mujoco.MjvOption(),
                        None,
                        mujoco.MjvCamera(),
                        mujoco.mjtCatBit.mjCAT_ALL,
                        scene,
                    )
                    resolved.append(
                        {
                            g.objid: g.rgba.copy()
                            for g in scene.geoms[: scene.ngeom]
                            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM
                        }
                    )
                for i, bit in enumerate(bits):
                    expected = [0.9] * 3 if bit else row["appearance", "lock_color"][i].numpy()
                    if np.array_equal(expected, [0.5] * 3):
                        expected = model.mat_rgba[old, :3]
                    np.testing.assert_allclose(model.mat_rgba[ids[i], :3], expected, atol=3e-8, rtol=0)
                    gid = renderer._button_geom_ids_list[i][0]
                    np.testing.assert_array_equal(resolved[0][gid], resolved[1][gid])
        finally:
            renderer.close()


def test_finger_visibility_groups_the_source_material_sites():
    pytest.importorskip("dm_control")
    from stable_envs.envs.dmcontrol import DMCFingerTurnHard

    with DMCFingerTurnHard(seed=17, device="cpu") as env:
        model = env._model()
        target = next(ids for kind, name, ids in env._hideable() if kind == "site" and name == "target")
        assert set(target) == {model.site("target").id, model.site("tip").id}
        assert env.resolve_style()["visibility.target"].item() == 0


def test_maze_observation_modes_have_separate_textures_and_cache_entries():
    pytest.importorskip("ogbench")
    from stable_envs.envs.ogbench.maze import _scene_env

    states = _scene_env("point", "large", 224, "states")
    pixels = _scene_env("point", "large", 224, "pixels")
    assert states[0] is not pixels[0]
    assert _scene_env("point", "large", 224, "states")[0] is states[0]
    # Source XML's checker has two colors; pixels mode replaces it with a gradient.
    assert len(np.unique(states[2]["tex"].reshape(-1, 3), axis=0)) == 2
    assert len(np.unique(pixels[2]["tex"].reshape(-1, 3), axis=0)) > 2


def test_maze_goal_is_rendered_from_supplied_state():
    pytest.importorskip("ogbench")
    import torch

    from stable_envs.envs.ogbench import PointMazeLargeNavigate

    with PointMazeLargeNavigate(seed=17, device="cpu") as env:
        state = env.get_state().expand(2).clone()
        state["task", "goal_pos"][1, :2] += 3
        packed = env._pack_render_state(state)
        assert packed.shape == (2, 5)
        scene = env._scene()
        positions, _ = env._fk(packed, scene)
        target = scene.agent_ids.index(scene.model.geom("target").id)
        torch.testing.assert_close(positions[:, target], state["task", "goal_pos"].float())
        assert len(scene.lights.dirs) == len(scene.lights.diffuse) == len(scene.lights.ambient)
        sampled = env.sample_states(3, np.random.default_rng(0))
        assert sampled.shape == (3, 5)
        np.testing.assert_array_equal(env._pack_render_state(env._unpack_render_state(sampled)).numpy(), sampled)
        reference = env.reference_env()
        _RETAINED_REFERENCES.append(reference)
        assert env.reference_render(reference, sampled).shape == (3, 224, 224, 3)
        np.testing.assert_array_equal(reference.unwrapped.model.geom("target").pos, sampled[-1, 2:].astype(np.float64))
