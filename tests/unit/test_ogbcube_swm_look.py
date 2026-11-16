"""agent.opacity and target.opacity: OGBench's pixel-task look by default, swm's OGBCube-v0 look at 1."""

import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")

from stable_envs.envs.ogbench.simulation import render_state as pixels
from stable_envs.render._resources import retain_process_lifetime
from stable_envs.envs.ogbench.simulation.compatibility import PIXEL_ALPHAS

pytestmark = pytest.mark.family("ogbench")

LOOK = ("agent.opacity", "target.opacity")


def _alphas(opacity=None):
    """The arm material alphas and the goal cube alpha of the render model."""
    row = None
    if opacity is not None:
        row = {("appearance", name): torch.tensor([opacity], dtype=torch.float64) for name in LOOK}
    env = pixels.build_model("Cube", row, None if row is None else LOOK)
    try:
        for name in LOOK:
            assert np.array_equal(env.variation_space._walk(name).init_value, [0.0])
        arm = {name: float(env.model.material(name).rgba[3]) for name in PIXEL_ALPHAS}
        target = {float(env.model.geom_rgba[i, 3]) for ids in env._cube_target_geom_ids_list for i in ids}
        return arm, target
    finally:
        env.close()


def test_default_look_is_the_pixel_task_look():
    arm, target = _alphas()
    assert arm == pytest.approx(PIXEL_ALPHAS)
    assert target == {0.0}


def test_opacity_one_is_the_swm_look():
    arm, target = _alphas(1.0)
    assert arm == pytest.approx(dict.fromkeys(PIXEL_ALPHAS, 1.0))
    assert sorted(target) == pytest.approx([0.2])


@pytest.mark.swm
def test_opacity_one_matches_swm_frames():
    """A plain swm OGBCube-v0 (goal cube shown) against ours at agent.opacity = target.opacity = 1."""
    pytest.importorskip("stable_worldmodel")
    import gymnasium as gym

    import stable_envs as snv

    import mujoco

    source = gym.make("swm/OGBCube-v0", width=224, height=224).unwrapped
    source.reset(seed=0)
    # Closing the source renderer would release its GL context, which corrupts the other live contexts.
    retain_process_lifetime(source, "swm OGBCube-v0 reference renderer")
    env = snv.make("OGBCube", num_envs=4, device="cpu", seed=0)
    try:
        env.set_variation({"agent.opacity": [1.0], "target.opacity": [1.0]})
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(25):
            env.step(torch.as_tensor(rng.uniform(-1, 1, (4, 5)), dtype=torch.float32))
        state = env.get_state()
        qpos = snv.get_field(state, "physics/qpos").numpy()
        mocap = snv.get_field(state, "physics/mocap_pos").numpy(), snv.get_field(state, "physics/mocap_quat").numpy()
        expected = []
        for row, position, orientation in zip(qpos, *mocap):
            source.set_state(row.astype(np.float64), np.zeros(source._model.nv))
            source._data.mocap_pos[:] = position.reshape(-1, 3)
            source._data.mocap_quat[:] = orientation.reshape(-1, 4)
            mujoco.mj_forward(source._model, source._data)
            expected.append(np.asarray(source.render())[..., :3])
        expected = np.stack(expected).astype(np.float64)
        opaque = env.render(state, backend="opengl").numpy().astype(np.float64)
        assert ((opaque - expected) ** 2).mean() <= 0.01
        default = state.clone()
        for namespace in ("compiled", "mjcf"):
            for name in LOOK:
                snv.get_field(default, f"model/{namespace}/{name.replace('.', '/')}").zero_()
        see_through = env.render(default, backend="opengl").numpy().astype(np.float64)
        assert ((see_through - expected) ** 2).mean() > 100
    finally:
        env.close()
