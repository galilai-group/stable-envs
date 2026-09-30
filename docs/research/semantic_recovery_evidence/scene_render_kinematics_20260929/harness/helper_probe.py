"""CPU-only checks for the separately staged Scene render-kinematics candidate."""

import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from stable_envs._adapters.ogbench_manipulation_compat import CompatibleScene

root = Path(__file__).parent
path = root / "overlay/stable_envs/envs/ogbench/manipulation_opengl.py"
spec = importlib.util.spec_from_file_location("stable_envs.envs.ogbench._kinematics_candidate", path)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
env = CompatibleScene(seed=0)
env._mjcf_model = env.modify_mjcf_model(env.build_mjcf_model())
env.compile_model_and_data()
renderer = SimpleNamespace(_scene_option=mujoco.MjvOption())
assert candidate._can_use_render_kinematics(env, renderer)
checks = []
for model_field, enum in [
    ("cam_mode", mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM),
    ("light_mode", mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM),
    ("wrap_type", mujoco.mjtWrap.mjWRAP_SPHERE),
]:
    value = getattr(env.model, model_field)
    old = value[0]
    value[0] = enum
    assert not candidate._can_use_render_kinematics(env, renderer), model_field
    value[0] = old
    checks.append(model_field)
for flag in [mujoco.mjtVisFlag.mjVIS_CONTACTPOINT, mujoco.mjtVisFlag.mjVIS_ACTUATOR]:
    renderer._scene_option.flags[flag] = True
    assert not candidate._can_use_render_kinematics(env, renderer)
    renderer._scene_option.flags[flag] = False
    checks.append(str(flag))
env.model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_SLEEP
assert not candidate._can_use_render_kinematics(env, renderer)
env.model.opt.enableflags = 0
checks.append("sleep")
assert candidate._can_use_render_kinematics(env, renderer)

rng = np.random.default_rng(17)
color_cases = 0
for buttons in itertools.product([0, 1], repeat=2):
    for colors in [rng.uniform(size=(2, 3)), np.full((2, 3), 0.5), np.zeros((2, 3)), np.ones((2, 3))]:
        env._cur_button_states = np.asarray(buttons)
        env.variation_space._walk("lock_color")._value = colors.copy()
        env._apply_button_states()
        geom = env.model.geom_rgba.copy()
        material = env.model.mat_rgba.copy()
        # Corrupt just the helper's target colors so equality cannot pass stale.
        for ids in env._button_geom_ids_list:
            env.model.geom_rgba[ids] = 0
        for name in ["drawer_handle", "window_handle"]:
            env.model.material(name).rgba[:] = 0
        candidate._apply_render_button_colors(env, colors)
        assert np.array_equal(env.model.geom_rgba, geom)
        assert np.array_equal(env.model.mat_rgba, material)
        color_cases += 1
print(
    json.dumps(
        dict(status="PASS", mujoco=mujoco.__version__, button_color_cases=color_cases, fallback_checks=checks),
        indent=2,
    )
)
