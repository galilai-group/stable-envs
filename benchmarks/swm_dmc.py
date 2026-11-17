"""Sources through stable-worldmodel (swm), and variation values that make ours draw swm's look.

Every source is swm. For a DMC class, the offline source and the frame MSE draw with the dm_control env inside swm's
registered wrapper (gym id from tests/helpers/reference/crosswalk.py) after a swm reset, so the swm model edits and
default variation values apply; the rows are written into that env with the class's own reference_render (qpos and
the model entries of the row), at the swm camera. OGBCube and OGBScene draw with swm's OGBCube-v0 and OGBScene-v0
(OGBench's env with ob_type="states": opaque robot arm, goal shown) at the frame size.

SNV_VARIATION (JSON, {class: {leaf: value}}) gives variation values that every stable-envs env of that class gets
right after construction, followed by a reset with seed 0 (the leaves compile at reset), e.g.
{"OGBCube": {"agent.opacity": [1.0]}} for the opaque robot arm of swm.
"""

from __future__ import annotations

import json
import os

_DONE = False


def swm_id(cls) -> str | None:
    from tests.helpers.reference.crosswalk import swm_ids_for_class

    ids = swm_ids_for_class(cls.__name__)
    return ids[0] if ids else None


def _dmc_reference_env(cls, size: int | None = None):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  registers the swm/ ids

    sid = swm_id(cls)
    wrapper = gym.make(sid, render_mode="rgb_array")
    wrapper.reset(seed=0)
    inner = wrapper.unwrapped
    if int(inner.camera_id) != int(cls.CAMERA):
        raise ValueError(f"{cls.__name__}: swm camera {inner.camera_id} != class camera {cls.CAMERA}")
    env = inner.env
    env._se_size = int(size or cls.default_size)
    # The wrapper owns the GL context of the dm_control env; keep it alive with the env.
    env.swm_wrapper = wrapper
    env.swm_id = sid
    return env


#: OGBench manipulation classes drawn with swm's registered env (OGBench's env with ob_type="states").
SWM_MANIPULATION = {"OGBCube": "swm/OGBCube-v0", "OGBScene": "swm/OGBScene-v0"}


def _manipulation_reference_env(cls, size: int | None = None):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    size = int(size or cls.native_size or cls.default_size)
    sid = SWM_MANIPULATION[cls.__name__]
    env = gym.make(sid, width=size, height=size)
    env.reset(seed=0)
    env.swm_id = sid
    if cls.__name__ == "OGBScene":
        # swm's SceneEnv.set_state takes the button states as keywords (button_state_0, ...); the reference
        # renderer passes them as a third positional argument, as OGBench's own SceneEnv does.
        u = env.unwrapped
        swm_set_state = u.set_state

        def set_state(qpos, qvel, button_states=None):
            bits = {} if button_states is None else {f"button_state_{i}": int(b) for i, b in enumerate(button_states)}
            return swm_set_state(qpos, qvel, **bits)

        u.set_state = set_state
    return env


def variations() -> dict:
    return json.loads(os.environ.get("SNV_VARIATION") or "{}")


def install() -> None:
    """Point DMC and OGBCube reference_env at swm, and apply SNV_VARIATION to every env stable_envs.make builds."""
    global _DONE
    if _DONE:
        return
    import stable_envs as snv
    from stable_envs import registry

    for name, cls in registry.REGISTRY.items():
        if getattr(cls, "family", "") == "dmcontrol" and swm_id(cls):
            cls.reference_env = classmethod(_dmc_reference_env)
            cls.reference_source = "swm"
    for name in SWM_MANIPULATION:
        if name in registry.REGISTRY:
            registry.REGISTRY[name].reference_env = classmethod(_manipulation_reference_env)
            registry.REGISTRY[name].reference_source = "swm"
    wanted = variations()
    if wanted:
        make = snv.make

        def make_with_variation(name, *args, **kwargs):
            env = make(name, *args, **kwargs)
            v = wanted.get(name)
            if v:
                env.set_variation(dict(v))
                env.reset(seed=0)
            return env

        snv.make = make_with_variation
    _DONE = True
