"""Test-only certification overlays: historical gym ids and native constructor refs.

Production ``stable_envs`` does not store SWM gymnasium ids. GPU/parity tests that still need
a live official renderer look them up here, or use the native upstream constructor (OGBench,
PyFlyt once a factory is installed).
"""

from __future__ import annotations

#: Historical gymnasium ids used only as a migration/certification oracle.
#: Keys are live stable-envs class names.
CERT_GYM_IDS: dict[str, dict] = {
    "TwoRoom": {"gym_id": "swm/TwoRoom-v1", "kwargs": {"render_target": False}},
    "TwoRoomTarget": {"gym_id": "swm/TwoRoom-v1", "kwargs": {"render_target": True}},
    "Piecewise": {"gym_id": "swm/Piecewise-v0", "kwargs": {}},
    "PiecewiseZones": {"gym_id": "swm/Piecewise-v0", "kwargs": {}, "render_zones_init": 1},
    "SimplePointMaze": {"gym_id": "swm/SimplePointMaze-v0", "kwargs": {"show_goal": True}},
    "PyFlytRocketLanding": {
        "gym_id": "swm/PFRocketLanding-v0",
        "render_resolution": True,
    },
}

#: Classes whose official reference is an upstream constructor, not a gymnasium id.
NATIVE_CONSTRUCTOR_REFERENCE = frozenset({"OGBCube", "OGBScene"})


def certification_gym_id(cls) -> str | None:
    """Historical gym id for ``cls``, or None."""
    name = cls if isinstance(cls, str) else getattr(cls, "__name__", "")
    spec = CERT_GYM_IDS.get(name)
    return None if spec is None else spec["gym_id"]


def has_reference_renderer(cls) -> bool:
    """Whether GPU/parity may try ``cls.reference_env``.

    Production ``reference_id`` (ogbench/, gym_pusht/, dm_control/, …) is the usual signal.
    Certification-only SWM gym ids and OGBench constructors are the two overlays.
    """
    if not hasattr(cls, "reference_env"):
        return False
    if getattr(cls, "reference_id", None):
        return True
    if getattr(cls, "_se_cert_gym_id", None):
        return True
    name = getattr(cls, "__name__", "")
    return name in CERT_GYM_IDS or name in NATIVE_CONSTRUCTOR_REFERENCE


def make_certification_env(cls, size: int | None = None):
    """Build the certification reference env for ``cls`` at ``size``."""
    name = cls if isinstance(cls, str) else cls.__name__
    spec = CERT_GYM_IDS[name]
    import gymnasium

    kwargs = dict(spec.get("kwargs") or {})
    if spec.get("render_resolution"):
        s = int(size) if size else 480
        kwargs["render_mode"] = "rgb_array"
        kwargs["render_resolution"] = (s, s)
    env = gymnasium.make(spec["gym_id"], **kwargs)
    if spec.get("render_zones_init"):
        env.unwrapped.variation_space["rendering"]["render_zones"].set_init_value(int(spec["render_zones_init"]))
    if spec.get("render_resolution"):
        env.reset(seed=0)
    return env


def install_on_classes() -> None:
    """Patch live classes so ``reference_env`` can still construct the certification overlay.

    Called from ``tests/conftest.py``. Production classes keep ``reference_id is None`` for
    these cells; the overlay is test-process state only.
    """
    from stable_envs.registry import REGISTRY

    for name, spec in CERT_GYM_IDS.items():
        cls = REGISTRY.get(name)
        if cls is None:
            continue
        gym_id = spec["gym_id"]

        def _reference_env(cls, size=None):
            return make_certification_env(cls, size)

        cls.reference_env = classmethod(_reference_env)
        cls._se_cert_gym_id = gym_id

    rocket = REGISTRY.get("PyFlytRocketLanding")
    if rocket is not None:
        import stable_envs.envs.rocket.legacy as rl

        def _cert_rocket(size: int):
            hit = rl._REF_CACHE.get(size)
            if hit is not None:
                return hit
            env = make_certification_env(rocket, size)
            env.reset(seed=0)
            rl._REF_CACHE[size] = env
            return env

        rl._reference_make = _cert_rocket
