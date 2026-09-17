"""The public namespace, the deprecated Step-8 aliases, the config, and resource lifetime.

Every assertion here is about the PROMISE rather than about an implementation: that the old names keep
working and say so once, that closing is explicit and idempotent, and that a closed env refuses to
draw rather than handing back a frame through a backend that may no longer exist.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

import stable_envs as snv
from stable_envs import _config

pytestmark = pytest.mark.unit


@pytest.fixture()
def records():
    """Collect what this package logs.

    Through a loguru sink rather than ``capsys``: loguru binds its default handler to the ``sys.stderr``
    it saw at import, so pytest's stream capture does not necessarily see it, and what is under test is
    what stable-envs EMITS, not which file descriptor it lands on.
    """
    from stable_envs._config import logger
    got: list[str] = []
    sink = logger.add(got.append, level="TRACE",
                      filter=lambda r: r["extra"].get("package") == "stable_envs")
    yield got
    logger.remove(sink)


@pytest.fixture(autouse=True)
def _clean_config():
    before = _config.get_config()
    yield
    _config.reset_config()
    _config.set(**before)


# ------------------------------------------------------------------ namespace


def test_the_public_namespace_is_the_contract():
    for name in ("Env", "State", "Spec", "CompositeSpec", "make", "spec", "list_envs", "set",
                 "get_config", "import_errors", "__version__"):
        assert hasattr(snv, name), f"snv.{name} is part of the Step-9 public namespace"
    assert snv.__version__ == __import__("stable_envs.__about__", fromlist=["x"]).__version__


def test_environment_classes_are_reachable_by_name_and_are_the_registry_s_classes():
    assert snv.PushT is snv.registry.REGISTRY["PushT"]
    assert issubclass(snv.PushT, snv.Env)
    assert "PushT" in dir(snv)


def test_an_unknown_attribute_says_how_to_find_the_environments():
    with pytest.raises(AttributeError) as e:
        snv.NoSuchEnv
    assert "list_envs()" in str(e.value) and "import_errors()" in str(e.value)


def test_state_is_the_tensordict_type_and_not_a_private_subclass():
    from tensordict import TensorDict
    assert snv.State is TensorDict


# ------------------------------------------------------------------ deprecated aliases


@pytest.mark.parametrize("name", ["StableEnv", "REGISTRY", "register"])
def test_the_step8_names_still_work_and_warn_exactly_once(name, records):
    _config.reset_config()
    obj = getattr(snv, name)
    getattr(snv, name)
    assert obj is not None
    warnings = [r for r in records if name in r]
    assert len(warnings) == 1, f"{name} must warn exactly once per process, got {len(warnings)}"
    assert "stable-envs" in warnings[0] and "WARNING" in warnings[0]


def test_stableenv_is_the_same_class_object_as_env():
    _config.set(warn_on_deprecated=False)
    assert snv.StableEnv is snv.Env
    from stable_envs.core.env import StableEnv as ShimStableEnv
    assert ShimStableEnv is snv.Env


def test_registry_is_the_one_registry_whichever_path_finds_it():
    _config.set(warn_on_deprecated=False)
    from stable_envs.core.env import REGISTRY as shim
    assert shim is snv.registry.REGISTRY is snv.REGISTRY


def test_make_with_batch_still_builds_an_env_and_warns_once(records):
    _config.reset_config()
    env = snv.make("PushT", batch=4, device="cpu")
    env2 = snv.make("PushT", batch=2, device="cpu")
    assert env.num_envs == 4 == env.B and env2.num_envs == 2
    warnings = [r for r in records if "batch=" in r]
    assert len(warnings) == 1 and "num_envs=" in warnings[0]
    env.close(); env2.close()


def test_num_envs_and_batch_cannot_disagree():
    with pytest.raises(TypeError, match="same argument"):
        snv.PushT(num_envs=8, batch=4, device="cpu")


# ------------------------------------------------------------------ config


def test_config_is_three_keys_and_a_typo_is_an_error():
    cfg = snv.get_config()
    assert set(cfg) == {"verbose", "default_device", "warn_on_deprecated"}
    snv.set(verbose="INFO")
    assert snv.get_config()["verbose"] == "INFO"
    with pytest.raises(KeyError, match="unknown config key"):
        snv.set(verbosity="INFO")
    with pytest.raises(ValueError, match="verbose must be one of"):
        snv.set(verbose="LOUD")


def test_default_device_is_honoured_when_the_caller_names_none():
    snv.set(default_device="cpu")
    env = snv.PushT(num_envs=2)
    assert env.device == torch.device("cpu")
    env.close()


def test_warn_on_deprecated_can_be_switched_off(records):
    _config.reset_config()
    snv.set(warn_on_deprecated=False)
    snv.StableEnv
    snv.make("PushT", batch=2, device="cpu").close()
    assert records == []


# ------------------------------------------------------------------ lifecycle


def test_close_is_idempotent_and_a_closed_env_refuses_to_render():
    env = snv.PushT(num_envs=2, device="cpu")
    st = env.reset(seed=0)
    env.close()
    env.close()
    assert env.closed
    for call, what in ((lambda: env.render(st), "render"), (lambda: env.render(), "render"),
                       (lambda: env.reset(seed=0), "reset"),
                       (lambda: env.step(torch.zeros(2, 2)), "step")):
        with pytest.raises(RuntimeError) as e:
            call()
        assert "closed" in str(e.value) and f"PushT.{what}" in str(e.value)
        assert "num_envs=2" in str(e.value), "the message must say how to get a working env back"


def test_the_context_manager_closes_and_does_not_swallow_exceptions():
    with snv.PushT(num_envs=2, device="cpu") as env:
        assert env.render(env.reset(seed=0)).shape[0] == 2
    assert env.closed

    with pytest.raises(ZeroDivisionError):
        with snv.PushT(num_envs=2, device="cpu") as env2:
            1 / 0
    assert env2.closed
