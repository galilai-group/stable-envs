"""The public namespace, the config, and resource lifetime.

``dir(snv)`` is the public contract, closing is explicit and idempotent, and a closed env refuses to
draw rather than hand back a frame through a backend that may be gone.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

import stable_envs as snv
from stable_envs import registry as R
from stable_envs._core import config as _config

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_config():
    before = _config.get_config()
    yield
    _config.reset_config()
    _config.set(**before)


def test_dir_advertises_the_documented_namespace_and_the_environments_only():
    """The top level exposes the happy path, and backend machinery and internal names do not resolve there."""
    listed = set(dir(snv))
    assert {"Env", "State", "StepResult", "VariationLike", "make", "info", "PushT"} <= listed
    assert not listed & {
        "Scene2D",
        "raster2d",
        "raycast3d",
        "render",
        "testing",
        "StyleTensor",
        "render_states",
        "StableEnv",
        "REGISTRY",
        "register",
        "SWM_IDS",
        "spec",
    }
    for name in (
        "Scene2D",
        "StyleTensor",
        "SWM_IDS",
        "StableEnv",
        "REGISTRY",
        "register",
        "render_states",
        "StateDataset",
        "TrajectoryDataset",
        "collate_batched",
        "BasePolicy",
        "sample",
    ):
        with pytest.raises(AttributeError):
            getattr(snv, name)


def test_the_public_namespace_is_the_contract():
    for name in (
        "Env",
        "State",
        "VariationLike",
        "Spec",
        "CompositeSpec",
        "make",
        "info",
        "list_envs",
        "set",
        "get_config",
        "import_errors",
        "StepResult",
        "__version__",
    ):
        assert hasattr(snv, name), f"snv.{name} is part of the public namespace"
    assert snv.__version__ == __import__("stable_envs.__about__", fromlist=["x"]).__version__


def test_environment_classes_are_reachable_by_name_and_are_the_registry_s_classes():
    assert snv.PushT is snv.registry.REGISTRY["PushT"]
    assert issubclass(snv.PushT, snv.Env)
    assert "PushT" in dir(snv)


def test_an_unknown_attribute_says_how_to_find_the_environments():
    with pytest.raises(AttributeError) as e:
        snv.NoSuchEnv
    assert "list_envs()" in str(e.value) and "import_errors()" in str(e.value)


def test_state_and_stepresult_are_the_tensordict_type_and_not_private_subclasses():
    from tensordict import TensorDict

    assert snv.State is TensorDict and snv.StepResult is TensorDict
    assert "StepResult" in snv.__all__, "`StepResult` names what step returns and must be exported"


@pytest.mark.family("pusht")
def test_make_takes_num_envs():
    env = snv.make("PushT", num_envs=4, device="cpu")
    env2 = snv.make("PushT", num_envs=2, device="cpu")
    assert env.num_envs == 4 == env.B and env2.num_envs == 2
    env.close()
    env2.close()


def test_no_registered_class_declares_batch_as_the_constructor_spelling():
    """``batch=`` is not a constructor spelling of ``num_envs=``."""
    import inspect

    holdouts = sorted(
        n
        for n, c in R.REGISTRY.items()
        if "batch" in inspect.signature(c.__init__).parameters
        and "num_envs" not in inspect.signature(c.__init__).parameters
    )
    assert holdouts == [], f"registered classes take batch= as the constructor: {holdouts}"


def test_batch_is_not_a_spelling_of_num_envs():
    with pytest.raises(TypeError, match="unexpected keyword argument 'batch'"):
        snv.PushT(batch=4, device="cpu")
    with pytest.raises(TypeError, match="unexpected keyword argument 'batch'"):
        snv.make("PushT", batch=4, device="cpu")


def test_config_is_two_keys_and_a_typo_is_an_error():
    cfg = snv.get_config()
    assert set(cfg) == {"verbose", "default_device"}
    assert cfg["default_device"] == "auto"
    snv.set(verbose="INFO")
    assert snv.get_config()["verbose"] == "INFO"
    with pytest.raises(KeyError, match="unknown config key"):
        snv.set(verbosity="INFO")
    with pytest.raises(ValueError, match="verbose must be one of"):
        snv.set(verbose="LOUD")
    with pytest.raises(KeyError, match="unknown config key"):
        snv.set(warn_on_deprecated=False)


@pytest.mark.family("pusht")
def test_default_device_auto_is_cpu_when_cuda_is_unavailable():
    snv.set(default_device="auto")
    env = snv.PushT(num_envs=2)
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert env.device == torch.device(expected)
    env.close()


@pytest.mark.family("pusht")
def test_default_device_is_honoured_when_the_caller_names_none():
    snv.set(default_device="cpu")
    env = snv.PushT(num_envs=2)
    assert env.device == torch.device("cpu")
    env.close()


@pytest.mark.family("pusht")
def test_close_is_idempotent_and_a_closed_env_refuses_to_render():
    env = snv.PushT(num_envs=2, device="cpu")
    st = env.reset(seed=0)
    env.close()
    env.close()
    assert env.closed
    for call, what in (
        (lambda: env.render(st), "render"),
        (lambda: env.render(), "render"),
        (lambda: env.reset(seed=0), "reset"),
        (lambda: env.step(torch.zeros(2, 2), frames=None), "step"),
    ):
        with pytest.raises(RuntimeError) as e:
            call()
        assert "closed" in str(e.value) and f"PushT.{what}" in str(e.value)
        assert "Create a new environment" in str(e.value), "the message must say how to get a working env back"


@pytest.mark.family("pusht")
def test_the_context_manager_closes_and_does_not_swallow_exceptions():
    with snv.PushT(num_envs=2, device="cpu") as env:
        assert env.render(env.reset(seed=0)).shape[0] == 2
    assert env.closed

    with pytest.raises(ZeroDivisionError):
        with snv.PushT(num_envs=2, device="cpu") as env2:
            1 / 0
    assert env2.closed


def test_testing_harness_asks_has_dynamics_not_hasattr_step():
    """Every Env has a ``step``; a ``hasattr(env, "step")`` guard is therefore always true and
    must not treat renderer-only cells as simulators."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2] / "stable_envs"
    offenders = [
        str(p.relative_to(root.parent))
        for p in root.rglob("*.py")
        if p.name != "dynamics.py" and re.search(r'hasattr\(\s*\w+\s*,\s*"step"\s*\)', p.read_text())
    ]
    assert offenders == [], offenders
