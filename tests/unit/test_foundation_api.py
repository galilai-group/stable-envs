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
from stable_envs import registry as R

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


def test_dir_advertises_the_documented_namespace_and_the_environments_only():
    """Contract §1: the top level exposes the happy path, not backend machinery.

    The retained Step-8 names still RESOLVE (escalation 3) — they must, until Workstream K migrates
    ~470 call sites — but tab completion must not teach them to anyone new.
    """
    listed = set(dir(snv))
    assert {"Env", "State", "StepResult", "make", "spec", "PushT"} <= listed
    assert not listed & {"Scene2D", "raster2d", "raycast3d", "render", "testing", "StyleTensor",
                         "render_states", "StableEnv", "REGISTRY", "register"}
    for name in ("Scene2D", "StyleTensor", "render_states"):
        assert getattr(snv, name) is not None, f"snv.{name} must still resolve, just not be listed"


def test_the_public_namespace_is_the_contract():
    for name in ("Env", "State", "Spec", "CompositeSpec", "make", "spec", "list_envs", "set",
                 "get_config", "import_errors", "StepResult", "__version__"):
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


def test_state_and_stepresult_are_the_tensordict_type_and_not_private_subclasses():
    from tensordict import TensorDict
    assert snv.State is TensorDict and snv.StepResult is TensorDict
    assert "StepResult" in snv.__all__, ("`StepResult` is the name five docstrings and the contract "
                                         "use for what step returns; it must be exported so cells do "
                                         "not each hand-roll the TensorDict call")


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


#: Classes whose own ``__init__`` still declares ``batch=`` and not ``num_envs=``. They accept the
#: Step-8 spelling and REJECT the Step-9 one, because they forward positionally to ``Env.__init__``
#: and never reach its deprecation branch. Migrating them is Wave-1 family-agent work (findings §2,
#: "Wave-1 scope"); the number is pinned here so it can only go DOWN.
#:
#: 41 at Wave-0. Workstream D migrated its 27 (ogbench 25 + ogb_manip 2) and the ceiling comes down
#: with the merge, measured on the merged tip rather than taken from the workstream's prose: D
#: reported the count but left the pin at 41, which is a ceiling that no longer constrains anything.
#: The remaining 14 are Workstream F's (highway 6, maze2d 4, mujoco_locomotion 4) and F lowers this
#: to 0 when it merges.
NUM_ENVS_HOLDOUTS = 14


def test_the_number_of_classes_that_still_reject_num_envs_only_goes_down():
    """Wave-0 migrated one cell. This is the size of the job Wave 1 inherits, as a number.

    Checked from the SIGNATURE rather than by constructing: most of these families need a GPU, an EGL
    context or a compiled scene, so a test that instantiated would skip on the machine where this
    question is asked and the count would quietly stop being enforced.
    """
    import inspect

    holdouts = sorted(n for n, c in R.REGISTRY.items()
                      if "batch" in inspect.signature(c.__init__).parameters
                      and "num_envs" not in inspect.signature(c.__init__).parameters)
    assert len(holdouts) <= NUM_ENVS_HOLDOUTS, (
        f"{len(holdouts)} registered classes reject num_envs=, up from {NUM_ENVS_HOLDOUTS}. A new "
        f"env must take the Step-9 constructor: {holdouts}")
    assert "PushT" not in holdouts, "the Wave-0 template must take num_envs="


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


def test_testing_harness_asks_has_dynamics_not_hasattr_step():
    """Wave 0 gave every Env a ``step``; a ``hasattr(env, "step")`` guard is therefore always true and
    sends renderer-only cells into ``env.sample``, which refuses them (Workstream A, R9)."""
    import pathlib, re
    root = pathlib.Path(__file__).resolve().parents[2] / "stable_envs"
    offenders = [str(p.relative_to(root.parent)) for p in root.rglob("*.py")
                 if p.name != "dynamics.py" and re.search(r'hasattr\(\s*\w+\s*,\s*"step"\s*\)', p.read_text())]
    assert offenders == [], offenders
