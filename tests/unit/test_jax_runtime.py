"""JAX's GPU preallocation is turned off when a JAX backend is built, not at import."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from loguru import logger

import stable_envs as snv
from stable_envs._core import jax_runtime

ROOT = Path(__file__).resolve().parents[2]
VAR = jax_runtime.PREALLOCATE


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(jax_runtime, "_prepared", False)
    for name in jax_runtime.ALLOCATOR_VARS:
        monkeypatch.delenv(name, raising=False)
    records = []
    sink = logger.add(lambda m: records.append(m.record["message"]), level="WARNING")
    saved = snv.get_config()["verbose"]
    snv.set(verbose="WARNING")
    yield records
    snv.set(verbose=saved)
    logger.remove(sink)


def test_prepare_sets_the_variable_before_jax_starts(monkeypatch, fresh):
    monkeypatch.setattr(jax_runtime, "_started_backends", lambda: ())
    jax_runtime.prepare()
    assert os.environ[VAR] == "false"
    assert fresh == []


def test_a_value_the_user_set_wins(monkeypatch, fresh):
    monkeypatch.setattr(jax_runtime, "_started_backends", lambda: ())
    monkeypatch.setenv(VAR, "true")
    jax_runtime.prepare()
    assert os.environ[VAR] == "true"


def test_jax_on_a_gpu_without_the_variable_warns_once(monkeypatch, fresh):
    monkeypatch.setattr(jax_runtime, "_started_backends", lambda: ("cpu", "cuda"))
    jax_runtime.prepare()
    jax_runtime.prepare()
    assert VAR not in os.environ
    assert len(fresh) == 1 and VAR in fresh[0] and "Torch" in fresh[0]


@pytest.mark.parametrize("name", jax_runtime.ALLOCATOR_VARS)
def test_jax_on_a_gpu_with_an_allocator_setting_does_not_warn(monkeypatch, fresh, name):
    monkeypatch.setattr(jax_runtime, "_started_backends", lambda: ("cuda",))
    monkeypatch.setenv(name, "0.5")
    jax_runtime.prepare()
    assert fresh == []


def test_jax_on_the_cpu_does_not_warn(monkeypatch, fresh):
    monkeypatch.setattr(jax_runtime, "_started_backends", lambda: ("cpu",))
    jax_runtime.prepare()
    assert VAR not in os.environ
    assert fresh == []


def test_only_the_helper_names_the_variable():
    named = [p.relative_to(ROOT).as_posix() for p in (ROOT / "stable_envs").rglob("*.py") if VAR in p.read_text()]
    assert named == ["stable_envs/_core/jax_runtime.py"]


@pytest.mark.family("navigation")
def test_importing_the_package_and_building_a_non_jax_env_leaves_the_variable_unset():
    code = (
        "import os, sys, torch\n"
        "import stable_envs as snv\n"
        "from stable_envs import registry\n"
        "registry.discover()\n"
        "env = snv.make('TwoRoom', num_envs=2, simulation_backend='torch')\n"
        "env.reset(seed=0)\n"
        "env.step(torch.zeros(2, 2))\n"
        f"assert {VAR!r} not in os.environ, os.environ[{VAR!r}]\n"
        "print('ok')\n"
    )
    env = {k: v for k, v in os.environ.items() if k != VAR}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT), env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().endswith("ok"), r.stdout


@pytest.mark.family("navigation")
def test_building_a_jax_env_turns_preallocation_off_before_jax_starts():
    code = (
        "import os\n"
        "import stable_envs as snv\n"
        f"assert {VAR!r} not in os.environ\n"
        "env = snv.make('TwoRoom', num_envs=2, device='cpu', simulation_backend='jax')\n"
        "from jax._src import xla_bridge\n"
        "assert xla_bridge.backends_are_initialized()\n"
        f"print(os.environ[{VAR!r}])\n"
    )
    env = {k: v for k, v in os.environ.items() if k != VAR}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT), env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == "false", r.stdout
