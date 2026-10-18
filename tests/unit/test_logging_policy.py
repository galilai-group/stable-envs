"""The logging policy: one gated logger, ``STABLE_ENVS_VERBOSE`` as the default level, no handler changes."""

from __future__ import annotations

import subprocess
import sys

import pytest
from loguru import logger

from stable_envs import env as env_module
from stable_envs._core import config, environ

pytestmark = pytest.mark.unit


@pytest.fixture
def sink():
    lines: list[str] = []
    sink_id = logger.add(lambda m: lines.append(m.record["message"]), level="DEBUG")
    saved = config.get_config()["verbose"]
    yield lines
    logger.remove(sink_id)
    config.set(verbose=saved)


def test_debug_is_hidden_at_warning_and_shown_at_debug(sink):
    config.set(verbose="WARNING")
    config.log("DEBUG", "hidden")
    config.log("WARNING", "shown")
    assert sink == ["shown"]
    config.set(verbose="DEBUG")
    config.log("DEBUG", "now shown")
    assert sink == ["shown", "now shown"]


def test_records_can_be_disabled_by_package(sink):
    config.set(verbose="DEBUG")
    logger.disable("stable_envs")
    try:
        config.log("WARNING", "silenced")
    finally:
        logger.enable("stable_envs")
    assert sink == []


def test_the_environment_variable_sets_the_default_and_set_overrides_it(monkeypatch):
    saved = config.get_config()["verbose"]
    try:
        monkeypatch.setenv("STABLE_ENVS_VERBOSE", "debug")
        environ._parsed.cache_clear()
        assert config.reset_config()["verbose"] == "DEBUG"
        assert config.set(verbose="ERROR")["verbose"] == "ERROR"
        monkeypatch.setenv("STABLE_ENVS_VERBOSE", "chatty")
        with pytest.raises(ValueError, match="STABLE_ENVS_VERBOSE"):
            config.reset_config()
    finally:
        monkeypatch.delenv("STABLE_ENVS_VERBOSE", raising=False)
        environ._parsed.cache_clear()
        config.reset_config()
        config.set(verbose=saved)


def test_auto_device_falling_back_to_cpu_logs_info_once(sink, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    env_module._log_auto_cpu.cache_clear()
    config.set(verbose="INFO")
    for _ in range(3):
        assert env_module.Env.resolve_device("auto") == "cpu"
    assert len([m for m in sink if "chose the CPU" in m]) == 1


def test_importing_the_package_leaves_logging_handlers_alone():
    code = (
        "import logging\n"
        "from loguru import logger\n"
        "before = (list(logging.getLogger().handlers), dict(logger._core.handlers))\n"
        "import stable_envs\n"
        "after = (list(logging.getLogger().handlers), dict(logger._core.handlers))\n"
        "assert before == after\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
