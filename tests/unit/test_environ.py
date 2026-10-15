"""Every ``STABLE_ENVS_*`` variable is registered, documented and read through one module."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from stable_envs import doctor
from stable_envs._core import environ

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
NAME = re.compile(r"STABLE_ENVS_[A-Z0-9_]*[A-Z0-9]")


def test_only_the_registry_reads_stable_envs_variables():
    offenders = []
    for path in sorted((ROOT / "stable_envs").rglob("*.py")):
        if path == ROOT / "stable_envs/_core/environ.py":
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if "STABLE_ENVS_" in line and re.search(r"environ\b|getenv", line) and "_environ." not in line:
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert offenders == []


def test_the_documentation_lists_exactly_the_registered_variables():
    page = (ROOT / "docs/environment-variables.md").read_text()
    assert set(NAME.findall(page)) == set(environ.variables())


def test_every_variable_names_itself_and_has_a_doc():
    for name, var in environ.variables().items():
        assert name == var.name and name.startswith(environ.PREFIX) and var.doc


@pytest.mark.parametrize(("text", "value"), [("1", True), ("0", False), ("", False), (" 1 ", True)])
def test_flags_parse(monkeypatch, text, value):
    monkeypatch.setenv("STABLE_ENVS_AUTOTUNE", text)
    environ._parsed.cache_clear()
    assert environ.read("STABLE_ENVS_AUTOTUNE") is value


@pytest.mark.parametrize("text", ["yes", "true", "2"])
def test_a_bad_flag_names_the_variable(monkeypatch, text):
    monkeypatch.setenv("STABLE_ENVS_TUNING_HINT", text)
    with pytest.raises(ValueError, match="STABLE_ENVS_TUNING_HINT"):
        environ.read("STABLE_ENVS_TUNING_HINT")


def test_unset_variables_take_their_default(monkeypatch):
    monkeypatch.delenv("STABLE_ENVS_TUNING_HINT", raising=False)
    monkeypatch.delenv("STABLE_ENVS_OGB_UNMEASURED", raising=False)
    assert environ.read("STABLE_ENVS_TUNING_HINT") is True
    assert environ.read("STABLE_ENVS_OGB_UNMEASURED") is False


def test_doctor_lists_set_variables_and_flags_unknown_names():
    assert doctor.check_variables({}).detail == "none set"
    ok = doctor.check_variables({"STABLE_ENVS_CPU3D_CHUNK": "8", "HOME": "/x"})
    assert ok.status == doctor.OK and ok.detail == "STABLE_ENVS_CPU3D_CHUNK=8"
    typo = doctor.check_variables({"STABLE_ENVS_CPU3D_CHUNKS": "8"})
    assert typo.status == doctor.WARN and "unknown: STABLE_ENVS_CPU3D_CHUNKS" in typo.detail
    bad = doctor.check_variables({"STABLE_ENVS_CPU3D_CHUNK": "0"})
    assert bad.status == doctor.WARN and "STABLE_ENVS_CPU3D_CHUNK" in bad.detail
