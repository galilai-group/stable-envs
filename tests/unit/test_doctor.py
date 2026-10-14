"""``stable-envs doctor`` and the two point-of-failure checks it shares with the render paths."""

from __future__ import annotations

import json
import subprocess
import sys
import types

import pytest

from stable_envs import cli, doctor
from stable_envs._core import mujoco_gl, warp_cuda

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("environ", "platform_name", "expected"),
    [
        ({}, "linux", "egl"),
        ({"MUJOCO_GL": "osmesa"}, "linux", "osmesa"),
        ({}, "darwin", "cgl"),
        ({"MUJOCO_GL": "egl"}, "darwin", "cgl"),
        ({"MUJOCO_GL": "glfw"}, "darwin", "glfw"),
    ],
)
def test_selected_backend(environ, platform_name, expected):
    assert mujoco_gl.selected_backend(environ, platform_name) == expected


@pytest.mark.parametrize(
    ("environ", "name", "status"),
    [
        ({"MUJOCO_GL": "glfw"}, "DISPLAY", doctor.FAIL),
        ({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "osmesa"}, "PYOPENGL_PLATFORM", doctor.FAIL),
        ({"MUJOCO_GL": "glfw", "DISPLAY": ":0", "PYOPENGL_PLATFORM": "egl"}, "PYOPENGL_PLATFORM", doctor.WARN),
        ({"MUJOCO_EGL_DEVICE_ID": "0", "CUDA_VISIBLE_DEVICES": "2,3"}, "MUJOCO_EGL_DEVICE_ID", doctor.WARN),
        ({"MUJOCO_GL": "wgl"}, "MUJOCO_GL", doctor.FAIL),
        ({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}, "MUJOCO_GL", doctor.OK),
    ],
)
def test_gl_environment_checks(environ, name, status):
    checks = {c.name: c for c in doctor.gl_env_checks(environ, "linux")}
    assert checks[name].status == status
    assert all(c.status == doctor.OK for c in checks.values()) == (status == doctor.OK)


def _fake_mujoco(monkeypatch, renderer=None):
    module = types.ModuleType("mujoco")
    if renderer is not None:
        module.Renderer = renderer
    monkeypatch.setitem(sys.modules, "mujoco", module)
    monkeypatch.setenv("MUJOCO_GL", "egl")


def _raises(exc):
    def renderer(*args, **kwargs):
        raise exc

    return renderer


def test_missing_mujoco_renderer_names_the_backend_and_the_fix(monkeypatch):
    _fake_mujoco(monkeypatch)
    with pytest.raises(RuntimeError, match="no OpenGL renderer with MUJOCO_GL=egl") as info:
        mujoco_gl.mujoco_renderer(object(), height=8, width=8)
    assert "MUJOCO_GL=osmesa" in str(info.value) and "libosmesa6" in str(info.value)


def test_a_failed_gl_context_is_chained_and_a_size_error_is_not(monkeypatch):
    cause = RuntimeError("Cannot initialize a EGL device display")
    _fake_mujoco(monkeypatch, _raises(cause))
    with pytest.raises(RuntimeError, match="could not create an OpenGL context") as info:
        mujoco_gl.mujoco_renderer(object(), height=8, width=8)
    assert info.value.__cause__ is cause
    _fake_mujoco(monkeypatch, _raises(ValueError("Image width 9 > framebuffer width 8")))
    with pytest.raises(ValueError, match="framebuffer"):
        mujoco_gl.mujoco_renderer(object(), height=9, width=9)


def _fake_warp(monkeypatch, available, toolkit=(13, 0), driver=(12, 9)):
    runtime = types.SimpleNamespace(toolkit_version=toolkit, driver_version=driver)
    module = types.ModuleType("warp")
    module.init = lambda: None
    module.is_cuda_available = lambda: available
    module.context = types.SimpleNamespace(runtime=runtime)
    module.config = types.SimpleNamespace(version="1.18.0", quiet=False)
    monkeypatch.setitem(sys.modules, "warp", module)
    monkeypatch.setitem(sys.modules, "warp._src", None)
    return module


@pytest.mark.parametrize(
    ("toolkit", "driver", "words"),
    [
        ((13, 0), (12, 9), ("CUDA 13.0", "CUDA 12.9", "580.65.06", f"warp-lang=={warp_cuda.CERTIFIED_WARP}")),
        ((12, 8), None, ("no NVIDIA CUDA driver",)),
        (None, None, ("without CUDA support",)),
        ((12, 8), (12, 9), ("sees no CUDA device",)),
    ],
)
def test_warp_cpu_fallback_is_explained(monkeypatch, toolkit, driver, words):
    _fake_warp(monkeypatch, False, toolkit, driver)
    problem = warp_cuda.warp_cuda_problem()
    assert all(w in problem for w in words), problem
    with pytest.raises(RuntimeError, match="Fetch GPU rendering needs Warp on CUDA"):
        warp_cuda.require_warp_cuda("Fetch GPU rendering")


def test_warp_on_cuda_passes(monkeypatch):
    module = _fake_warp(monkeypatch, True)
    assert warp_cuda.warp_cuda_problem() is None
    assert warp_cuda.require_warp_cuda("x") is module


@pytest.mark.parametrize(
    ("result", "status", "detail"),
    [
        (subprocess.CompletedProcess([], 0, '{"renderer": "llvmpipe"}\n', ""), doctor.OK, "llvmpipe"),
        (subprocess.CompletedProcess([], 1, '{"error": "ImportError: no EGL"}\n', ""), doctor.FAIL, "no EGL"),
        (subprocess.CompletedProcess([], -11, "", ""), doctor.FAIL, "signal 11"),
        (subprocess.TimeoutExpired("python", 60), doctor.FAIL, "timed out"),
    ],
)
def test_gl_probe_outcomes(monkeypatch, result, status, detail):
    def run(cmd, env, **kwargs):
        assert env["MUJOCO_GL"] == env["PYOPENGL_PLATFORM"] == "osmesa"
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(doctor.subprocess, "run", run)
    got_status, got_detail = doctor.probe_gl("osmesa", {"PYOPENGL_PLATFORM": "egl"})
    assert got_status == status and detail in got_detail


DMC = next(f for f in doctor.FAMILIES if f.name == "dmcontrol")


def _caps(**overrides):
    caps = {name: doctor.Check(name, doctor.OK) for name in ("gl", "cuda", "warp", "triton", "nvdiffrast")}
    caps.update(overrides)
    return caps


@pytest.mark.parametrize(
    ("installed", "errors", "caps", "has_gpu", "expected"),
    [
        (False, {}, _caps(), True, (doctor.SKIP, doctor.SKIP, doctor.SKIP, "stable-envs[dmcontrol]")),
        (True, {"dmcontrol": "ImportError: skew"}, _caps(), True, (doctor.FAIL, doctor.FAIL, doctor.FAIL, "")),
        (True, {}, _caps(), False, (doctor.OK, doctor.OK, doctor.SKIP, "")),
        (
            True,
            {},
            _caps(gl=doctor.Check("gl", doctor.FAIL, "egl: no EGL", "export MUJOCO_GL=osmesa")),
            True,
            (doctor.FAIL, doctor.FAIL, doctor.FAIL, "MUJOCO_GL=osmesa"),
        ),
        (
            True,
            {},
            _caps(warp=doctor.Check("warp", doctor.FAIL, "not installed")),
            True,
            (doctor.WARN, doctor.OK, doctor.WARN, "stable-envs[dmcontrol-gpu]"),
        ),
    ],
)
def test_family_verdicts(monkeypatch, installed, errors, caps, has_gpu, expected):
    monkeypatch.setattr(doctor, "_dist_version", lambda name: "1.0" if installed else None)
    row = doctor.family_report(DMC, errors, caps, has_gpu)
    assert (row.status, row.cpu, row.gpu) == expected[:3]
    assert expected[3] in row.fix


@pytest.mark.parametrize(("family_status", "code"), [(doctor.WARN, 0), (doctor.FAIL, 1)])
def test_cli_exit_code_and_json(monkeypatch, capsys, family_status, code):
    report = doctor.Report(
        system=[doctor.Check("python", doctor.OK, "3.12")],
        families=[doctor.FamilyReport("fetch", family_status, doctor.OK, family_status, fix="do this")],
    )
    monkeypatch.setattr(doctor, "run", lambda: report)
    assert cli.main(["doctor", "--json"]) == code
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is (code == 0) and out["families"][0]["fix"] == "do this"
    assert cli.main(["doctor"]) == code
    assert "fix: do this" in capsys.readouterr().out
