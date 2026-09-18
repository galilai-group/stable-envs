"""Fast guards for the OSS benchmark CLI, configuration, and JSON contract."""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks import _common
from benchmarks import pipeline as pipeline_benchmark

ROOT = Path(__file__).resolve().parents[2]


def _run(script: str, output: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env["MUJOCO_GL"] = "glfw"
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    return subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / script), *args, "--output", str(output)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def test_common_parsers_reject_ambiguous_or_invalid_configuration():
    assert _common.positive_ints("1,16,64") == [1, 16, 64]
    assert _common.native_int("native") is None
    assert _common.native_int("2") == 2
    with pytest.raises(argparse.ArgumentTypeError):
        _common.positive_ints("1,0")
    with pytest.raises(argparse.ArgumentTypeError):
        _common.names("PushT,PushT")


def test_native_settings_are_omitted_from_public_constructor():
    args = argparse.Namespace(
        device="cpu", backend="auto", seed=7, size=None, supersample=None)
    kwargs = _common.env_kwargs(args, 8)
    assert kwargs == {"num_envs": 8, "device": "cpu", "backend": "auto", "seed": 7}


def test_unavailable_requested_cuda_is_an_error(monkeypatch):
    monkeypatch.setattr(_common.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="choose --device cpu"):
        _common.checked_device("cuda")


def test_renderer_cli_cpu_smoke_and_machine_output(tmp_path):
    output = tmp_path / "render.json"
    result = _run(
        "render.py", output,
        "--envs", "PushT", "--device", "cpu", "--batches", "1,2",
        "--warmup", "0", "--repeats", "2", "--threads", "1")
    assert result.returncode == 0, result.stderr
    record = json.loads(output.read_text())
    assert record["schema"] == _common.SCHEMA
    assert record["benchmark"] == "renderer-throughput"
    points = record["environments"]["PushT"]["points"]
    assert [point["batch"] for point in points] == [1, 2]
    assert all(point["status"] == "ok" for point in points)
    assert all(len(point["timing"]["samples_ms"]) == 2 for point in points)
    assert {"tier", "passed", "certification"}.isdisjoint(_all_keys(record))


def test_pipeline_cli_cpu_smoke_uses_public_state_contract(tmp_path):
    output = tmp_path / "pipeline.json"
    result = _run(
        "pipeline.py", output,
        "--envs", "PushT", "--device", "cpu", "--batch", "2",
        "--warmup", "0", "--repeats", "1", "--threads", "1")
    assert result.returncode == 0, result.stderr
    record = json.loads(output.read_text())
    row = record["environments"]["PushT"]
    assert record["benchmark"] == "host-state-render-pipeline"
    assert record["methodology"]["stable_worldmodel_required"] is False
    assert row["state_source"] == "env.reset(seed) -> env.pack(public State)"
    assert row["packed_state_shape"] == [2, 5]
    assert row["frame_shape"] == [96, 96, 3]


def test_unknown_environment_is_recorded_and_exits_nonzero(tmp_path):
    output = tmp_path / "error.json"
    result = _run(
        "render.py", output,
        "--envs", "DefinitelyNotAnEnvironment", "--batches", "1",
        "--warmup", "0", "--repeats", "1")
    assert result.returncode == 1
    row = json.loads(output.read_text())["environments"]["DefinitelyNotAnEnvironment"]["points"][0]
    assert row["status"] == "error"
    assert row["error_type"] == "KeyError"
    assert "unknown env" in row["error"]


def test_short_state_corpus_is_not_duplicated(tmp_path):
    path = tmp_path / "states.npy"
    np.save(path, np.zeros((1, 5), np.float32))
    fake_env = argparse.Namespace(state_dim=5)
    args = argparse.Namespace(state_file=str(path), state_key="states", batch=2)
    with pytest.raises(ValueError, match="will not duplicate rows silently"):
        pipeline_benchmark._load_states(fake_env, args)


@pytest.mark.parametrize("script", ["render.py", "pipeline.py"])
def test_scripts_do_not_import_stable_worldmodel(script):
    tree = ast.parse((ROOT / "benchmarks" / script).read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    assert not any(name.startswith("stable_worldmodel") for name in imported)
