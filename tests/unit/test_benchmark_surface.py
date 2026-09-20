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
import torch

from benchmarks import _common
from benchmarks import pipeline as pipeline_benchmark
from benchmarks import render as render_benchmark

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "tests" / "fixtures" / "regression" / "references" / "benchmarks"
FORBIDDEN_IMPORT_PREFIXES = (
    "stable_worldmodel",
    "stable_envs.testing",
    "stable_envs.render",
    "stable_envs.certification",
)


def _run(
    script: str, output: Path | str, *args: str, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env["MUJOCO_GL"] = "glfw"
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / script), *args, "--output", str(output)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported.update(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    return imported


def test_common_parsers_reject_ambiguous_or_invalid_configuration():
    assert _common.positive_ints("1,16,64") == [1, 16, 64]
    assert _common.native_int("native") is None
    assert _common.native_int("2") == 2
    with pytest.raises(argparse.ArgumentTypeError):
        _common.positive_ints("1,0")
    with pytest.raises(argparse.ArgumentTypeError):
        _common.names("PushT,PushT")
    with pytest.raises(argparse.ArgumentTypeError):
        _common.names("")
    with pytest.raises(argparse.ArgumentTypeError):
        _common.nonnegative_int("-1")


def test_native_settings_are_omitted_from_public_constructor():
    args = argparse.Namespace(device="cpu", backend="auto", seed=7, size=None, supersample=None)
    kwargs = _common.env_kwargs(args, 8)
    assert kwargs == {"num_envs": 8, "device": "cpu", "seed": 7}
    args.backend = "cuda"
    args.size = 96
    args.supersample = 2
    assert _common.env_kwargs(args, 8) == {
        "num_envs": 8,
        "device": "cpu",
        "seed": 7,
        "size": 96,
        "supersample": 2,
        "backend": "cuda",
    }


def test_unavailable_requested_cuda_is_an_error(monkeypatch):
    monkeypatch.setattr(_common.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="choose --device cpu"):
        _common.checked_device("cuda")


def test_missing_cuda_index_is_an_error(monkeypatch):
    monkeypatch.setattr(_common.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_common.torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="device_count"):
        _common.checked_device("cuda:1")


def test_measure_warmup_is_untimed_and_repeats_are_retained():
    calls = []

    def fn():
        calls.append(len(calls))

    timing = _common.measure(fn, device=torch.device("cpu"), warmup=3, repeats=4)
    assert calls == [0, 1, 2, 3, 4, 5, 6]
    assert len(timing["samples_ms"]) == 4
    assert timing["min_ms"] <= timing["median_ms"] <= timing["max_ms"]


def test_cuda_peak_reset_happens_after_warmup(monkeypatch):
    calls = []

    def fake_reset(device=None):
        calls.append(("reset", device))

    monkeypatch.setattr(_common, "synchronize", lambda device: None)
    monkeypatch.setattr(_common.torch.cuda, "reset_peak_memory_stats", fake_reset)
    _common.measure(lambda: calls.append("fn"), device=torch.device("cuda"), warmup=2, repeats=1)
    assert calls[:3] == ["fn", "fn", ("reset", torch.device("cuda"))]
    assert calls.count("fn") == 3


def test_configure_threads_does_not_override_exported_openmp(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    _common.configure_threads(1)
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["MKL_NUM_THREADS"] == "1"


def test_public_env_uses_canonical_class_not_make(monkeypatch):
    import stable_envs as snv

    captured = {}

    class Fake(snv.Env):
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(snv, "PushT", Fake, raising=False)

    def boom(*_args, **_kwargs):
        raise AssertionError("snv.make must not be used for a public class name")

    monkeypatch.setattr(snv, "make", boom)
    env = _common.public_env("PushT", {"num_envs": 2, "device": "cpu", "seed": 0})
    assert isinstance(env, Fake)
    assert captured == {"num_envs": 2, "device": "cpu", "seed": 0}


def test_renderer_cli_cpu_smoke_and_machine_output(tmp_path):
    output = tmp_path / "render.json"
    result = _run(
        "render.py",
        output,
        "--envs",
        "PushT",
        "--device",
        "cpu",
        "--batches",
        "1,2",
        "--warmup",
        "0",
        "--repeats",
        "2",
        "--threads",
        "1",
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(output.read_text())
    assert record["schema"] == _common.SCHEMA
    assert record["benchmark"] == "renderer-throughput"
    assert record["resolved_device"] == "cpu"
    assert record["config"]["warmup"] == 0
    assert record["config"]["repeats"] == 2
    points = record["environments"]["PushT"]["points"]
    assert [point["batch"] for point in points] == [1, 2]
    assert all(point["status"] == "ok" for point in points)
    assert all(len(point["timing"]["samples_ms"]) == 2 for point in points)
    assert {"tier", "passed", "certification", "badge", "best"}.isdisjoint(_all_keys(record))
    assert not (tmp_path / "render.json.tmp").exists()


def test_pipeline_cli_cpu_smoke_uses_public_state_contract(tmp_path):
    output = tmp_path / "pipeline.json"
    result = _run(
        "pipeline.py",
        output,
        "--envs",
        "PushT",
        "--device",
        "cpu",
        "--batch",
        "2",
        "--warmup",
        "0",
        "--repeats",
        "1",
        "--threads",
        "1",
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(output.read_text())
    row = record["environments"]["PushT"]
    assert record["benchmark"] == "host-state-render-pipeline"
    assert record["methodology"]["stable_worldmodel_required"] is False
    assert "jpeg_comparison" not in row
    assert row["state_source"] == "env.reset(seed) -> env.pack_render_state(public State)"
    assert row["packed_state_shape"] == [2, 5]
    assert row["frame_shape"] == [96, 96, 3]
    assert {"tier", "passed", "certification"}.isdisjoint(_all_keys(record))


def test_unknown_environment_is_recorded_and_exits_nonzero(tmp_path):
    output = tmp_path / "error.json"
    result = _run(
        "render.py",
        output,
        "--envs",
        "DefinitelyNotAnEnvironment",
        "--batches",
        "1",
        "--warmup",
        "0",
        "--repeats",
        "1",
    )
    assert result.returncode == 1
    row = json.loads(output.read_text())["environments"]["DefinitelyNotAnEnvironment"]["points"][0]
    assert row["status"] == "error"
    assert row["error_type"] == "KeyError"
    assert "unknown env" in row["error"]
    assert "DefinitelyNotAnEnvironment" in result.stderr


def test_short_state_corpus_is_not_duplicated(tmp_path):
    path = tmp_path / "states.npy"
    np.save(path, np.zeros((1, 5), np.float32))
    fake_env = argparse.Namespace(state_dim=5)
    args = argparse.Namespace(state_file=str(path), state_key="states", batch=2)
    with pytest.raises(ValueError, match="will not duplicate rows silently"):
        pipeline_benchmark._load_states(fake_env, args)


def test_oversized_point_is_skipped_not_failed(tmp_path):
    output = tmp_path / "skip.json"
    result = _run(
        "render.py",
        output,
        "--envs",
        "PushT",
        "--batches",
        "64",
        "--max-output-mib",
        "1",
        "--warmup",
        "0",
        "--repeats",
        "1",
    )
    assert result.returncode == 0, result.stderr
    point = json.loads(output.read_text())["environments"]["PushT"]["points"][0]
    assert point["status"] == "skipped"
    assert point["batch"] == 64
    assert "estimated_output_mib" in point


def test_stdout_output_keeps_diagnostics_on_stderr(tmp_path):
    result = _run(
        "render.py", "-", "--envs", "DefinitelyNotAnEnvironment", "--batches", "1", "--warmup", "0", "--repeats", "1"
    )
    assert result.returncode == 1
    record = json.loads(result.stdout)
    assert record["environments"]["DefinitelyNotAnEnvironment"]["points"][0]["status"] == "error"
    assert "DefinitelyNotAnEnvironment" in result.stderr


def test_optional_jpeg_dir_is_recorded_without_erasing_render(tmp_path):
    output = tmp_path / "pipeline.json"
    result = _run(
        "pipeline.py",
        output,
        "--envs",
        "PushT",
        "--device",
        "cpu",
        "--batch",
        "1",
        "--warmup",
        "0",
        "--repeats",
        "1",
        "--jpeg-dir",
        str(tmp_path / "missing"),
    )
    assert result.returncode == 1
    row = json.loads(output.read_text())["environments"]["PushT"]
    assert row["status"] == "ok"
    assert row["jpeg_comparison"]["status"] == "error"
    assert row["jpeg_comparison"]["error_type"] == "FileNotFoundError"


def test_optional_jpeg_baseline_uses_real_files(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    jpeg_dir = tmp_path / "frames"
    jpeg_dir.mkdir()
    for index in range(2):
        Image.new("RGB", (8, 8), color=(index, 0, 0)).save(jpeg_dir / f"{index}.jpg")
    output = tmp_path / "pipeline.json"
    result = _run(
        "pipeline.py",
        output,
        "--envs",
        "PushT",
        "--device",
        "cpu",
        "--batch",
        "2",
        "--warmup",
        "0",
        "--repeats",
        "1",
        "--jpeg-dir",
        str(jpeg_dir),
    )
    assert result.returncode == 0, result.stderr
    row = json.loads(output.read_text())["environments"]["PushT"]
    comparison = row["jpeg_comparison"]
    assert comparison["status"] == "ok"
    assert comparison["files_per_call"] == 2
    assert comparison["frame_shape"] == [8, 8, 3]
    assert "render_fps_over_jpeg_fps" in row


def test_cli_refuses_unavailable_cuda_without_json(tmp_path, monkeypatch):
    monkeypatch.setattr(render_benchmark.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as exc:
        render_benchmark.main(
            [
                "--device",
                "cuda",
                "--envs",
                "PushT",
                "--batches",
                "1",
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--output",
                str(tmp_path / "no.json"),
            ]
        )
    assert exc.value.code == 2
    assert not (tmp_path / "no.json").exists()


@pytest.mark.parametrize("script", ["__init__.py", "_common.py", "render.py", "pipeline.py"])
def test_scripts_do_not_import_worldmodel_testing_or_render_internals(script):
    imported = _imported_modules(ROOT / "benchmarks" / script)
    assert not any(
        name == prefix or name.startswith(prefix + ".") for name in imported for prefix in FORBIDDEN_IMPORT_PREFIXES
    )


@pytest.mark.parametrize(
    "filename,benchmark",
    [
        ("render_cpu_local.json", "renderer-throughput"),
        ("pipeline_cpu_local.json", "host-state-render-pipeline"),
    ],
)
def test_committed_cpu_evidence_matches_schema(filename, benchmark):
    path = EVIDENCE / filename
    if not path.is_file():
        pytest.skip("CPU evidence JSON is not present")
    record = json.loads(path.read_text())
    assert record["schema"] == _common.SCHEMA
    assert record["benchmark"] == benchmark
    assert record["config"]["device"] == "cpu"
    assert record["config"]["warmup"] == 2
    assert record["config"]["repeats"] == 7
    assert {"tier", "passed", "certification"}.isdisjoint(_all_keys(record))
    assert "/oscar/" not in path.read_text()
    environments = record["environments"]
    assert environments
    for row in environments.values():
        points = row["points"] if "points" in row else [row]
        assert all(point["status"] == "ok" for point in points)
        for point in points:
            assert len(point["timing"]["samples_ms"]) == 7
