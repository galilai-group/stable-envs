"""Synthetic report checks: incomplete or wrong-device evidence must not be published."""

import copy
import hashlib
import itertools
import json

import pytest
import torch

from benchmarks.online.dmc_source_pipeline import compare_tasks, tasks_from_public
from benchmarks.online.summarize_dmc_source import OPERATIONS, summarize


@pytest.fixture
def synthetic(tmp_path):
    source = tmp_path / "candidate"
    upstream = tmp_path / "upstream"
    source.mkdir()
    upstream.mkdir()
    candidate_files = {}
    for f in ("benchmarks/online/dmc_source_pipeline.py",
              "benchmarks/online/dmc_parallel_source.py",
              "benchmarks/online/classic_parallel_source.py",
              "benchmarks/online/common.py", "stable_envs/_core/stepping.py", "stable_envs/env.py"):
        path = source / f
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic fixture, not benchmark evidence")
        candidate_files[f] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = {}
    for i in range(20):
        f = ("stable_worldmodel/envs/dmcontrol/dmcontrol.py" if i == 0 else
             "stable_worldmodel/envs/dmcontrol/acrobot.py" if i == 1 else f"source{i}.py")
        (upstream / f).parent.mkdir(parents=True, exist_ok=True)
        (upstream / f).write_text(f"synthetic source {i}")
        hashes[f] = hashlib.sha256((upstream / f).read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(commit="176d54532b3c0a251a4f509b74f2fdacc143a63d", files=hashes)))
    results = tmp_path / "results"
    for pid, (mode, device) in enumerate(itertools.product(("source_serial", "source_parallel", "ours"), ("cpu", "cuda"))):
        identity = dict(renderer="synthetic llvmpipe" if device == "cpu" else "synthetic GPU", vendor="Mesa" if device == "cpu" else "NVIDIA", software=device == "cpu")
        actor_count = 0 if mode == "ours" else (1 if mode == "source_serial" else 8)
        cells = []
        for horizon, selection in itertools.product((1, 8), ("none", "final", "all")):
            frames = 64 * (horizon if selection == "all" else int(selection == "final"))
            mse = 0.0 if mode == "ours" and frames else None
            scale = {"source_serial": 4, "source_parallel": 2, "ours": 1}[mode]
            cells.append(dict(batch=64, horizon=horizon, selection=selection, source_gl=identity,
                actor_gl=[identity] * actor_count, physics_workers=8 if mode == "ours" else actor_count,
                valid_steps=64 * horizon, frames_checked=frames, exact_state_task="PASS", output_ownership="PASS",
                initial_state_sha256="0" * 64, endpoint_state_sha256="1" * 64, source_tasks_sha256="2" * 64,
                mse_mean=mse, mse_sd=mse, mse_max=mse, seconds=[scale * t for t in (.8, 1, 1.2, .8, 1, 1.2)]))
        report = dict(status="PASS", gate_only=False, mode=mode, device=device, affinity=list(range(8)),
            source_workers=8, parent_torch_threads=8, source_threads_per_worker=1, size=64,
            gpu=None if device == "cpu" else "synthetic RTX 3090", source_physics_device="cpu", source_render_device=device,
            source_gpu_supported=True, source_readback_in_timing=True, source_state_export_in_timing=False,
            ours_state_history=False, ours_mandatory_final_state=True, oracle_closed_before_population=True,
            fresh_strategy_process=True, pid=pid, source_pin="176d54532b3c0a251a4f509b74f2fdacc143a63d",
            source_files=hashes, domain="acrobot", host="synthetic", job="synthetic", versions={n: "synthetic" for n in ("torch", "numpy", "gymnasium", "mujoco", "dm-control", "tensordict", "mujoco-warp", "warp-lang")},
            source_runtime_files={str(upstream / f): h for f, h in hashes.items()}, candidate_files=candidate_files, initial_state_policy="synthetic", action_policy="synthetic",
            opengl=identity, cells=cells, components=[dict(batch=64, operation=op, seconds=[1., 2., 3., 1., 2., 3.])
            for op in OPERATIONS] if mode == "ours" else [])
        path = results / mode / f"{device}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report))
    return results, source, upstream, manifest, [64]


def test_complete_synthetic_join_counts_and_units(synthetic):
    rows, components = summarize(*synthetic)
    assert len(rows) == 12 and len(components) == 6
    assert all(r["ours_vs_serial"] == 4 and r["ours_vs_parallel"] == 2 and r["ours_vs_best_source"] == 2 for r in rows)
    assert {r["rate_unit"] for r in components} == {"frames/s", "steps/s", "states/s"}
    assert all(r["ours_fps_mean"] is None for r in rows if r["selection"] == "none")


@pytest.mark.parametrize("mutation", ["missing_cell", "wrong_gl", "missing_worker_gl", "wrong_state", "bad_timing", "missing_component", "wrong_host", "same_pid", "missing_mse", "wrong_loaded_source"])
def test_reject_incomplete_or_incomparable_evidence(synthetic, mutation):
    results = synthetic[0]
    mode = "source_parallel" if mutation == "missing_worker_gl" else "ours"
    path = results / mode / "cuda.json"
    report = json.loads(path.read_text())
    if mutation == "missing_cell":
        report["cells"].pop()
    elif mutation == "wrong_gl":
        report["opengl"] = dict(renderer="llvmpipe", vendor="Mesa", software=True)
    elif mutation == "missing_worker_gl":
        report["cells"][0]["actor_gl"].pop()
    elif mutation == "wrong_state":
        report["cells"][0]["endpoint_state_sha256"] = "f" * 64
    elif mutation == "bad_timing":
        report["cells"][0]["seconds"][0] = float("nan")
    elif mutation == "missing_component":
        report["components"].pop()
    elif mutation == "wrong_host":
        report["host"] = "different"
    elif mutation == "same_pid":
        report["pid"] = 0
    elif mutation == "missing_mse":
        report["cells"][1]["mse_max"] = None
    elif mutation == "wrong_loaded_source":
        report["source_runtime_files"] = {}
    path.write_text(json.dumps(report))
    with pytest.raises((AssertionError, TypeError)):
        summarize(*synthetic)


def test_image_mismatch_and_nan_task_comparison():
    expected = {"success": torch.tensor([[float("nan")]]), "frames": torch.zeros(1, 1, 2, 2, 3, dtype=torch.uint8)}
    actual = copy.deepcopy(expected)
    assert compare_tasks(actual, expected, library=True) == [0.0]
    actual["frames"].fill_(255)
    with pytest.raises(AssertionError):
        compare_tasks(actual, expected, library=True)


def test_single_step_normalization_preserves_frame_axis():
    from tensordict import TensorDict

    result = TensorDict(dict(state=TensorDict({"qpos": torch.zeros(2, 1)}, [2]),
        valid=torch.ones(2, 1, dtype=torch.bool), frame_valid=torch.ones(2, 1, dtype=torch.bool),
        frames=torch.zeros(2, 1, 4, 4, 3, dtype=torch.uint8), reward=torch.zeros(2)), [2])
    tasks = tasks_from_public(result, 1)
    assert set(tasks) == {"frames", "reward"}
    assert tasks["frames"].shape == (2, 1, 4, 4, 3)
    assert tasks["reward"].shape == (2, 1)


@pytest.mark.parametrize("options", [
    ["--benchmark", "dmc_source_pipeline"],
    ["--benchmark", "dmc_pipeline", "--source-mode", "ours"],
])
def test_coordinator_rejects_incomplete_or_misapplied_source_options(tmp_path, monkeypatch, options):
    import sys

    from benchmarks.online import dmc_state_export_oscar

    output = tmp_path / "should-not-exist"
    monkeypatch.setattr(sys, "argv", ["coordinator", "--domain", "acrobot", "--output", str(output), *options])
    with pytest.raises(SystemExit) as error:
        dmc_state_export_oscar.main()
    assert error.value.code == 2
    assert not output.exists()
