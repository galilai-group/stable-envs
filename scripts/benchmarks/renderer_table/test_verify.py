"""Adversarial verifier checks; synthetic fixtures are never report measurements."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from scripts.benchmarks.renderer_table.common import METRICS, catalog, distribution, quality
from scripts.benchmarks.renderer_table.verify import validate


def test_state_archive_preserves_literal_separators(tmp_path):
    from scripts.benchmarks.renderer_table.common import equal_state, load_state, save_state, state_arrays

    state = TensorDict({("variation", "rng", "parent/child"): torch.arange(4)[None]}, [1])
    new = tmp_path / "new.npz"
    save_state(new, state)
    equal_state(state, load_state(new))
    assert set(state.keys(True, True)) == set(load_state(new).keys(True, True))
    old = tmp_path / "legacy.npz"
    np.savez_compressed(old, **state_arrays(state))
    equal_state(state, load_state(old, template=state))
    assert set(state.keys(True, True)) == set(load_state(old, template=state).keys(True, True))


@pytest.fixture
def valid():
    metric = quality(np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 3), np.uint8))
    sha = "a" * 64
    frames = []
    for i in range(16):
        frames.append(
            dict(
                index=i,
                seed=[0, 17, 42][i % 3],
                state_hash=sha,
                prefix_hash=sha,
                camera="test camera",
                shape=[2, 2, 3],
                oracle_sha256=metric["oracle_sha256"],
                oracle_render_nonmutation=True,
                candidate_render_nonmutation=True,
                proof=dict(
                    kind="EXACT_NUMERIC_STATE",
                    equal=True,
                    candidate_state_hash=sha,
                    oracle_state_hash=sha,
                    compared_leaves=1,
                ),
                cpu_quality=copy.deepcopy(metric),
                gpu=dict(
                    state_hash=sha,
                    camera="test camera",
                    shape=[2, 2, 3],
                    supplied_and_live_state_nonmutation=True,
                    gpu_quality=copy.deepcopy(metric),
                ),
            )
        )
    cells = {}
    for backend in ["cpu", "gpu", "original"]:
        cells[backend] = dict(
            status="MEASURED",
            samples_ms=[1.0] * 100,
            statistics_ms=distribution([1.0] * 100),
            minimum_warmups=10,
            includes_reset_step_get_set_state=False,
            synchronization="synchronize before and after" if backend == "gpu" else "CPU eager",
            output_location="cuda" if backend == "gpu" else "cpu",
            execution_device="cuda" if backend == "gpu" else "cpu",
            renderer_backend="test fixture software renderer",
        )
    aggregate = {k: distribution([metric[k]] * 16) for k in METRICS}
    rows = []
    for row in catalog().values():
        rows.append(
            dict(
                row,
                frames=copy.deepcopy(frames),
                matched_frames=16,
                camera="test camera",
                shape=[2, 2, 3],
                timings=copy.deepcopy(cells),
                quality={"cpu": copy.deepcopy(aggregate), "gpu": copy.deepcopy(aggregate)},
                profiles={"cpu": {"host": "test"}, "gpu": {"host": "test"}},
            )
        )
        if row["name"].startswith("Craftax"):
            leaves = {"meta/signature": "a" * 64, "engine/map": "b" * 64}
            digest = hashlib.sha256(json.dumps(leaves, sort_keys=True).encode()).hexdigest()
            for f in rows[-1]["frames"]:
                f["state_hash"] = f["gpu"]["state_hash"] = digest
                f["proof"]["candidate_state_hash"] = f["proof"]["oracle_state_hash"] = digest
                f["gpu"]["device_replay_proof"] = dict(
                    kind="EXACT_REPLAY_EXCEPT_PLATFORM_SIGNATURE",
                    excluded_leaf="meta/signature",
                    reason="synthetic verifier test",
                    cpu_leaf_hashes=leaves.copy(),
                    gpu_leaf_hashes=leaves.copy(),
                    actual_gpu_state_hash=digest,
                    action_prefix_hash=sha,
                )
    return dict(metric_definitions=METRICS.copy(), registrations=rows)


def test_complete_fixture(valid):
    assert validate(valid)["registrations"] == 42


def test_craftax_engine_difference_rejected(valid):
    row = next(r for r in valid["registrations"] if r["name"].startswith("Craftax"))
    row["frames"][0]["gpu"]["device_replay_proof"]["gpu_leaf_hashes"]["engine/map"] = "c" * 64
    with pytest.raises(AssertionError):
        validate(valid)


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "duplicate",
        "state",
        "oracle_frame",
        "camera",
        "resolution",
        "sync",
        "cpu_gpu",
        "definition",
        "count",
        "nan",
        "reason",
        "aggregation",
    ],
)
def test_rejects_corruption(valid, damage):
    row = valid["registrations"][0]
    frame = row["frames"][0]
    if damage == "missing":
        valid["registrations"].pop()
    elif damage == "duplicate":
        valid["registrations"][1] = copy.deepcopy(row)
    elif damage == "state":
        frame["gpu"]["state_hash"] = "b" * 64
    elif damage == "oracle_frame":
        frame["gpu"]["gpu_quality"]["oracle_sha256"] = "b" * 64
    elif damage == "camera":
        frame["gpu"]["camera"] = "other camera"
    elif damage == "resolution":
        frame["gpu"]["shape"] = [4, 4, 3]
    elif damage == "sync":
        row["timings"]["gpu"]["synchronization"] = "none"
    elif damage == "cpu_gpu":
        row["timings"]["original"]["execution_device"] = "cuda"
    elif damage == "definition":
        valid["metric_definitions"].pop("mse_normalized")
    elif damage == "count":
        row["timings"]["gpu"]["samples_ms"].pop()
    elif damage == "nan":
        frame["cpu_quality"]["mse_normalized"] = float("nan")
    elif damage == "reason":
        row["timings"]["cpu"] = {"status": "UNSUPPORTED", "reason": ""}
    elif damage == "aggregation":
        row["quality"]["gpu"]["mse_normalized"]["count"] -= 1
    with pytest.raises(AssertionError):
        validate(valid)
