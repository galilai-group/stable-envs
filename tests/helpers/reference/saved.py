"""Read-only checks over the saved native and Rocket corpora.

They verify recorded measurements without regenerating trajectories.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from stable_envs.testing.evidence import sha256
from stable_envs.testing.metrics import normalized_mse, summarize_errors
from stable_envs.testing.provenance import verify_recorded_code
from tests.helpers.reference.cases import CaseKey, coverage

SWM_PIN = "21446f1ede6d5284e981bd7b47f432b994e6d812"
# Commit the frozen native corpora were recorded against; their envs are unchanged at SWM_PIN.
CORPUS_PIN = "176d54532b3c0a251a4f509b74f2fdacc143a63d"
ROCKET_SOURCE = "5ede3640ce1e2add32b6b0a6c36e190bb1130e38bb258ab182af12a6fbedee23"

# Absolute roundoff budget for cross-machine frozen trajectories only: Linux x86_64
# against the macOS arm64 recordings differs by up to 1.1e-11 (DM Control
# contact-heavy velocities). Live source contracts and State restoration stay exact.
FROZEN_FLOAT64_ATOL = 1e-9


def assert_frozen_equal(actual, expected, *, equal_nan=False, msg=None):
    """Floats within FROZEN_FLOAT64_ATOL; integers, bools, shape and dtype exact."""
    actual = actual.detach().cpu() if torch.is_tensor(actual) else torch.as_tensor(actual)
    expected = expected.detach().cpu() if torch.is_tensor(expected) else torch.as_tensor(expected)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0,
        atol=FROZEN_FLOAT64_ATOL if expected.is_floating_point() else 0,
        equal_nan=equal_nan,
        msg=msg,
    )


def assert_frozen_state(actual, expected):
    assert actual.batch_size == expected.batch_size
    assert set(actual.keys(True, True)) == set(expected.keys(True, True))
    for key in expected.keys(True, True):
        try:
            assert_frozen_equal(actual[key], expected[key])
        except AssertionError as error:
            raise AssertionError(f"Frozen State {key}: {error}") from error


def native_corpus(directory, domain):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["reference_commit"] == CORPUS_PIN
    entry = manifest["domains"][domain]
    path = root / entry["file"]
    assert path.resolve().parent == root.resolve()
    assert sha256(path) == entry["sha256"], "native corpus hash"
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(entry["arrays"]), "native corpus fields"
        arrays = {key: archive[key] for key in archive.files}
    for key, spec in entry["arrays"].items():
        assert list(arrays[key].shape) == spec["shape"] and str(arrays[key].dtype) == spec["dtype"], key
    assert arrays["actions"].shape[:2] == (manifest["steps"], manifest["batch"])
    return manifest, arrays


def rocket_cases():
    checkpoints = {
        "fall": (0, 1, 100, 300, 301, 302, 303, 310),
        "mixed": (0, 1, 12, 100),
        "success": (0, 1, 6, 7, 12),
        "horizon": (0, 300, 600, 900, 1201, 1202, 1204),
        "all": (0, 1, 12),
    }
    base = [
        (f"{label}_{step}", seed, dtype)
        for label, steps in checkpoints.items()
        for step in steps
        for seed in (0, 17, 42)
        for dtype in ("float32", "float64")
    ]
    variations = (
        "rocket.body_color",
        "rocket.booster_color",
        "rocket.fin_color",
        "rocket.leg_color",
        "pad.color",
        "environment.sky_color",
        "environment.start_height_ratio",
        "environment.start_horizontal_offset",
        "environment.start_tilt",
        "rocket",
        "pad",
        "environment",
        "rocket_rocket.body_color",
        "environment_environment.start_tilt",
    )
    return base + [("variation_" + label, seed, "float64") for label in variations for seed in (0, 17, 42)]


def verify_rocket_render(directory, candidate_root, *, revision=None):
    """Verify the saved 204-case Rocket render report, including its per-group visual criteria."""
    root = Path(directory)
    report = json.loads((root / "result.json").read_text())
    assert report["status"] in ("PUBLIC_CPU_RENDER_PASS", "PUBLIC_GPU_PASS")
    assert report["per_frame_mse_pass"] is True
    assert report["oracle_source"]["sha256"] == ROCKET_SOURCE
    assert report["versions"]["PyFlyt"] == "0.29.0" and report["versions"]["pybullet"] == "3.2.7"
    identity = verify_recorded_code(report["production_hashes"], candidate_root, revision=revision)

    def key(label, seed, dtype):
        return CaseKey("swm/PFRocketLanding-v0", "render", f"{label}/{seed}/{dtype}", report["device"])

    required = [key(*case) for case in rocket_cases()]
    observed = [(key(row["label"], row["seed"], row["dtype"]), "PASS") for row in report["records"]]
    accounting = coverage(required, observed)
    assert accounting["status"] == "PASS", accounting
    assert set(report["groups"]) == {label for label, _, _ in rocket_cases()}
    with np.load(root / "frames.npz", allow_pickle=False) as archive:
        ours, source = archive["ours"], archive["oracle"]
    assert ours.shape == source.shape == (204, 480, 480, 3)
    errors = normalized_mse(ours, source)
    for error, row in zip(errors, report["records"], strict=True):
        assert all(
            row[k] is True for k in ("exact_source_numeric", "independent_restore_exact", "immediate_pixels_exact")
        )
        assert np.isclose(error, row["metrics"]["normalized_mse"], rtol=1e-12, atol=0)
        assert error <= 0.001
    from stable_envs.testing.conformance import ssim

    # Fixed limits for this report format, independent of the env's declared thresholds.
    limits = dict(mean_abs_max=10.0, within8_min=0.91, ssim_min=0.80)
    for label, group in report["groups"].items():
        indices = [i for i, row in enumerate(report["records"]) if row["label"] == label]
        assert len(indices) == group["count"] and group["passed"] is True
        a, b = ours[indices], source[indices]
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
        mean_abs, within8 = float(delta.mean()), float((delta <= 8).mean())
        score = 1.0 if not np.any(delta) else ssim(a, b)
        assert mean_abs == group["metrics"]["mean_abs"] and within8 == group["metrics"]["within8_frac"]
        assert abs(score - group["metrics"]["ssim"]) < 1e-12
        assert mean_abs <= limits["mean_abs_max"] and within8 >= limits["within8_min"]
        assert score >= limits["ssim_min"]
    return dict(
        status="PASS",
        scope="saved public render witness verification, not fresh acceptance",
        code_identity=identity,
        fresh_acceptance=False,
        registration="swm/PFRocketLanding-v0",
        device=report["device"],
        population=summarize_errors(errors, threshold=0.001),
        historical_sample=summarize_errors(errors, ddof=1, threshold=0.001),
        evidence_hashes={name: sha256(root / name) for name in ("result.json", "frames.npz")},
    )
