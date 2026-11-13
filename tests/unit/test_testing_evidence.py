"""Negative controls for shared measurements and fail-closed evidence accounting."""

import json

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from stable_envs.testing.evidence import EvidenceRun, verify_seal
from stable_envs.testing.metrics import leaf_difference, normalized_mse, summarize_errors
from tests.helpers.reference.cases import CaseKey, coverage
from tests.helpers.reference.semantics import compare_state


def test_unpinned_new_source_profile_is_rejected_before_loading_source():
    from stable_envs.testing.provenance import pinned_source_identity

    for pins in ({}, {"gymnasium": None}, {"gymnasium": "present"}):
        with pytest.raises(ValueError, match="explicit package pins"):
            pinned_source_identity("/does/not/exist", "abc", pins)


def test_legacy_render_roundtrip_name_is_only_a_compatibility_alias():
    from stable_envs.testing import conformance

    class RenderRows:
        B = 2

        def sample_states(self, batch):
            return np.arange(batch * 3, dtype=np.float32).reshape(batch, 3)

        def _set_render_state(self, rows):
            self.rows = torch.from_numpy(rows.copy())

        def _get_render_state(self):
            return self.rows

        def get_state(self):
            raise AssertionError("This check must not claim the public simulator State")

    env = RenderRows()
    assert (
        conformance.check_state_roundtrip(env) == conformance.check_render_state_roundtrip(env) == {"max_abs_err": 0.0}
    )


def test_rgb_integer_overflow_and_noncontiguous_batches():
    black = np.zeros((2, 3, 4, 5, 3), np.uint8)
    white = np.full_like(black, 255)
    np.testing.assert_array_equal(normalized_mse(black, white), np.ones((2, 3)))
    white[0, 0] = 0
    np.testing.assert_array_equal(normalized_mse(black[:, ::-1], white[:, ::-1]), [[1, 1, 0], [1, 1, 1]])
    assert normalized_mse(black[0, 0], white[0, 0]).shape == ()


@pytest.mark.parametrize("kind", ["dtype", "shape", "channels", "empty_image"])
def test_invalid_images_fail_without_conversion(kind):
    a = np.zeros((2, 3, 4, 3), np.uint8)
    b = (
        a.astype(float)
        if kind == "dtype"
        else a[:1]
        if kind == "shape"
        else a[..., :1]
        if kind == "channels"
        else a[:, :0]
    )
    with pytest.raises((TypeError, ValueError)):
        normalized_mse(a, b)


def test_empty_selection_is_not_perfect_parity_and_bad_frame_is_not_hidden():
    assert normalized_mse(np.zeros((0, 4, 4, 3), np.uint8), np.zeros((0, 4, 4, 3), np.uint8)).size == 0
    with pytest.raises(ValueError):
        summarize_errors([])
    summary = summarize_errors([0] * 100 + [0.002], threshold=0.001)
    assert summary["mean"] < 0.001 and summary["over_threshold"] == 1 and summary["worst_index"] == 100
    assert summary["ddof"] == 0
    assert summarize_errors([0, 1], ddof=1)["sd"] == pytest.approx(2**-0.5)


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [-1]])
def test_nonfinite_or_negative_errors_are_not_a_pass(values):
    with pytest.raises(ValueError):
        summarize_errors(values)


def test_rng_integer_precision_and_float_tolerances_do_not_mix():
    a = torch.tensor([2**62], dtype=torch.int64)
    b = a + 1
    assert leaf_difference(a, b)["max_abs_error"] == 1
    with pytest.raises(AssertionError, match="rng"):
        compare_state(TensorDict({"rng": a}, [1]), TensorDict({"rng": b}, [1]), atol=10)
    extremes = leaf_difference(np.array([-(2**63)], np.int64), np.array([2**63 - 1], np.int64))
    assert extremes["max_abs_error"] == 2**64 - 1
    assert leaf_difference(np.array([np.inf]), np.array([-np.inf]))["max_abs_error"] == float("inf")
    assert leaf_difference(np.array([np.nan]), np.array([np.nan]))["exact"]


def test_missing_shape_dtype_and_empty_leaf():
    for other in (np.zeros((2, 1)), np.zeros(2, np.int64)):
        assert not leaf_difference(np.zeros(2), other)["comparable"]
    assert leaf_difference(np.zeros(0), np.zeros(0))["exact"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "sibling", "skipped", "blocked", "na", "failed"])
def test_required_coverage_never_borrows_a_sibling_or_skip(defect):
    a = CaseKey("swm/FetchReach-v3", "state", "restore")
    b = CaseKey("swm/FetchReachDict-v3", "state", "restore")
    rows = [(a, "PASS"), (b, "PASS")]
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(rows[0])
    elif defect == "sibling":
        rows[1] = rows[0]
    else:
        rows[1] = (b, {"skipped": "SKIPPED", "blocked": "BLOCKED", "na": "NOT_APPLICABLE", "failed": "FAIL"}[defect])
    assert coverage([a, b], rows)["status"] != "PASS"
    assert coverage([a, b], [(a, "PASS"), (b, "PASS")])["status"] == "PASS"


def test_evidence_write_once_roundtrip_and_no_pickle(tmp_path):
    run = EvidenceRun(tmp_path / "run")
    run.json("case/result.json", {"status": "PASS"})
    run.arrays("states.npz", rng=np.array([2**62 + 1], np.int64))
    with pytest.raises(FileExistsError):
        run.json("case/result.json", {})
    with pytest.raises(TypeError):
        run.arrays("objects.npz", bad=np.array([object()]))
    run.seal()
    assert len(verify_seal(run.root)) == 2
    with pytest.raises(ValueError):
        run.json("new.json", {})
    with pytest.raises(FileExistsError):
        EvidenceRun(run.root)
    with np.load(run.root / "states.npz", allow_pickle=False) as saved:
        assert saved["rng"][0] == 2**62 + 1


def test_failure_images_preserve_exact_pixels_and_batch_index(tmp_path):
    source = np.zeros((2, 3, 4, 5, 3), np.uint8)
    ours = source.copy()
    ours[1, 2, 3, 4] = 255
    run = EvidenceRun(tmp_path / "failure")
    run.frame_diagnostic("worst", ours, source, metadata={"case": "synthetic-negative-control"})
    run.seal()
    report = json.loads((run.root / "worst.json").read_text())
    assert report["frame_index"] == [1, 2]
    with np.load(run.root / "worst.npz", allow_pickle=False) as frames:
        np.testing.assert_array_equal(frames["ours"], ours[1, 2])
        np.testing.assert_array_equal(frames["source"], source[1, 2])
        assert frames["difference"].dtype == np.int16 and frames["difference"].max() == 255


@pytest.mark.parametrize("defect", ["missing", "changed", "extra", "duplicate", "escape", "symlink"])
def test_evidence_corruption_is_rejected(tmp_path, defect):
    run = EvidenceRun(tmp_path / "run")
    run.json("result.json", {"status": "PASS"})
    run.seal()
    path, seal = run.root / "result.json", run.root / "SHA256SUMS"
    if defect == "missing":
        path.unlink()
    elif defect == "changed":
        path.write_text(json.dumps({"status": "FAIL"}))
    elif defect == "extra":
        (run.root / "not-sealed").write_text("partial")
    elif defect == "duplicate":
        seal.write_text(seal.read_text() * 2)
    elif defect == "escape":
        seal.write_text(seal.read_text().replace("result.json", "../result.json"))
    else:
        path.rename(tmp_path / "outside.json")
        path.symlink_to(tmp_path / "outside.json")
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_seal(run.root)
