import hashlib

import numpy as np
import pytest

from scripts.benchmarks.renderer_table.export_parity_campaign import compare, stats


def role(tmp_path, name, pixels, state="same"):
    path = tmp_path / (name + ".npy")
    image = np.full((2, 2, 3), pixels, dtype=np.uint8)
    np.save(path, image)
    return dict(
        status="COMPLETE",
        frames={
            0: dict(
                index=0,
                case=0,
                seed=17,
                step=0,
                state_hash=state,
                image_path=str(path),
                image_hash=hashlib.sha256(image.tobytes()).hexdigest(),
            )
        },
    )


def test_normalized_error_and_no_false_match(tmp_path):
    left, right = role(tmp_path, "ours", 255), role(tmp_path, "source", 0)
    result = compare(left, right)
    assert result["statistics"]["mean"] == 1.0
    assert result["all_frames_le_001"] is False
    right["frames"][0]["state_hash"] = "different"
    assert compare(left, right)["status"] == "NONCOMPARABLE_STATE"
    right["status"] = "ERROR"
    assert compare(left, right)["status"] == "UNAVAILABLE"


def test_corrupted_evidence_is_rejected(tmp_path):
    left, right = role(tmp_path, "ours", 5), role(tmp_path, "source", 5)
    np.save(left["frames"][0]["image_path"], np.zeros((2, 2, 3), np.uint8))
    with pytest.raises(AssertionError):
        compare(left, right)


def test_sample_sd_and_empty_are_explicit():
    assert stats([1.0, 3.0])["sd"] == pytest.approx(2.0**0.5)
    assert stats([]) == dict(n=0, mean=None, sd=None, max=None)
