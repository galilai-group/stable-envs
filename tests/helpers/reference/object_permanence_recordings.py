"""Pinned source geometry and separate candidate recording samples."""

import hashlib
import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
RECORDINGS = FIXTURES / "saved/object_permanence_rigid"
DEPTH_MSE_MAX = 1e-8
MASK_IOU_MIN = 0.99


def load_geometry_directory(directory):
    """Read a source oracle, including uncertified diagnostic captures."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / manifest["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    assert set(arrays) == set(manifest["arrays"])
    for name, array in arrays.items():
        assert list(array.shape) == manifest["arrays"][name]["shape"]
        assert str(array.dtype) == manifest["arrays"][name]["dtype"]
    if "depth_parts" in manifest:
        parts, end = [], 0
        for part in manifest["depth_parts"]:
            assert part["start"] == end and part["stop"] > end
            path = directory / part["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == part["sha256"]
            with np.load(path, allow_pickle=False) as archive:
                assert archive.files == ["depth_xor"]
                bits = archive["depth_xor"]
            assert list(bits.shape) == part["shape"] and str(bits.dtype) == part["dtype"] == "uint32"
            assert bits.shape == (part["stop"] - end, manifest["height"], manifest["width"])
            end = part["stop"]
            parts.append(bits)
        assert end == len(arrays["source_frames"])
        arrays["depth_xor"] = np.concatenate(parts)
    if "depth_xor" in arrays:
        arrays["depth"] = (arrays["depth_xor"] ^ arrays["depth_base"].view(np.uint32)).view(np.float32)
        arrays["geometry_frames"] = arrays["source_frames"]
    for array in arrays.values():
        array.flags.writeable = False
    return manifest, arrays
