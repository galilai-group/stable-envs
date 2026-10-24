"""Pinned source geometry and separate candidate recording samples."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
RECORDINGS = FIXTURES / "saved/object_permanence_rigid"
DEPTH_MSE_MAX = 1e-8
MASK_IOU_MIN = 0.99


@lru_cache(maxsize=2)
def load_geometry(task):
    directory = FIXTURES / ("semantics/occlusion_ball" if task == "g34" else "semantics/object_permanence/g62/nominal")
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / manifest["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    assert set(arrays) == set(manifest["arrays"])
    for name, array in arrays.items():
        assert list(array.shape) == manifest["arrays"][name]["shape"]
        assert str(array.dtype) == manifest["arrays"][name]["dtype"]
    if "depth_xor" in arrays:
        arrays["depth"] = (arrays["depth_xor"] ^ arrays["depth_base"].view(np.uint32)).view(np.float32)
        arrays["geometry_frames"] = arrays["source_frames"]
    for array in arrays.values():
        array.flags.writeable = False
    return manifest, arrays
