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
RECORDED_TASKS = ("g27", "g34", "g37", "g38", "g55", "g62", "g64")
RAY_TASKS = (
    "g02",
    "g03",
    "g05",
    "g06",
    "g07",
    "g28",
    "g30",
    "g32",
    "g33",
    "g35",
    "g39",
    "g40",
    "g41",
    "g42",
    "g44",
    "g56",
    "g58",
    "g59",
    "g60",
    "g63",
    "g65",
    "g66",
    "g68",
    "g69",
    "g70",
    "g71",
    "g72",
    "g73",
    "g74",
    "g79",
    "g80",
    "g81",
    "g82",
    "g83",
    "g84",
    "g85",
    "g86",
    "g87",
    "g88",
    "g89",
    "g93",
    "g94",
    "g95",
    "g96",
    "g97",
    "g99",
    "g100",
    "g103",
    "g104",
    "g105",
    "g106",
    "g107",
    "g108",
    "g111",
    "g113",
    "g114",
    "g115",
    "g116",
    "g117",
    "g118",
    "g119",
    "g120",
    "g121",
    "g123",
    "g124",
    "g125",
    "g126",
    "g127",
    "g129",
    "g130",
    "g131",
    "g133",
    "g134",
    "g135",
    "g136",
    "g137",
    "g138",
    "g139",
    "g140",
    "g141",
    "g142",
    "g143",
    "g144",
    "g145",
    "g147",
    "g148",
    "g149",
    "g150",
    "g04",
    "g29",
    "g31",
)


@lru_cache(maxsize=len(RAY_TASKS))
def load_ray_geometry(task):
    if task not in RAY_TASKS:
        raise ValueError("unknown versioned ray scene")
    manifest, arrays = load_geometry_directory(
        FIXTURES / f"semantics/object_permanence/{task}/opaque_pixel_rays_v1/nominal"
    )
    assert manifest["schema"] == "stable-envs/object-permanence-rigid-oracle-v2"
    assert manifest["visibility_contract"] == "stable-envs/opaque-pixel-rays-v1"
    return manifest, arrays


@lru_cache(maxsize=len(RECORDED_TASKS))
def load_geometry(task):
    if task not in RECORDED_TASKS:
        raise ValueError("unknown frozen rigid scene")
    directory = FIXTURES / (
        "semantics/occlusion_ball" if task == "g34" else f"semantics/object_permanence/{task}/nominal"
    )
    return load_geometry_directory(directory)


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
