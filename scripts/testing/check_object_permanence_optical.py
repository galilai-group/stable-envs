"""Check independent optical masks/depth with the existing geometry budgets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from stable_envs.envs.object_permanence import RigidTrajectory
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
from stable_envs.testing.geometry import mask_iou
from tests.helpers.reference.object_permanence_recordings import DEPTH_MSE_MAX, MASK_IOU_MIN


def load(directory):
    directory = Path(directory)
    m = json.loads((directory / "manifest.json").read_text())
    path = directory / m["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == m["sha256"]
    with np.load(path, allow_pickle=False) as a:
        arrays = dict(a)
    for field, parts in m["parts"].items():
        rows = []
        end = 0
        for part in parts:
            assert part["start"] == end
            path = directory / part["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == part["sha256"]
            with np.load(path, allow_pickle=False) as a:
                bits = a["depth_xor"]
            assert list(bits.shape) == part["shape"] and bits.dtype == np.uint32
            base = arrays[field + "_base"]
            value = bits ^ (base.view(np.uint32) if base.dtype == np.float32 else base.astype(np.uint32))
            rows.append(value.view(np.float32) if base.dtype == np.float32 else value.astype(base.dtype))
            end = part["stop"]
        assert end == m["frames"]
        arrays[field] = np.concatenate(rows)
    arrays["masks"] = np.unpackbits(arrays["masks_packed"], axis=-1, count=m["width"]).astype(bool)
    return m, arrays


def check(recordings, oracle, *, cache_static=True):
    m, source = load(oracle)
    failures = []
    worst = 0.0
    minimum = 1.0
    with RigidTrajectory(recordings, m["clip"]) as clip:
        metadata = clip.manifest
        assert m["visibility_contract"] == metadata["visibility_contract"]
        assert m["material_convention"] == metadata["material_convention"]
        assert m["optical_policy"] == metadata["optical_policy"]
        assert m["source_render_engine"] == metadata["source"]["render_engine"]
        assert m["source_use_raytracing"] == metadata["source"]["use_raytracing"]
        assert m["source_commit"] == metadata["source"]["commit"]
        assert m["source_script_sha256"] == metadata["source"]["script_sha256"]
        assert m["source_task_manifest_sha256"] == metadata["source"]["task_manifest_sha256"]
        assert m["source_preparation_recipe"] == metadata["source"]["preparation_recipe"]
        assert m["object_ids"] == {o["name"]: o["id"] for o in metadata["objects"]}
        assert m["object_order"] == [o["name"] for o in metadata["objects"]]
        for obj in metadata["objects"]:
            mat = m["materials"][obj["name"]]
            assert mat["alpha"] == obj["rgba"][3]
            assert {k: v for k, v in mat.items() if k != "alpha"} == obj["optics"]
        for key in ("world", "projection", "near", "far", "projection_type"):
            np.testing.assert_array_equal(m["camera"][key], metadata["camera"][key])
        np.testing.assert_array_equal(source["source_frames"], clip.frames)
        with RigidSceneRenderer(
            clip, size=m["height"], cache_static=cache_static, visibility_contract=m["visibility_contract"]
        ) as renderer:
            for start in range(0, len(clip), 4):
                out = renderer.render_many(
                    np.arange(start, min(start + 4, len(clip))), modes=("depth", "ids", "masks", "path_depth")
                )
                out = {k: v.cpu().numpy() for k, v in out.items()}
                for row in range(len(out["depth"])):
                    index = start + row
                    bad = {}
                    mse = 0.0
                    roi = (source["ids"][index] > 0) | (out["ids"][row] > 0)
                    if roi.any():
                        mse = float(
                            np.mean((out["depth"][row][roi].astype(np.float64) - source["depth"][index][roi]) ** 2)
                        )
                    for column, name in enumerate(m["object_order"]):
                        label = m["object_ids"][name]
                        first_iou = float(mask_iou(out["ids"][row] == label, source["ids"][index] == label))
                        if np.isfinite(first_iou):
                            minimum = min(minimum, first_iou)
                            if first_iou < MASK_IOU_MIN:
                                bad["first/" + name] = first_iou
                        a, b = out["masks"][row, column], source["masks"][index, column]
                        iou = float(mask_iou(a, b))
                        if np.isfinite(iou):
                            minimum = min(minimum, iou)
                            if iou < MASK_IOU_MIN:
                                bad[name] = iou
                        # Union ROI preserves the old rule: a missed object is
                        # penalized in depth too, not hidden by an intersection ROI.
                        roi = a | b
                        if roi.any():
                            value = float(
                                np.mean(
                                    (
                                        out["path_depth"][row, column][roi].astype(np.float64)
                                        - source["path_depth"][index, column][roi]
                                    )
                                    ** 2
                                )
                            )
                            mse = max(mse, value)
                    worst = max(worst, mse)
                    if mse > DEPTH_MSE_MAX or bad or not np.isfinite(mse):
                        failures.append({"frame": int(source["source_frames"][index]), "depth_mse": mse, "masks": bad})
    return {
        "clip": m["clip"],
        "frames": m["frames"],
        "cache_static": cache_static,
        "depth_mse_max": worst,
        "mask_iou_min": minimum,
        "thresholds": {"depth_mse_max": DEPTH_MSE_MAX, "mask_iou_min": MASK_IOU_MIN},
        "passed": not failures,
        "failures": failures,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recordings", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--uncached", action="store_true")
    a = p.parse_args()
    result = check(a.recordings, a.oracle, cache_static=not a.uncached)
    a.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
