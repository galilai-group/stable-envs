"""Validate ordered source alpha layers, overlapping masks and unchanged budgets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from stable_envs.envs.object_permanence import RigidTrajectory
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
from stable_envs.testing.geometry import depth_mse, mask_iou
from tests.helpers.reference.object_permanence_recordings import DEPTH_MSE_MAX, MASK_IOU_MIN


def load(directory):
    directory = Path(directory)
    m = json.loads((directory / "manifest.json").read_text())
    path = directory / m["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == m["sha256"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    assert set(arrays) == set(m["arrays"])
    for field, parts in m["parts"].items():
        rows = []
        end = 0
        for part in parts:
            assert part["start"] == end and part["stop"] > end
            path = directory / part["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == part["sha256"]
            with np.load(path, allow_pickle=False) as archive:
                bits = archive["depth_xor"]
            assert list(bits.shape) == part["shape"] and bits.dtype == np.uint32
            base = arrays[field + "_base"]
            base = base.view(np.uint32) if field == "layer_depth" else base.astype(np.uint32)
            value = bits ^ base
            rows.append(value.view(np.float32) if field == "layer_depth" else value.astype(np.int16))
            end = part["stop"]
        assert end == len(arrays["source_frames"])
        arrays[field] = np.concatenate(rows)
    return m, arrays


def check(recordings, oracle, task, *, cache_static=True):
    m, source = load(oracle)
    worst = 0.0
    minimum = 1.0
    failures = []
    with RigidTrajectory(recordings, task.lower() + "-nominal") as clip:
        metadata = clip.manifest
        assert m["visibility_contract"] == metadata["visibility_contract"]
        assert m["material_convention"] == metadata["material_convention"]
        assert (
            m["source_script_sha256"] == metadata["source"]["script_sha256"]
            and m["source_commit"] == metadata["source"]["commit"]
        )
        assert m["object_ids"] == {o["name"]: o["id"] for o in metadata["objects"]}
        assert m["object_order"] == [o["name"] for o in metadata["objects"]]
        assert m["source_preparation_recipe"] == metadata["source"]["preparation_recipe"]
        for key in ("world", "projection"):
            np.testing.assert_array_equal(m["camera"][key], metadata["camera"][key])
        for key in ("near", "far", "projection_type"):
            assert m["camera"][key] == metadata["camera"][key]
        np.testing.assert_array_equal(m["alphas"], [o["rgba"][3] for o in metadata["objects"]])
        np.testing.assert_array_equal(clip.frames, source["source_frames"])
        with RigidSceneRenderer(
            clip,
            size=m["height"],
            visibility_contract=m["visibility_contract"],
            max_layers=m["max_layers"],
            cache_static=cache_static,
        ) as renderer:
            for start in range(0, len(clip), 4):
                stop = min(start + 4, len(clip))
                out = renderer.render_many(np.arange(start, stop), modes=("layer_depth", "layer_ids", "masks"))
                d, ids, masks = (out[k].cpu().numpy() for k in ("layer_depth", "layer_ids", "masks"))
                for row, frame in enumerate(source["source_frames"][start:stop]):
                    expected_i = source["layer_ids"][start + row]
                    expected_d = source["layer_depth"][start + row]
                    failed = {}
                    mse = 0.0
                    labels = expected_i.reshape(-1, m["max_layers"])
                    valid = labels > 0
                    pixels = np.broadcast_to(np.arange(m["height"] * m["width"])[:, None], labels.shape)
                    columns = np.full(max(m["object_ids"].values()) + 1, -1, np.int64)
                    for column, label in enumerate(m["object_ids"].values()):
                        columns[label] = column
                    expected_masks = np.zeros((len(metadata["objects"]), m["height"] * m["width"]), bool)
                    expected_masks[columns[labels[valid]], pixels[valid]] = True
                    expected_masks = expected_masks.reshape(len(metadata["objects"]), m["height"], m["width"])
                    for layer in range(m["max_layers"]):
                        roi = (expected_i[..., layer] > 0) | (ids[row, ..., layer] > 0)
                        if roi.any():
                            mse = max(mse, float(depth_mse(d[row, ..., layer], expected_d[..., layer], roi)))
                        present = np.union1d(np.unique(ids[row, ..., layer]), np.unique(expected_i[..., layer]))
                        for label in present[present > 0]:
                            value = float(mask_iou(ids[row, ..., layer] == label, expected_i[..., layer] == label))
                            minimum = min(minimum, value)
                            if value < MASK_IOU_MIN:
                                failed[f"layer{layer}/id{label}"] = value
                    for column, (name, label) in enumerate(m["object_ids"].items()):
                        value = float(mask_iou(masks[row, column], expected_masks[column]))
                        if not np.isnan(value):
                            minimum = min(minimum, value)
                            if value < MASK_IOU_MIN:
                                failed[name] = value
                    worst = max(worst, mse)
                    if mse > DEPTH_MSE_MAX or failed or not np.isfinite(mse):
                        failures.append({"frame": int(frame), "depth_mse": mse, "masks": failed})
    return {
        "task": task,
        "frames": len(source["source_frames"]),
        "visibility_contract": m["visibility_contract"],
        "cache_static": cache_static,
        "depth_mse_max": worst,
        "mask_iou_min": minimum,
        "thresholds": {"depth_mse_max": DEPTH_MSE_MAX, "mask_iou_min": MASK_IOU_MIN},
        "failures": failures,
        "passed": not failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--uncached", action="store_true")
    args = parser.parse_args()
    r = check(args.recordings, args.oracle, args.task, cache_static=not args.uncached)
    args.report.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=2))
    if not r["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
