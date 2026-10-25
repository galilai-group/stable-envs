"""Check every recorded frame against an independent Blender geometry oracle.

Run in the CUDA interpreter, separately from source export/freeze. Reports include
all failures; a source-fidelity-deferred scene never becomes a successful import
merely because its rigid poses serialized correctly. No reference is modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stable_envs.envs.object_permanence import RigidTrajectory
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
from stable_envs.testing.geometry import depth_mse, mask_iou
from tests.helpers.reference.object_permanence_recordings import (
    DEPTH_MSE_MAX,
    MASK_IOU_MIN,
    load_geometry_directory,
)


def check(
    recordings,
    oracle,
    task,
    *,
    cache_static=True,
    geometry_precision="float32",
    camera_probe=None,
    visibility_contract=None,
):
    manifest, reference = load_geometry_directory(oracle)
    if manifest.get("visibility_contract") != visibility_contract:
        raise ValueError("oracle and renderer visibility contracts differ; select the version explicitly")
    if visibility_contract is not None and manifest["schema"] != "stable-envs/object-permanence-rigid-oracle-v2":
        raise ValueError("versioned visibility requires a v2 geometric oracle")
    if visibility_contract is not None and camera_probe is not None:
        raise ValueError("versioned visibility cannot use diagnostic camera overrides")
    if visibility_contract is not None:
        geometry_precision = "float64"
    failures, minimum, worst = [], 1.0, 0.0
    with RigidTrajectory(recordings, f"{task.lower()}-nominal") as clip:
        metadata = clip.manifest
        if metadata.get("scale_convention") != manifest.get("scale_convention"):
            raise ValueError("source and recording scale conventions differ")
        if metadata.get("geometry_conversion") != manifest.get("geometry_conversion"):
            raise ValueError("source and recording geometry conversion policies differ")
        if metadata.get("geometry_conversion") is not None:
            descriptions = {o["name"]: o["source_geometry"] for o in metadata["objects"] if "source_geometry" in o}
            if descriptions != manifest.get("source_geometry"):
                raise ValueError("source and recording converted geometry identities differ")
        if visibility_contract is not None and (
            not np.array_equal(manifest["camera"]["world"], metadata["camera"]["world"])
            or not np.array_equal(manifest["camera"]["projection"], metadata["camera"]["projection"])
            or any(manifest["camera"][k] != metadata["camera"][k] for k in ("near", "far"))
            or manifest["camera"].get("projection_type", "perspective")
            != metadata["camera"].get("projection_type", "perspective")
        ):
            raise ValueError("oracle and recording cameras differ")
        if (
            metadata["source"]["commit"] != manifest["source_commit"]
            or metadata["source"]["script_sha256"] != manifest["source_script_sha256"]
        ):
            raise ValueError("source and recording definitions differ")
        if metadata["source"].get("execution_context") != manifest.get("source_execution_context"):
            raise ValueError("source and recording execution contexts differ")
        if metadata["source"].get("realization") != manifest.get("source_realization"):
            raise ValueError("source and recording native realizations differ")
        if (
            visibility_contract == "stable-envs/opaque-pixel-rays-v2"
            or metadata.get("geometry_conversion")
            or clip.has_scales
        ) and (
            not manifest.get("source_preparation_recipe")
            or metadata["source"].get("preparation_recipe") != manifest["source_preparation_recipe"]
        ):
            raise ValueError("source and recording preparation recipes differ")
        if manifest["object_ids"] != {o["name"]: o["id"] for o in metadata["objects"]}:
            raise ValueError("source and recording object identities differ")
        indices = np.searchsorted(clip.frames, reference["source_frames"])
        if (indices >= len(clip)).any() or not np.array_equal(clip.frames[indices], reference["source_frames"]):
            raise ValueError("source and recording frame numbers differ")
        if clip.has_scales:
            if manifest.get("source_scale_space") != "absolute-world-matrix-decomposition":
                raise ValueError("source scales require independently sampled world decomposition")
            absolute = reference.get("source_scales")
            if absolute is None or absolute.shape != (len(indices), len(metadata["objects"]), 3):
                raise ValueError("source scales do not match frames and objects")
            if not np.isfinite(absolute).all() or (absolute <= 0).any():
                raise ValueError("source scales must be finite and positive")
            np.testing.assert_allclose(clip.object_scales(indices), absolute / absolute[:1], atol=2e-6, rtol=0)
        with RigidSceneRenderer(
            clip,
            size=manifest["height"],
            cache_static=cache_static,
            geometry_precision=geometry_precision,
            visibility_contract=visibility_contract,
        ) as renderer:
            if camera_probe is not None:
                import torch

                if geometry_precision != "float64":
                    raise ValueError("source camera calibration requires float64 geometry")
                if (
                    camera_probe["source_commit"] != metadata["source"]["commit"]
                    or camera_probe["source_script"] != metadata["source"]["script"]
                    or camera_probe["source_script_sha256"] != metadata["source"]["script_sha256"]
                    or camera_probe["task"].lower() != task.lower()
                    or camera_probe["size"] != [manifest["height"], manifest["width"]]
                    or not camera_probe["queries"]
                ):
                    raise ValueError("camera probe does not match source definition/image")
                calibration = camera_probe["queries"][0]
                for query in camera_probe["queries"]:
                    if (
                        not np.array_equal(query["camera_world"], metadata["camera"]["world"])
                        or not np.array_equal(query["recorded_projection"], metadata["camera"]["projection"])
                        or not np.array_equal(query["oracle_ray_projection"], calibration["oracle_ray_projection"])
                    ):
                        raise ValueError("camera probe requires a matching fixed recorded camera")
                projection = np.asarray(calibration["oracle_ray_projection"], np.float64)
                if projection.shape != (4, 4) or not np.isfinite(projection).all():
                    raise ValueError("camera calibration requires a finite [4,4] matrix")
                renderer.camera.projection = torch.tensor(projection, dtype=torch.float64, device=renderer.device)
            output = renderer.render_many(indices, modes=("depth", "ids"))
            depth, ids = output["depth"].cpu().numpy(), output["ids"].cpu().numpy()
    if depth.shape != reference["depth"].shape or ids.shape != reference["object_ids"].shape:
        raise ValueError("source and rendering dimensions differ")
    for row, frame in enumerate(reference["source_frames"]):
        expected = reference["object_ids"][row]
        roi = (expected > 0) | (ids[row] > 0)
        mse = float(depth_mse(depth[row], reference["depth"][row], roi))
        worst = max(worst, mse)
        masks = {}
        for name, label in manifest["object_ids"].items():
            iou = float(mask_iou(ids[row] == label, expected == label))
            if not np.isnan(iou):
                minimum = min(minimum, iou)
                if iou < MASK_IOU_MIN:
                    masks[name] = iou
        if mse > DEPTH_MSE_MAX or masks or not np.isfinite(mse):
            failures.append({"frame": int(frame), "depth_mse": mse, "failed_masks": masks})
    return {
        "task": task.upper(),
        "visibility_contract": visibility_contract,
        "source_commit": manifest["source_commit"],
        "cache_static": cache_static,
        "geometry_precision": geometry_precision,
        "camera_convention": "recorded_projection" if camera_probe is None else "source_probe_view_frame",
        "certification_eligible": camera_probe is None,
        "frames": len(reference["source_frames"]),
        "depth_mse_max": worst,
        "mask_iou_min": minimum,
        "thresholds": {"depth_mse_max": DEPTH_MSE_MAX, "mask_iou_min": MASK_IOU_MIN},
        "within_thresholds": not failures,
        "passed": not failures and camera_probe is None,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--uncached", action="store_true")
    parser.add_argument("--geometry-precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--camera-probe", type=Path, help="diagnostic-only source camera calibration report")
    parser.add_argument("--visibility-contract", help="explicit version shared by renderer and new oracle")
    args = parser.parse_args()
    result = check(
        args.recordings,
        args.oracle,
        args.task,
        cache_static=not args.uncached,
        geometry_precision=args.geometry_precision,
        camera_probe=None if args.camera_probe is None else json.loads(args.camera_probe.read_text()),
        visibility_contract=args.visibility_contract,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
