"""Freeze all G62 frames directly from pinned Blender, without candidate imports.

The depth encoding XORs IEEE float32 bits against frame one, losslessly removing
repeated background data. It does not change depth precision or acceptance gates.
Existing output directories are never overwritten, including the G34 fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import runpy
import subprocess
from pathlib import Path

import numpy as np

from .freeze_object_permanence import capture_bvh

SOURCE_COMMIT = "824baddcdad087810d35638bf8f273b884f74975"
SOURCE_SCRIPT = (
    "object_permanence/generator/tasks/G62_two_balls_cross_tunnel/"
    "pb_task_00000108_pb_two_balls_cross_tunnel_0109_blender.py"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(root, output, width=320, height=180):
    import bpy

    if bpy.app.version[:2] != (4, 4):
        raise ValueError("G62 oracle requires Blender 4.4.x")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT or subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ):
        raise ValueError("reference source must be the clean pinned commit")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite oracle: {output}")
    script = root / SOURCE_SCRIPT
    namespace = runpy.run_path(str(script), run_name="g62_independent_reference")
    namespace["clear_scene"]()
    scene, metadata = namespace["build_scene_by_kind"]()
    namespace["animate_scene_by_kind"](scene, metadata)
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    arrays, names, labels = capture_bvh(scene, width, height, ("cross_ball_red", "cross_ball_blue"), "G62")
    output.mkdir(parents=True)
    archive = output / "blender_g62_nominal.npz"
    np.savez_compressed(archive, **arrays)
    manifest = {
        "schema": "stable-envs/object-permanence-g62-oracle-v1",
        "source_url": "https://github.com/hokindeng/object-permanence",
        "source_commit": commit,
        "source_script": SOURCE_SCRIPT,
        "source_script_sha256": sha256(script),
        "source_profile": "nominal first member; no diversity wrapper",
        "source_license": "CC-BY-NC-4.0",
        "generator": "scripts/testing/freeze_object_permanence_g62.py",
        "generator_sha256": sha256(Path(__file__)),
        "versions": {"blender": bpy.app.version_string, "numpy": np.__version__, "python": platform.python_version()},
        "file": archive.name,
        "sha256": sha256(archive),
        "width": width,
        "height": height,
        "fps": float(scene.render.fps / scene.render.fps_base),
        "object_ids": labels,
        "object_order": names,
        "targets": ["cross_ball_red", "cross_ball_blue"],
        "depth_encoding": "float32 bitwise XOR against depth_base; lossless",
        "depth_convention": "positive camera-axis distance in source scene units; zero for no hit",
        "mask_convention": "visible source mesh at the pixel center; zero for no hit",
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "SOURCE_LICENSE.txt").write_text((root / "LICENSE").read_text())
    (output / "README.md").write_text(
        "# Blender G62 nominal geometry oracle\n\n"
        "All 120 source frames at 320x180, generated independently from pinned Blender 4.4.\n"
        "No candidate exporter, recording or renderer is imported. Source, camera dimensions,\n"
        "IDs, units, array shapes, hashes and versions are in manifest.json.\n"
        "Depth XOR encoding is bit-exact; shader RGB is not an oracle.\n\n"
        "Regenerate in a new directory with `python -m scripts.testing.freeze_object_permanence_g62\n"
        "--source-root <pinned-checkout> --output <new-directory>` (one command).\n"
        "Existing fixtures are never overwritten. CC-BY-NC-4.0 source-derived data;\n"
        "see SOURCE_LICENSE.txt for full attribution and license terms.\n"
    )
    print(f"frozen {archive}: {archive.stat().st_size} bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.source_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
