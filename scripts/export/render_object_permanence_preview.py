"""Render native Blender nominal RGB previews, independently of candidate pixels.

Cycles CPU is explicit for headless portability. Resolution/sample count are
preview settings, not a reproduction of the dataset's original video quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .object_permanence import AUDITED_TASKS, SOURCE_COMMIT, prepare, sha256, source_execution_context


def render(root, output, task, *, samples=16, frame_stop=None):
    import bpy

    if output.exists():
        raise FileExistsError(f"refusing to overwrite preview: {output}")
    scene, script, _ = prepare(root, task)
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = 320, 180
    scene.render.resolution_percentage = 100
    scene.render.threads_mode, scene.render.threads = "FIXED", 4
    scene.render.use_persistent_data = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    stop = scene.frame_end if frame_stop is None else min(frame_stop, scene.frame_end)
    output.mkdir(parents=True)
    frames = list(range(scene.frame_start, stop + 1))
    for frame in frames:
        scene.frame_set(frame)
        scene.render.filepath = str(output / f"{frame:04d}.png")
        bpy.ops.render.render(write_still=True)
        if frame % 10 == 0 or frame == frames[0]:
            print(f"{task}: native Blender preview {frame}/{stop}", flush=True)
    metadata = {
        "source_commit": SOURCE_COMMIT,
        "source_script": str(script.relative_to(root)),
        "source_script_sha256": sha256(script),
        "task": task,
        "profile": "nominal-no-diversity",
        "blender": bpy.app.version_string,
        "engine": "Cycles CPU",
        "samples": samples,
        "denoising": True,
        "size": [180, 320],
        "fps": float(scene.render.fps / scene.render.fps_base),
        "frames": frames,
        "note": "native source scene, preview rendering quality; not original MP4 pixels",
    }
    context = source_execution_context(root, task, script)
    if context is not None:
        metadata["source_execution_context"] = context
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(AUDITED_TASKS), required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--frame-stop", type=int)
    args = parser.parse_args()
    if args.samples < 1 or (args.frame_stop is not None and args.frame_stop < 1):
        parser.error("samples and frame-stop must be positive")
    render(
        args.source_root.resolve(), args.output.resolve(), args.task, samples=args.samples, frame_stop=args.frame_stop
    )


if __name__ == "__main__":
    main()
