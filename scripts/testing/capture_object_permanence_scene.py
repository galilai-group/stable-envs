"""Capture one native source realization before independent candidate/reference reads.

Procedural Solidify construction can vary corner normals between native builds.
The saved scene preserves one unmodified upstream result, including animation,
not candidate meshes or a replacement normal algorithm. Rebuilding produces a
new realization and requires new recordings/references; outputs never overwrite.
"""

import argparse
import hashlib
import json
import platform
import runpy
import subprocess
from pathlib import Path

SOURCE_COMMIT = "824baddcdad087810d35638bf8f273b884f74975"
CASES = {
    "G98": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G132": ("build_scene_by_kind", "animate_scene_by_kind"),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(root, output, task):
    import bpy

    if output.exists():
        raise FileExistsError("refusing to replace a native source realization")
    if task not in CASES or bpy.app.version != (4, 4, 0):
        raise ValueError("native realization requires an audited entry and Blender 4.4.0")
    if subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() != SOURCE_COMMIT:
        raise ValueError("source commit differs")
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"]):
        raise ValueError("source checkout must be clean")
    (taskfile,) = (root / "object_permanence/generator/tasks").glob(task + "_*/task.json")
    script = taskfile.parent / json.loads(taskfile.read_text())["member_scripts"][0]
    source = runpy.run_path(str(script), run_name="independent_native_scene_capture")
    source["clear_scene"]()
    build, animate = CASES[task]
    objects = source[build]()
    source[animate](objects)
    scene = bpy.context.scene
    scene.frame_set(scene.frame_start)
    bpy.context.view_layer.update()
    output.mkdir(parents=True)
    path = output / "source.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(path), compress=True, check_existing=False)
    metadata = {
        "schema": "stable-envs/native-source-realization-v1",
        "task": task,
        "member": 0,
        "source_commit": SOURCE_COMMIT,
        "source_url": "https://github.com/hokindeng/object-permanence",
        "source_script": str(script.relative_to(root)),
        "source_script_sha256": sha256(script),
        "source_task_manifest_sha256": sha256(taskfile),
        "source_preparation_recipe": {"build": build, "animate": animate},
        "scope": "one unmodified native build; no diversity, normal replacement or source rewriting",
        "file": path.name,
        "sha256": sha256(path),
        "frames": [scene.frame_start, scene.frame_end],
        "fps": float(scene.render.fps / scene.render.fps_base),
        "objects": sorted(o.name for o in scene.objects if o.type == "MESH"),
        "versions": {"blender": bpy.app.version_string, "python": platform.python_version()},
        "generator": "scripts/testing/capture_object_permanence_scene.py",
        "generator_sha256": sha256(Path(__file__)),
        "source_license": "CC-BY-NC-4.0",
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output / "SOURCE_LICENSE.txt").write_text((root / "LICENSE").read_text())
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", choices=tuple(CASES), required=True)
    a = p.parse_args()
    print(json.dumps(capture(a.source_root.resolve(), a.output.resolve(), a.task), indent=2))


if __name__ == "__main__":
    main()
