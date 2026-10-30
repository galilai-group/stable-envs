"""Export audited optical members without changing existing nominal exports."""

import argparse
import platform
from pathlib import Path

from .object_permanence import SOURCE_COMMIT, sample_scene, sha256, source_recipe, write_recording

MEMBERS = {
    ("G01", 0): ("$prefix:render_png", ""),
    ("G01", 1): ("$prefix:render_png", ""),
    ("G03", 1): ("build_scene", "animate_scene"),
}


def prepare(root, task, member):
    import bpy

    if bpy.app.version[:2] != (4, 4) or type(member) is not int or (task, member) not in MEMBERS:
        raise ValueError("optical export requires Blender 4.4 and an audited member")
    from .object_permanence import prepare as prepare_source

    return prepare_source(root, task, member=member)


def export(root, output, *, task="G01", member=0):
    import bpy

    clip = task.lower() + ("-nominal" if member == 0 else f"-member{member}")
    if (output / "clips" / clip).exists():
        raise FileExistsError("refusing to overwrite optical recording")
    scene, script, _ = prepare(root, task, member)
    _, options = source_recipe(task, member)
    recording = sample_scene(scene, **options)
    source = {
        "url": "https://github.com/hokindeng/object-permanence",
        "commit": SOURCE_COMMIT,
        "script": str(script.relative_to(root)),
        "script_sha256": sha256(script),
        "task_manifest_sha256": sha256(script.parent / "task.json"),
        "task": task,
        "member": member,
        "profile": "audited-optical-member-no-diversity",
        "parameters": {},
        "seed": None,
        "license": "CC-BY-NC-4.0",
        "license_text": (root / "LICENSE").read_text(),
        "preparation_recipe": dict(zip(("build", "animate"), MEMBERS[task, member], strict=True)),
        "render_engine": scene.render.engine,
        "use_raytracing": bool(scene.eevee.use_raytracing),
        "versions": {
            "blender": bpy.app.version_string,
            "numpy": __import__("numpy").__version__,
            "python": platform.python_version(),
        },
        "exporter": "scripts/export/object_permanence_optical.py",
        "exporter_sha256": sha256(Path(__file__)),
        "geometry_exporter_sha256": sha256(Path(__file__).with_name("object_permanence.py")),
        "optical_policy_sha256": sha256(
            Path(__file__).parents[2] / "stable_envs/envs/object_permanence/rendering/optics.py"
        ),
    }
    return write_recording(output, clip, recording, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--member", type=int, choices=(0, 1), default=0)
    parser.add_argument("--task", choices=("G01", "G03"), default="G01")
    a = parser.parse_args()
    print(export(a.source_root, a.output, task=a.task, member=a.member))


if __name__ == "__main__":
    main()
