"""Render source-only GIFs and compare them with opt-in optical replay GIFs."""

import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


def durations(count, fps):
    """Preserve total clip time within GIF centisecond resolution."""
    times = [round(100 * i / fps) * 10 for i in range(count + 1)]
    return [b - a for a, b in zip(times, times[1:])]


def source_preview(root, output, *, task="G01", member=0, width=320, height=180, preview_samples=None):
    """Use source shaders; optional preview-only AA does not affect the oracle."""
    import subprocess

    import bpy

    from .freeze_object_permanence import SOURCE_COMMIT
    from .freeze_object_permanence_optical import prepare

    if subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip() != SOURCE_COMMIT or subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True
    ):
        raise ValueError("source preview requires pinned source")
    (path,) = (root / "object_permanence/generator/tasks").glob(task + "_*/task.json")
    script = path.parent / json.loads(path.read_text())["member_scripts"][member]
    prepare(script, task)
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    if preview_samples is not None:
        if type(preview_samples) is not int or preview_samples < 1:
            raise ValueError("preview sample count must be positive")
        scene.eevee.taa_render_samples = preview_samples
        print("Source visual preview only: EEVEE render samples", preview_samples, flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    with tempfile.TemporaryDirectory(prefix=".optical-source-gif-", dir=output.parent) as tmp:
        path = Path(tmp) / "frame.png"
        scene.render.filepath = str(path)
        for frame in range(scene.frame_start, scene.frame_end + 1):
            scene.frame_set(frame)
            bpy.ops.render.render(write_still=True)
            with Image.open(path) as im:
                frames.append(im.convert("RGB").copy())
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations(len(frames), scene.render.fps / scene.render.fps_base),
        loop=0,
    )


def comparison(recordings, source, output, *, clip="g01-nominal", size=180, source_header=0):
    from stable_envs.envs.object_permanence import RigidTrajectory
    from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer

    frames = []
    with RigidTrajectory(recordings, clip) as trajectory:
        fps = trajectory.manifest["fps"]
        with RigidSceneRenderer(
            trajectory, size=size, visibility_contract=trajectory.manifest["visibility_contract"]
        ) as renderer:
            with Image.open(source) as original:
                if original.n_frames != len(trajectory):
                    raise ValueError("source preview must cover every recorded frame")
                for index in range(len(trajectory)):
                    original.seek(index)
                    left = original.convert("RGB")
                    left = left.crop((0, source_header, left.width, left.height)).resize((renderer.width, size))
                    right = Image.fromarray(renderer.render([index])[0].cpu().numpy())
                    canvas = Image.new("RGB", (renderer.width * 2, size + 38), "white")
                    canvas.paste(left, (0, 38))
                    canvas.paste(right, (renderer.width, 38))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((4, 3), f"{clip} | source frame {int(trajectory.frames[index])}", fill="black")
                    draw.text((4, 20), "Source EEVEE (source shaders)", fill="black")
                    draw.text((renderer.width + 4, 20), "Sampled ideal surface optics v1", fill="black")
                    frames.append(canvas)
                    if index % 20 == 0:
                        print(clip, "comparison GIF frame", index + 1, flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations(len(frames), fps), loop=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("source")
    a.add_argument("--source-root", type=Path, required=True)
    a.add_argument("--task", choices=("G01", "G03"), default="G01")
    a.add_argument("--member", type=int, default=0)
    a.add_argument("--preview-samples", type=int, default=None)
    a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("compare")
    a.add_argument("--recordings", type=Path, required=True)
    a.add_argument("--source-gif", type=Path, required=True)
    a.add_argument("--clip", default="g01-nominal")
    a.add_argument("--source-header", type=int, default=0)
    a.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "source":
        source_preview(
            args.source_root, args.output, task=args.task, member=args.member, preview_samples=args.preview_samples
        )
    else:
        comparison(args.recordings, args.source_gif, args.output, clip=args.clip, source_header=args.source_header)


if __name__ == "__main__":
    main()
