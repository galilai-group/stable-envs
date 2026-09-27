"""Report actual package profiles and prove software OpenGL on an allocated node."""

import importlib.metadata as md
import json
import os
import platform
import sys
import traceback
from pathlib import Path

record = dict(
    executable=sys.executable,
    host=platform.node(),
    environment={
        k: os.environ.get(k)
        for k in ["MUJOCO_GL", "DISPLAY", "LIBGL_ALWAYS_SOFTWARE", "GALLIUM_DRIVER", "CUDA_VISIBLE_DEVICES"]
    },
    versions={},
)
for name in [
    "torch",
    "numpy",
    "tensordict",
    "mujoco",
    "dm-control",
    "gymnasium",
    "gymnasium-robotics",
    "ogbench",
    "craftax",
    "jax",
    "jaxlib",
    "flax",
    "pymunk",
    "PyFlyt",
    "pybullet",
    "nvdiffrast",
    "glfw",
    "PyOpenGL",
]:
    try:
        d = md.distribution(name)
        record["versions"][name] = dict(version=d.version, path=str(d.locate_file("")))
    except md.PackageNotFoundError:
        pass
if "--gl" in sys.argv:
    try:
        import mujoco
        from OpenGL import GL

        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=64, width=64) as renderer:
            renderer.update_scene(data)
            frame = renderer.render()
            renderer._gl_context.make_current()
            record["gl_renderer"] = GL.glGetString(GL.GL_RENDERER).decode()
            record["gl_vendor"] = GL.glGetString(GL.GL_VENDOR).decode()
            record["frame_shape"] = list(frame.shape)
            assert any(k in record["gl_renderer"].lower() for k in ["llvmpipe", "softpipe", "swrast"])
            record["software_cpu_gl_proven"] = True
    except Exception:
        record["gl_error"] = traceback.format_exc()
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
