"""Native reference rendering, static-plate baking and accelerated dispatch."""

from __future__ import annotations

from functools import cache
from threading import RLock

import mujoco
import numpy as np
import torch

from ...._core.mujoco_gl import mujoco_renderer
from ....render._resources import retain_process_lifetime
from ....render.raycast3d import Camera
from ..scenes.g34 import OBJECT_NAMES, make_model, set_pose


@cache
def _bundle(height, width):
    model = make_model()
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    renderer = mujoco_renderer(model, height=height, width=width)
    retain_process_lifetime(renderer, "occlusion-ball native renderer per resolution")
    return model, mujoco.MjData(model), renderer, RLock()


def _semantic_ids(image, model):
    mapped = np.zeros(image.shape[:2], np.int16)
    for label, name in enumerate(OBJECT_NAMES, 1):
        mapped[(image[..., 0] == model.geom(name).id) & (image[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))] = label
    return mapped


def bake_static(height, width):
    """Bake the fixed scene without the moving ball, restoring shared model fields."""
    model, data, renderer, lock = _bundle(height, width)
    with lock:
        ball = model.geom("small_orange_ball").id
        group = model.geom_group[ball].copy()
        option = mujoco.MjvOption()
        option.geomgroup[5] = 0
        try:
            model.geom_group[ball] = 5
            mujoco.mj_forward(model, data)
            renderer.disable_depth_rendering()
            renderer.disable_segmentation_rendering()
            renderer.update_scene(data, camera="source", scene_option=option)
            rgb = renderer.render().copy()
            camera = Camera.from_mjv(renderer.scene.camera[0], renderer.scene.camera[1])
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            renderer.enable_segmentation_rendering()
            ids = _semantic_ids(renderer.render(), model)
        finally:
            model.geom_group[ball] = group
            renderer.disable_depth_rendering()
            renderer.disable_segmentation_rendering()
        depth[ids == 0] = 0
        return dict(
            rgb=rgb,
            depth=depth,
            ids=ids,
            camera=camera,
            radius=float(model.geom_size[ball, 0]),
            slices=int(model.vis.quality.numslices),
            stacks=int(model.vis.quality.numstacks),
            color=model.geom_rgba[ball, :3].copy(),
            light_position=model.light_pos[0].copy(),
            light_diffuse=float(model.light_diffuse[0, 0]),
            head_ambient=float(model.vis.headlight.ambient[0]),
            head_diffuse=float(model.vis.headlight.diffuse[0]),
            near=camera.near,
            far=float(model.vis.map.zfar * model.stat.extent),
        )


def copy_output(result, out):
    if out is None:
        return result
    if (
        not torch.is_tensor(out)
        or out.shape != result.shape
        or out.dtype != result.dtype
        or out.device != result.device
    ):
        raise ValueError("out must match frame shape, dtype and device")
    out.copy_(result)
    return out


def render_rows(rows, *, height, width, device, mode="rgb", out=None, backend="opengl"):
    """Render owned RGB, camera-axis depth, or semantic object IDs for each row."""
    if mode not in ("rgb", "depth", "ids"):
        raise ValueError(f"Unknown render mode {mode!r}")
    if backend == "raycast":
        from .g34_raycast import render_raycast

        return render_raycast(rows, height=height, width=width, mode=mode, out=out)
    host = rows.detach().cpu().numpy().reshape(-1, 7)
    model, data, renderer, lock = _bundle(height, width)
    images = []
    with lock:
        try:
            if mode == "depth":
                renderer.enable_depth_rendering()
            elif mode == "ids":
                renderer.enable_segmentation_rendering()
            else:
                renderer.disable_depth_rendering()
                renderer.disable_segmentation_rendering()
            for row in host:
                set_pose(model, data, row[:3], row[3:])
                renderer.update_scene(data, camera="source")
                image = renderer.render().copy()
                if mode == "ids":
                    image = _semantic_ids(image, model)
                images.append(image)
        finally:
            renderer.disable_depth_rendering()
            renderer.disable_segmentation_rendering()
    result = torch.from_numpy(np.stack(images)).to(device)
    return copy_output(result, out)
