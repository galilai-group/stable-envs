import numpy as np

from ...render.cpu3d import texture_array


class Textures:
    """Upload floor mipmaps and cache the sky for a fixed-orientation tracking camera."""

    def __init__(self, model, context, sky: np.ndarray):
        import mujoco
        import warp as wp

        registry = list(context.textures_registry)
        materials = {
            int(model.geom_matid[g])
            for g in range(model.ngeom)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE and model.geom_matid[g] >= 0
        }
        for material in materials:
            texture = int(model.mat_texid[material, 1])
            if texture < 0:
                continue
            rgb = texture_array(model, texture).astype(np.float32) / 255
            rgba = np.concatenate((rgb, np.ones((*rgb.shape[:2], 1), dtype=np.float32)), axis=-1)
            registry[texture] = wp.Texture2D(
                rgba,
                num_mip_levels=0,
                filter_mode=wp.TextureFilterMode.LINEAR,
                address_mode=wp.TextureAddressMode.WRAP,
            )
        context.textures_registry = registry
        context.textures = wp.array(registry, dtype=wp.Texture2D)
        rgb = np.asarray(sky, dtype=np.uint32).reshape(-1, 3)
        self.plate = wp.array((255 << 24) | (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2], dtype=wp.uint32)

        @wp.kernel
        def replace_sky(segmentation: wp.array2d[wp.vec2i], plate: wp.array[wp.uint32], output: wp.array2d[wp.uint32]):
            world, pixel = wp.tid()
            if segmentation[world, pixel][0] == -1:
                output[world, pixel] = plate[pixel]

        self.kernel, self.context, self.wp = replace_sky, context, wp

    def render(self) -> None:
        c = self.context
        self.wp.launch(self.kernel, dim=c.rgb_data.shape, inputs=[c.seg_data, self.plate, c.rgb_data])
