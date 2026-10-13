"""Filtered floor shading and a reflected-ray pass for TrackCom."""

import dataclasses

import numpy as np

from ...render.cpu3d import texture_array
from ...render.texture_mip import plane_grid, plane_texgen


class Reflection:
    """Reuse the primary renderer's segmentation, BVH and lighting on floor hits.

    ``palette`` (texel slots of the floor checker) lets each world recolour those
    texels: the floor samples a texture of per-texel palette weights instead, and
    the texels outside the slots keep their one shared colour."""

    def __init__(self, model, data, context, source_model, palette=None):
        import mujoco
        import torch
        import warp as wp
        from mujoco_warp._src import render as upstream

        materials = np.asarray(source_model.geom_matid)
        reflective = [
            i for i, material in enumerate(materials) if material >= 0 and source_model.mat_reflectance[material] > 0
        ]
        if len(reflective) != 1 or source_model.geom_type[reflective[0]] != mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError("TrackCom reflection requires exactly one reflective plane")
        if context.nrender != 1 or context.splat_count or context.bvh_nflexgeom:
            raise ValueError("TrackCom reflection requires one camera and rigid geometry")
        if not bool(context.render_seg.numpy().all()):
            raise ValueError("TrackCom reflection requires primary render_seg=True")
        if not context.use_precomputed_rays:
            raise ValueError("TrackCom reflection requires precomputed camera rays")
        flags = {
            f.name: getattr(context, f.name)
            for f in dataclasses.fields(context)
            if f.type in (int, wp.uint32, bool, float, wp.vec3)
        }
        cast = upstream._make_cast_ray(context.geom_ray_types)
        shadow_cast = upstream._make_cast_ray(context.geom_ray_types, first_hit=True)
        lighting = upstream._make_compute_lighting(shadow_cast)
        sample_texture = upstream.sample_texture
        plane = reflective[0]
        reflectance = float(source_model.mat_reflectance[materials[plane]])
        nlight = int(model.nlight)
        plane_material = int(materials[plane])
        plane_texture = int(source_model.mat_texid[materials[plane], 1])
        texture_width = int(source_model.tex_width[plane_texture]) if plane_texture >= 0 else 1
        texture_height = int(source_model.tex_height[plane_texture]) if plane_texture >= 0 else 1
        width, height = map(int, context.cam_res.numpy()[0])
        size = np.asarray(source_model.geom_size[plane, :2], dtype=float)
        repeat = np.asarray(source_model.mat_texrepeat[plane_material], dtype=float)
        uniform = bool(source_model.mat_texuniform[plane_material])
        scale, offset = plane_texgen(size, repeat, uniform, tuple(map(int, mujoco.__version__.split(".")[:2])))
        uv_sx, uv_sy = map(float, scale)
        uv_ox, uv_oy = map(float, offset)
        grids = [
            plane_grid(
                float(size[i]),
                float(source_model.geom_size[plane, 2]),
                float(source_model.vis.map.zfar * source_model.stat.extent),
                float(repeat[i]),
                mujoco.mjMAXPLANEGRID,
            )
            for i in range(2)
        ]
        recolour = palette is not None and plane_texture >= 0
        nworld = int(data.nworld)
        device = data.geom_xpos.device
        if recolour:
            texels = texture_array(source_model, plane_texture).reshape(-1, 3)
            masks = np.stack([np.asarray(m, bool).reshape(-1) for m in palette["masks"]])
            rest = ~masks.any(0)
            if not (texels[rest] == texels[rest][:1]).all():
                raise ValueError("TrackCom floor recolouring requires the unslotted texels to share one colour")
            colours = np.stack([texels[m][0] for m in (*masks, rest)]).astype(np.float32) / 255
            weights = np.stack([*masks, rest, np.ones_like(rest)], -1).astype(np.float32)
            weights = weights.reshape(texture_height, texture_width, 4)
            texture = wp.Texture2D(
                weights,
                num_mip_levels=0,
                filter_mode=wp.TextureFilterMode.LINEAR,
                address_mode=wp.TextureAddressMode.WRAP,
            )
            extra = {
                "floor_weights": wp.array([texture], dtype=wp.Texture2D, device=device),
                "floor_palette": wp.array(np.tile(colours, (nworld, 1, 1)), dtype=wp.vec3, device=device),
                "floor_recolour": wp.zeros(nworld, dtype=int, device=device),
            }
            self._default = torch.as_tensor(colours[:-1], device=wp.device_to_torch(device))
            # The array holds only the texture's handle; the object must outlive the kernel's reads.
            self._weights = texture
        else:
            extra = {
                "floor_weights": context.textures,
                "floor_palette": wp.zeros((1, 3), dtype=wp.vec3, device=device),
                "floor_recolour": wp.zeros(1, dtype=int, device=device),
            }
        self.palette = palette if recolour else None
        self._recoloured = False
        # Small floor grids are shaded once per cell corner and shadow mask, then gathered per pixel.
        cells = (len(grids[0]) - 1) * (len(grids[1]) - 1)
        shadow_masks = 1 << nlight if flags["use_shadows"] else 1
        tabulate = nlight <= 4 and cells * 4 * shadow_masks <= width * height
        extra["tabulated"] = wp.array([int(tabulate)], dtype=int, device=device)
        extra["vertex_shade"] = wp.zeros(
            (nworld, cells * 4 * shadow_masks if tabulate else 1), dtype=wp.vec3, device=device
        )
        # Per-world bounds of every non-floor geom: rays that miss them cannot hit anything but the floor.
        extra["scene_lower"] = wp.zeros(nworld, dtype=wp.vec3, device=device)
        extra["scene_upper"] = wp.zeros(nworld, dtype=wp.vec3, device=device)
        extra["geom_lower"] = context.lower
        extra["geom_upper"] = context.upper

        @wp.struct
        class Scene:
            geom_type: wp.array[int]
            geom_dataid: wp.array2d[int]
            geom_matid: wp.array2d[int]
            geom_size: wp.array2d[wp.vec3]
            geom_rgba: wp.array2d[wp.vec4]
            flex_vertadr: wp.array[int]
            flex_edge: wp.array[wp.vec2i]
            flex_radius: wp.array[float]
            mesh_faceadr: wp.array[int]
            mat_texid: wp.array3d[int]
            mat_texrepeat: wp.array2d[wp.vec2]
            mat_rgba: wp.array2d[wp.vec4]
            mat_specular: wp.array2d[float]
            mat_shininess: wp.array2d[float]
            mat_emission: wp.array2d[float]
            light_active: wp.array2d[bool]
            light_type: wp.array2d[int]
            light_castshadow: wp.array2d[bool]
            light_attenuation: wp.array2d[wp.vec3]
            light_cutoff: wp.array2d[float]
            light_exponent: wp.array2d[float]
            light_ambient: wp.array2d[wp.vec3]
            light_diffuse: wp.array2d[wp.vec3]
            light_specular: wp.array2d[wp.vec3]
            geom_xpos: wp.array2d[wp.vec3]
            geom_xmat: wp.array2d[wp.mat33]
            flexvert_xpos: wp.array2d[wp.vec3]
            cam_xpos: wp.array2d[wp.vec3]
            cam_xmat: wp.array2d[wp.mat33]
            light_xpos: wp.array2d[wp.vec3]
            light_xdir: wp.array2d[wp.vec3]
            bvh_id: wp.uint64
            group_root: wp.array[int]
            enabled_geom_ids: wp.array[int]
            mesh_bvh_id: wp.array[wp.uint64]
            hfield_bvh_id: wp.array[wp.uint64]
            flex_geom_flexid: wp.array[int]
            flex_geom_edgeid: wp.array[int]
            flex_bvh_id: wp.array[wp.uint64]
            flex_group_root: wp.array2d[int]
            mesh_facetexcoord: wp.array[wp.vec3i]
            mesh_texcoord: wp.array[wp.vec2]
            mesh_texcoord_offsets: wp.array[int]
            textures: wp.array[wp.Texture2D]
            ray: wp.array[wp.vec3]
            cam_id_map: wp.array[int]
            seg_data: wp.array2d[wp.vec2i]
            rgb_data: wp.array2d[wp.uint32]
            grid_x: wp.array[float]
            grid_y: wp.array[float]
            floor_weights: wp.array[wp.Texture2D]
            floor_palette: wp.array2d[wp.vec3]
            floor_recolour: wp.array[int]
            vertex_shade: wp.array2d[wp.vec3]
            tabulated: wp.array[int]
            scene_lower: wp.array[wp.vec3]
            scene_upper: wp.array[wp.vec3]
            geom_lower: wp.array[wp.vec3]
            geom_upper: wp.array[wp.vec3]

        ray_plane = upstream.ray_plane

        @wp.func
        def misses(s: Scene, world: int, origin: wp.vec3, direction: wp.vec3):
            lower = s.scene_lower[world]
            upper = s.scene_upper[world]
            near = float(-1.0e30)  # noqa: UP018
            far = float(1.0e30)  # noqa: UP018
            for axis in range(3):
                if direction[axis] == 0.0:
                    if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                        return True
                else:
                    inverse = 1.0 / direction[axis]
                    t1 = (lower[axis] - origin[axis]) * inverse
                    t2 = (upper[axis] - origin[axis]) * inverse
                    near = wp.max(near, wp.min(t1, t2))
                    far = wp.min(far, wp.max(t1, t2))
            return far < 0.0 or far < near

        @wp.func
        def blocked(
            s: Scene, world: int, point: wp.vec3, normal: wp.vec3, kind: int, pos: wp.vec3, direction: wp.vec3
        ):
            shadow_direction = -direction
            limit = float(1.0e8)  # noqa: UP018
            if kind != 1:
                delta = pos - point
                shadow_direction = wp.normalize(delta)
                limit = wp.length(delta) - 1.0e-3
            start = point + normal * 1.0e-4
            if misses(s, world, start, shadow_direction):
                floor, floor_normal = ray_plane(
                    s.geom_xpos[world, wp.static(plane)],
                    s.geom_xmat[world, wp.static(plane)],
                    s.geom_size[world % s.geom_size.shape[0], wp.static(plane)],
                    start,
                    shadow_direction,
                )
                if floor < 0.0:
                    return False
            blocker, distance, hit_normal, u, v, face, mesh = shadow_cast(
                s.geom_type,
                s.geom_dataid,
                s.geom_size,
                s.flex_vertadr,
                s.flex_edge,
                s.flex_radius,
                s.geom_xpos,
                s.geom_xmat,
                s.flexvert_xpos,
                s.bvh_id,
                s.group_root[world],
                world,
                wp.static(flags["bvh_ngeom"]),
                wp.static(flags["bvh_nflexgeom"]),
                s.enabled_geom_ids,
                s.mesh_bvh_id,
                s.hfield_bvh_id,
                s.flex_geom_flexid,
                s.flex_geom_edgeid,
                s.flex_bvh_id,
                s.flex_group_root,
                start,
                shadow_direction,
                limit,
                wp.static(flags["enable_backface_culling"]),
            )
            return blocker != -1

        @wp.func
        def cell(grid: wp.array[float], position: float):
            lo = int(0)  # noqa: UP018
            hi = grid.shape[0] - 1
            while hi - lo > 1:
                middle = (lo + hi) // 2
                if grid[middle] <= position:
                    lo = middle
                else:
                    hi = middle
            return lo

        @wp.func
        def illuminate(
            s: Scene,
            world: int,
            normal: wp.vec3,
            point: wp.vec3,
            view: wp.vec3,
            spec: float,
            shin: float,
            active: bool,
            kind: int,
            shadow: bool,
            pos: wp.vec3,
            direction: wp.vec3,
            attenuation: wp.vec3,
            cutoff: float,
            exponent: float,
            diffuse: wp.vec3,
            light_spec: wp.vec3,
        ):
            return lighting(
                s.geom_type,
                s.geom_dataid,
                s.geom_size,
                s.flex_vertadr,
                s.flex_edge,
                s.flex_radius,
                s.geom_xpos,
                s.geom_xmat,
                s.flexvert_xpos,
                False,
                s.bvh_id,
                s.group_root[world],
                wp.static(flags["bvh_ngeom"]),
                wp.static(flags["bvh_nflexgeom"]),
                s.enabled_geom_ids,
                world,
                s.mesh_bvh_id,
                s.hfield_bvh_id,
                s.flex_geom_flexid,
                s.flex_geom_edgeid,
                s.flex_bvh_id,
                s.flex_group_root,
                active,
                kind,
                shadow,
                pos,
                direction,
                attenuation,
                cutoff,
                exponent,
                diffuse,
                light_spec,
                normal,
                point,
                view,
                spec,
                shin,
                wp.static(flags["enable_backface_culling"]),
                wp.static(flags["enable_specular"]),
                False,
                True,
            )

        @wp.func
        def shade(
            s: Scene,
            world: int,
            geom: int,
            point: wp.vec3,
            normal: wp.vec3,
            view: wp.vec3,
            forward: wp.vec3,
            origin: wp.vec3,
            u: float,
            v: float,
            face: int,
            mesh: int,
            lod: float,
            gain: float,
            blocked_mask: int,
        ):
            material = s.geom_matid[world % s.geom_matid.shape[0], geom]
            color = s.geom_rgba[world % s.geom_rgba.shape[0], geom]
            spec = float(0.5)  # noqa: UP018
            shin = float(64.0)  # noqa: UP018
            emission = float(0.0)  # noqa: UP018
            if material >= 0:
                color = s.mat_rgba[world % s.mat_rgba.shape[0], material]
                if wp.static(flags["enable_specular"]):
                    spec = s.mat_specular[world % s.mat_specular.shape[0], material]
                    shin = s.mat_shininess[world % s.mat_shininess.shape[0], material] * 128.0
                if wp.static(flags["enable_emission"]):
                    emission = s.mat_emission[world % s.mat_emission.shape[0], material]
            base = wp.vec3(color[0], color[1], color[2])
            if wp.static(flags["use_textures"]):
                if material >= 0 and geom != wp.static(plane):
                    texture = s.mat_texid[world % s.mat_texid.shape[0], material, 1]
                    if texture >= 0:
                        tex_color = sample_texture(
                            s.geom_type,
                            s.mesh_faceadr,
                            geom,
                            s.mat_texrepeat[world % s.mat_texrepeat.shape[0], material],
                            s.textures[texture],
                            s.geom_xpos[world, geom],
                            s.geom_xmat[world, geom],
                            s.mesh_facetexcoord,
                            s.mesh_texcoord,
                            s.mesh_texcoord_offsets,
                            point,
                            u,
                            v,
                            face,
                            mesh,
                        )
                        base = wp.cw_mul(base, tex_color)
            base *= gain
            result = base * emission
            shadow_passes = wp.vec3(0.0)
            if wp.static(flags["use_ambient_lighting"]):
                if wp.static(flags["headlight_active"]):
                    result += wp.cw_mul(base, wp.static(flags["headlight_ambient"]))
                elif wp.static(nlight == 0):
                    result += base * wp.static(upstream.NO_LIGHT_AMBIENT_FALLBACK)
            for light in range(wp.static(nlight)):
                active = s.light_active[world % s.light_active.shape[0], light] and (blocked_mask & (1 << light)) == 0
                separate_pass = (
                    geom == wp.static(plane)
                    and s.light_castshadow[world % s.light_castshadow.shape[0], light]
                    and wp.static(flags["use_shadows"])
                )
                ambient = wp.vec3(0.0)
                if wp.static(flags["use_ambient_lighting"] and flags["enable_per_light_ambient"]):
                    if active:
                        ambient = wp.cw_mul(base, s.light_ambient[world % s.light_ambient.shape[0], light])
                diffuse, specular = illuminate(
                    s,
                    world,
                    normal,
                    point,
                    view,
                    spec,
                    shin,
                    active,
                    s.light_type[world % s.light_type.shape[0], light],
                    False,
                    s.light_xpos[world, light],
                    s.light_xdir[world, light],
                    s.light_attenuation[world % s.light_attenuation.shape[0], light],
                    s.light_cutoff[world % s.light_cutoff.shape[0], light] * (wp.pi / 180.0),
                    s.light_exponent[world % s.light_exponent.shape[0], light],
                    s.light_diffuse[world % s.light_diffuse.shape[0], light],
                    s.light_specular[world % s.light_specular.shape[0], light],
                )
                contribution = ambient + wp.cw_mul(base, diffuse) + specular
                if separate_pass:
                    shadow_passes += wp.min(wp.max(contribution, wp.vec3(0.0)), wp.vec3(1.0))
                else:
                    result += contribution
            if wp.static(flags["headlight_active"]):
                diffuse, specular = illuminate(
                    s,
                    world,
                    normal,
                    point,
                    view,
                    spec,
                    shin,
                    True,
                    1,
                    False,
                    origin,
                    forward,
                    wp.vec3(1.0, 0.0, 0.0),
                    0.0,
                    0.0,
                    wp.static(flags["headlight_diffuse"]),
                    wp.static(flags["headlight_specular"]),
                )
                result += wp.cw_mul(base, diffuse) + specular
            result = wp.min(wp.max(result, wp.vec3(0.0)), wp.vec3(1.0))
            # Each GL shadow-light pass clamps independently, then adds to the regular pass.
            return result + shadow_passes

        @wp.func
        def pack(colour: wp.vec3):
            rgb = colour * 255.0
            red = wp.uint32(wp.clamp(rgb[0] + 0.5, 0.0, 255.0))
            green = wp.uint32(wp.clamp(rgb[1] + 0.5, 0.0, 255.0))
            blue = wp.uint32(wp.clamp(rgb[2] + 0.5, 0.0, 255.0))
            return (wp.uint32(255) << wp.uint32(24)) | (red << wp.uint32(16)) | (green << wp.uint32(8)) | blue

        @wp.func
        def shade_vertex(s: Scene, world: int, x: float, y: float, blocked_mask: int):
            camera = s.cam_id_map[0]
            origin = s.cam_xpos[world, camera]
            rotation = s.cam_xmat[world, camera]
            plane_rotation = s.geom_xmat[world, wp.static(plane)]
            vertex = s.geom_xpos[world, wp.static(plane)] + plane_rotation @ wp.vec3(x, y, 0.0)
            return shade(
                s,
                world,
                wp.static(plane),
                vertex,
                plane_rotation[:, 2],
                wp.normalize(origin - vertex),
                -rotation[:, 2],
                origin,
                0.0,
                0.0,
                -1,
                -1,
                0.0,
                1.0,
                blocked_mask,
            )

        # Repeats the pixel kernel's corner loop so the compiler rounds each corner's shade alike in both.
        @wp.kernel(module="unique", enable_backward=False)
        def vertex_pixels(s: Scene):
            world, square, mask = wp.tid()
            ix = square % wp.static(len(grids[0]) - 1)
            iy = square // wp.static(len(grids[0]) - 1)
            x0, x1 = s.grid_x[ix] + 0.0, s.grid_x[ix + 1] + 0.0
            y0, y1 = s.grid_y[iy] + 0.0, s.grid_y[iy + 1] + 0.0
            for corner in range(4):
                vx, vy = x0, y0
                if corner == 1:
                    vx = x1
                elif corner == 2:
                    vy = y1
                elif corner == 3:
                    vx, vy = x1, y1
                index = (square * 4 + corner) * wp.static(shadow_masks) + mask
                s.vertex_shade[world, index] = shade_vertex(s, world, vx, vy, mask)

        @wp.kernel(module="unique", enable_backward=False)
        def scene_bounds(s: Scene):
            world = wp.tid()
            lower = wp.vec3(1.0e30)
            upper = wp.vec3(-1.0e30)
            for local in range(wp.static(flags["bvh_ngeom"])):
                if s.enabled_geom_ids[local] != wp.static(plane):
                    lower = wp.min(lower, s.geom_lower[world * wp.static(flags["bvh_ngeom"]) + local])
                    upper = wp.max(upper, s.geom_upper[world * wp.static(flags["bvh_ngeom"]) + local])
            pad = 1.0e-3 * (1.0 + wp.max(wp.max(wp.abs(lower[0]), wp.abs(lower[1])), wp.abs(lower[2])))
            pad = wp.max(pad, 1.0e-3 * (1.0 + wp.max(wp.max(wp.abs(upper[0]), wp.abs(upper[1])), wp.abs(upper[2]))))
            s.scene_lower[world] = lower - wp.vec3(pad)
            s.scene_upper[world] = upper + wp.vec3(pad)

        @wp.kernel(module="unique", enable_backward=False)
        def reflected_pixels(s: Scene):
            world, pixel = wp.tid()
            if s.seg_data[world, pixel][0] != wp.static(plane):
                return
            camera = s.cam_id_map[0]
            origin = s.cam_xpos[world, camera]
            rotation = s.cam_xmat[world, camera]
            direction = rotation @ s.ray[pixel]
            normal_plane = s.geom_xmat[world, wp.static(plane)][:, 2]
            plane_pos = s.geom_xpos[world, wp.static(plane)]
            distance = wp.dot(plane_pos - origin, normal_plane) / wp.dot(direction, normal_plane)
            on_plane = origin + direction * distance
            footprint = float(1.0)  # noqa: UP018
            plane_rotation = s.geom_xmat[world, wp.static(plane)]
            for axis in range(2):
                neighbor = int(0)  # noqa: UP018
                if axis == 0:
                    neighbor = pixel + 1
                    if pixel % wp.static(width) == wp.static(width - 1):
                        neighbor = pixel - 1
                else:
                    neighbor = pixel + wp.static(width)
                    if pixel // wp.static(width) == wp.static(height - 1):
                        neighbor = pixel - wp.static(width)
                neighbor = wp.clamp(neighbor, 0, wp.static(width * height - 1))
                ray_neighbor = rotation @ s.ray[neighbor]
                divisor = wp.dot(ray_neighbor, normal_plane)
                if wp.abs(divisor) > 1.0e-8:
                    neighbor_point = origin + ray_neighbor * (wp.dot(plane_pos - origin, normal_plane) / divisor)
                    delta = wp.transpose(plane_rotation) @ (neighbor_point - on_plane)
                    uv_delta = wp.vec2(
                        delta[0] * 0.5 * wp.static(uv_sx * texture_width),
                        delta[1] * 0.5 * wp.static(uv_sy * texture_height),
                    )
                    footprint = wp.max(footprint, wp.length(uv_delta))
            lod = wp.log(footprint) / wp.log(2.0)
            blocked_mask = int(0)  # noqa: UP018
            if wp.static(flags["use_shadows"]):
                for light in range(wp.static(nlight)):
                    if (
                        s.light_active[world % s.light_active.shape[0], light]
                        and s.light_castshadow[world % s.light_castshadow.shape[0], light]
                    ):
                        if blocked(
                            s,
                            world,
                            on_plane,
                            normal_plane,
                            s.light_type[world % s.light_type.shape[0], light],
                            s.light_xpos[world, light],
                            s.light_xdir[world, light],
                        ):
                            blocked_mask |= 1 << light
            local = wp.transpose(plane_rotation) @ (on_plane - plane_pos)
            ix = cell(s.grid_x, local[0])
            iy = cell(s.grid_y, local[1])
            # Materialize scalar values before reassigning corner coordinates in Warp branches.
            x0, x1 = s.grid_x[ix] + 0.0, s.grid_x[ix + 1] + 0.0
            y0, y1 = s.grid_y[iy] + 0.0, s.grid_y[iy + 1] + 0.0
            fx = (local[0] - x0) / (x1 - x0)
            fy = (local[1] - y0) / (y1 - y0)
            surface = wp.vec3(0.0)
            for corner in range(4):
                vx, vy = x0, y0
                weight = 1.0 - wp.max(fx, fy)
                if corner == 1:
                    vx = x1
                    weight = wp.max(fx - fy, 0.0)
                elif corner == 2:
                    vy = y1
                    weight = wp.max(fy - fx, 0.0)
                elif corner == 3:
                    vx, vy = x1, y1
                    weight = wp.min(fx, fy)
                if weight > 0.0:
                    # A runtime branch: compiling the per-vertex shading out reorders the pixel's own float math.
                    if s.tabulated[0] != 0:
                        cell_index = iy * wp.static(len(grids[0]) - 1) + ix
                        index = (cell_index * 4 + corner) * wp.static(shadow_masks) + blocked_mask
                        surface += weight * s.vertex_shade[world, index]
                    else:
                        surface += weight * shade_vertex(s, world, vx, vy, blocked_mask)
            if wp.static(flags["use_textures"] and plane_texture >= 0):
                uv = wp.vec2(
                    0.5 * local[0] * wp.static(uv_sx) + wp.static(uv_ox),
                    -0.5 * local[1] * wp.static(uv_sy) + wp.static(uv_oy),
                )
                uv -= wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
                tap = wp.texture_sample(s.textures[wp.static(plane_texture)], uv, dtype=wp.vec4, lod=lod)
                texel = wp.vec3(tap[0], tap[1], tap[2])
                if wp.static(recolour):
                    if s.floor_recolour[world] != 0:
                        w = wp.texture_sample(s.floor_weights[0], uv, dtype=wp.vec4, lod=lod)
                        texel = (
                            w[0] * s.floor_palette[world, 0]
                            + w[1] * s.floor_palette[world, 1]
                            + w[2] * s.floor_palette[world, 2]
                        )
                surface = wp.cw_mul(surface, texel)
            result = surface
            reflected = direction - 2.0 * wp.dot(direction, normal_plane) * normal_plane
            ray_origin = on_plane + normal_plane * 1.0e-4
            if misses(s, world, ray_origin, reflected):
                s.rgb_data[world, pixel] = pack(surface)
                return
            geom, distance, normal, u, v, face, mesh = cast(
                s.geom_type,
                s.geom_dataid,
                s.geom_size,
                s.flex_vertadr,
                s.flex_edge,
                s.flex_radius,
                s.geom_xpos,
                s.geom_xmat,
                s.flexvert_xpos,
                s.bvh_id,
                s.group_root[world],
                world,
                wp.static(flags["bvh_ngeom"]),
                wp.static(flags["bvh_nflexgeom"]),
                s.enabled_geom_ids,
                s.mesh_bvh_id,
                s.hfield_bvh_id,
                s.flex_geom_flexid,
                s.flex_geom_edgeid,
                s.flex_bvh_id,
                s.flex_group_root,
                ray_origin,
                reflected,
                1.0e10,
                wp.static(flags["enable_backface_culling"]),
            )
            if geom >= 0 and geom != wp.static(plane):
                point = ray_origin + reflected * distance
                forward = -rotation[:, 2]
                mirror = shade(
                    s,
                    world,
                    geom,
                    point,
                    normal,
                    -reflected,
                    forward,
                    origin,
                    u,
                    v,
                    face,
                    mesh,
                    0.0,
                    wp.static(reflectance),
                    0,
                )
                # MuJoCo adds the lit plane; reflectance scales mirror RGB, not specular.
                result = surface + mirror
            s.rgb_data[world, pixel] = pack(result)

        scene = Scene()
        grid_arrays = {
            name: wp.array(values, dtype=float, device=data.geom_xpos.device)
            for name, values in zip(("grid_x", "grid_y"), grids, strict=True)
        }
        grid_arrays.update(extra)
        for name in Scene.vars:
            if name in grid_arrays:
                setattr(scene, name, grid_arrays[name])
                continue
            owner = next(owner for owner in (model, data, context) if hasattr(owner, name))
            setattr(scene, name, getattr(owner, name))
        self._wp, self._kernel, self._scene = wp, reflected_pixels, scene
        self._bounds = scene_bounds
        self._vertices = (vertex_pixels, (nworld, cells, shadow_masks)) if tabulate else None
        self._shape = (int(data.nworld), int(context.total_rays))
        self._device = data.geom_xpos.device

    def recolour(self, rows) -> None:
        """Draw the first ``len(rows)`` worlds' checker in ``rows`` (n,P,3); None draws the compiled texture."""
        if rows is None and not self._recoloured:
            return
        flags = self._wp.to_torch(self._scene.floor_recolour)
        flags.zero_()
        self._recoloured = rows is not None
        if rows is None:
            return
        n = int(rows.shape[0])
        self._wp.to_torch(self._scene.floor_palette)[:n, : rows.shape[1]] = rows
        flags[:n] = (rows != self._default).flatten(1).any(1).to(flags.dtype)

    def render(self):
        launch = self._wp.launch
        launch(self._bounds, dim=self._shape[0], inputs=[self._scene], device=self._device)
        if self._vertices is not None:
            launch(self._vertices[0], dim=self._vertices[1], inputs=[self._scene], device=self._device)
        launch(self._kernel, dim=self._shape, inputs=[self._scene], device=self._device)
