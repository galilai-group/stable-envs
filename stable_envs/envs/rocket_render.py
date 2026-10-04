"""Rocket pixels from explicit current body poses and the cached camera pose.

The camera and visible links have different update times in PyFlyt. TinyRenderer
also forms projection*(view*model) with float32 reverse-order dot products; at
large world coordinates an equivalent double-precision expression draws a
different image. These are family-local rendering details, not simulator State.
"""

import threading

import numpy as np
import torch

from . import rocket_landing as frozen

ND = frozen.ND
FIELDS = {"source_pose": (3, 7), "camera_pose": (7,), "rgba": (6, 4)}


def project(state):
    return {
        ("renderer", "source_pose"): torch.stack(
            [
                torch.cat(
                    [
                        state["current", "physics", f"body_{i}", "position"],
                        state["current", "physics", f"body_{i}", "quaternion"],
                    ],
                    -1,
                )
                for i in range(3)
            ],
            1,
        ).to(torch.float64),
        ("renderer", "camera_pose"): state["current", "diagnostic", "cached_links"][:, 0, :7]
        .to(torch.float64)
        .clone(),
        ("renderer", "rgba"): state["render", "rgba"].to(torch.float64).clone(),
    }


def camera_matrices(link, bullet):
    rotation = np.array(bullet.getMatrixFromQuaternion(link[3:7])).reshape(3, 3)
    eye = link[:3] + rotation @ frozen.CAM_OFFSET
    angles = np.array(bullet.getEulerFromQuaternion(link[3:7]))
    angles[1] += np.pi / 6
    up = np.array(bullet.getMatrixFromQuaternion(bullet.getQuaternionFromEuler(angles))).reshape(3, 3)[:, 2]
    return bullet.computeViewMatrix(eye, link[:3], up), bullet.computeProjectionMatrixFOV(90.0, 1.0, 0.1, 255.0)


def _mm(a, b):
    out = np.zeros(np.broadcast_shapes(a.shape[:-2], b.shape[:-2]) + (4, 4), np.float32)
    for k in (3, 2, 1, 0):
        out += a[..., :, k : k + 1] * b[..., k : k + 1, :]
    return out


class _Scene(ND.Scene):
    def clip(self, xpos, xmat, view, proj):
        _, normals, world = super().clip(xpos, xmat, view, proj)
        batch = len(xpos)
        positions = torch.sparse.mm(self.S4, self.source_mvp.permute(1, 3, 0, 2).reshape(4 * self.G, 4 * batch))
        return positions.reshape(self.V, batch, 4).permute(1, 0, 2).contiguous(), normals, world


class _Renderer(ND.BatchRenderer):
    def _rasterize(self, positions, resolution):
        from ._rocket_raster import rasterize

        raster = rasterize(positions, self.scene.face, resolution)
        # Rocket uses nearest textures without mipmaps. Nvdiffrast's shared
        # interpolation API still requires the four-channel derivative buffer.
        return raster, torch.zeros_like(raster)


def _gpu_context(env, size, ss):
    cache = env.__dict__.setdefault("_rocket_pixel_gpu", {})
    key = (size, ss)
    if key not in cache:
        vertices, normals, faces, geom, uv = frozen.build_scene_arrays()
        rgba = np.ones((frozen.N_GEOMS, 4))
        rgba[11] = frozen.FLAME_RGBA
        texture = np.full(frozen.N_GEOMS, -1, np.int64)
        texture[0] = 0
        scene = _Scene.from_triangles(
            vertices,
            normals,
            faces,
            geom,
            rgba,
            uv=uv,
            textures={0: frozen.checker_texture()},
            tex_of_geom=texture,
            device=str(env.device),
            tex_policy="none",
        )
        renderer = _Renderer(
            None,
            res=size,
            device=str(env.device),
            supersample=ss,
            mipmap=False,
            skybox=False,
            quant="trunc",
            scene=scene,
            shading="tiny",
            tiny=ND.TinyShading(),
            tex_filter="nearest",
            sample_offset=(0.5, 0.5),
        )
        renderer.owns_scene = True
        renderer.background = np.ones(3)
        per_world = frozen.NVD_BYTES_PER_PIXEL * (size * ss) ** 2 + frozen.NVD_BYTES_PER_VERTEX * scene.V
        cache[key] = renderer, max(1, int(frozen.NVD_BUDGET_BYTES // per_world))
    return cache[key]


def _cpu_context(env):
    from pathlib import Path

    import pybullet as bullet
    import pybullet_data
    import PyFlyt

    if not hasattr(env, "_rocket_pixel_cpu"):
        client = bullet.connect(bullet.DIRECT)
        try:
            assets = Path(PyFlyt.__file__).parent
            ground = bullet.loadURDF(
                str(Path(pybullet_data.getDataPath()) / "plane.urdf"), useFixedBase=True, physicsClientId=client
            )
            rocket = bullet.loadURDF(
                str(assets / "models/vehicles/rocket/rocket.urdf"),
                flags=bullet.URDF_USE_INERTIA_FROM_FILE,
                physicsClientId=client,
            )
            pad = bullet.loadURDF(str(assets / "models/landing_pad.urdf"), physicsClientId=client)
            names = {
                bullet.getJointInfo(rocket, i, physicsClientId=client)[12].decode(): i
                for i in range(bullet.getNumJoints(rocket, physicsClientId=client))
            }
            env._rocket_pixel_cpu = client, (ground, rocket, pad), names
        except BaseException:
            bullet.disconnect(client)
            raise
    return env._rocket_pixel_cpu


def render(env, state, size, ss, chunk_size=None):
    import pybullet as bullet

    env._check_open("render")
    host = {k: state["renderer", k].detach().cpu().numpy() for k in FIELDS}
    body, links, colors = host["source_pose"], host["camera_pose"], host["rgba"]
    lock = env.__dict__.setdefault("_rocket_pixel_lock", threading.RLock())
    with lock:
        if env.device.type == "cpu":
            client, bodies, names = _cpu_context(env)
            images = []
            for i in range(len(state)):
                for j, uid in enumerate(bodies):
                    bullet.resetBasePositionAndOrientation(uid, body[i, j, :3], body[i, j, 3:], physicsClientId=client)
                bullet.changeVisualShape(bodies[1], -1, rgbaColor=colors[i, 0], physicsClientId=client)
                bullet.changeVisualShape(bodies[2], -1, rgbaColor=colors[i, 4], physicsClientId=client)
                for name, index in names.items():
                    color = (
                        1
                        if "fin" in name
                        else 2
                        if "leg" in name
                        else 3
                        if "booster" in name
                        else 5
                        if "flame" in name
                        else None
                    )
                    if color is not None:
                        bullet.changeVisualShape(bodies[1], index, rgbaColor=colors[i, color], physicsClientId=client)
                view, projection = camera_matrices(links[i], bullet)
                raw = bullet.getCameraImage(
                    size * ss,
                    size * ss,
                    viewMatrix=view,
                    projectionMatrix=projection,
                    renderer=bullet.ER_TINY_RENDERER,
                    physicsClientId=client,
                )[2]
                image = np.asarray(raw, np.uint8).reshape(size * ss, size * ss, 4)[..., :3]
                if ss > 1:
                    image = np.rint(image.reshape(size, ss, size, ss, 3).mean((1, 3))).astype(np.uint8)
                images.append(image.copy())
            return torch.from_numpy(np.stack(images))
        if env.device.type != "cuda":
            raise ValueError("Rocket rendering supports CPU TinyRenderer or CUDA nvdiffrast")
        renderer, limit = _gpu_context(env, size, ss)
        limit = min(limit, chunk_size or limit)
        positions = np.zeros((len(state), 13, 3), np.float64)
        quaternions = np.zeros((len(state), 13, 4), np.float64)
        quaternions[..., 3] = 1
        for geom, source in [(0, 0), (1, 1), (12, 2)]:
            positions[:, geom], quaternions[:, geom] = body[:, source, :3], body[:, source, 3:]
        local_pos, local_quat = frozen.link_local_frames()
        rotation = frozen.quat_to_mat(quaternions[:, 1])
        for j in range(1, len(local_pos)):
            positions[:, 1 + j] = positions[:, 1] + np.einsum("bij,j->bi", rotation, local_pos[j])
            quaternions[:, 1 + j] = frozen.quat_mul(quaternions[:, 1], local_quat[j : j + 1])
        rotations = frozen.quat_to_mat(quaternions)
        model = np.zeros((len(state), 13, 4, 4), np.float32)
        model[..., :3, :3], model[..., :3, 3], model[..., 3, 3] = rotations, positions, 1
        views, projections = [], []
        for link in links:
            view, projection = camera_matrices(link, bullet)
            views.append(np.asarray(view, np.float32).reshape(4, 4).T)
            projections.append(np.asarray(projection, np.float32).reshape(4, 4).T)
        mvp = _mm(np.stack(projections)[:, None], _mm(np.stack(views)[:, None], model))
        origin = positions[:, 1].copy()
        positions -= origin[:, None]
        eye, camera = frozen.camera_pose(links[:, :3] - origin, links[:, 3:])
        # rgba is authoritative in the complete projection on both devices.
        # The old style/pose fields remain for legacy compact consumers only.
        material = np.ones((len(state), frozen.N_GEOMS, 3), np.float64)
        material[:, 1] = colors[:, 0, :3]
        material[:, 3] = colors[:, 3, :3]
        material[:, 4:8] = colors[:, 1:2, :3]
        material[:, 8:11] = colors[:, 2:3, :3]
        material[:, 11] = colors[:, 5, :3]
        material[:, 12] = colors[:, 4, :3]
        output = []
        for start in range(0, len(state), limit):
            stop = start + limit
            renderer.scene.source_mvp = torch.as_tensor(mvp[start:stop], device=env.device)
            image = renderer.render(
                torch.as_tensor(positions[start:stop], dtype=torch.float32, device=env.device),
                torch.as_tensor(rotations[start:stop], dtype=torch.float32, device=env.device),
                ND.Camera(eye[start:stop], camera[start:stop], frozen.CAM_FOV, frozen.CAM_NEAR, frozen.CAM_FAR),
                mat_rgb=torch.as_tensor(material[start:stop], dtype=torch.float32, device=env.device),
            )
            output.append(image)
        return torch.cat(output)


def close(env):
    if hasattr(env, "_rocket_pixel_cpu"):
        import pybullet

        pybullet.disconnect(env._rocket_pixel_cpu[0])
        del env._rocket_pixel_cpu
    for renderer, _ in env.__dict__.pop("_rocket_pixel_gpu", {}).values():
        renderer.close()
