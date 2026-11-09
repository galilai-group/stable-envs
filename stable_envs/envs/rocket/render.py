"""Rocket pixels from explicit current body poses and the cached camera pose.

The camera and visible links have different update times in PyFlyt. TinyRenderer
also forms projection*(view*model) with float32 reverse-order dot products; at
large world coordinates an equivalent double-precision expression draws a
different image. These are family-local rendering details, not simulator State.
"""

import math
import threading

import numpy as np
import torch

from ... import tuning
from . import legacy as geometry

ND = geometry.ND
BODY_COLOR, FIN_COLOR, LEG_COLOR, BOOSTER_COLOR, PAD_COLOR, FLAME_COLOR = range(6)
GROUND_GEOM, BODY_GEOM, BOOSTER_GEOM, FLAME_GEOM, PAD_GEOM = (
    geometry.GEOM_NAMES.index(name) for name in ("ground", "body", "booster", "flame", "pad")
)
FIN_GEOMS = slice(geometry.GEOM_NAMES.index("fin_pos_x"), geometry.GEOM_NAMES.index("leg_1"))
LEG_GEOMS = slice(geometry.GEOM_NAMES.index("leg_1"), FLAME_GEOM)
LINK_COLORS = (("fin", FIN_COLOR), ("leg", LEG_COLOR), ("booster", BOOSTER_COLOR), ("flame", FLAME_COLOR))
FIELDS = {"source_pose": (3, 7), "camera_pose": (7,), "rgba": (6, 4)}
#: Fewest CPU frames one ``render`` draws in render workers when the ``render_worker_min_rows`` knob is not set.
MIN_WORKER_ROWS = 2


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
    eye = link[:3] + rotation @ geometry.CAM_OFFSET
    angles = np.array(bullet.getEulerFromQuaternion(link[3:7]))
    angles[1] += math.radians(geometry.CAM_PITCH_DEG)
    up = np.array(bullet.getMatrixFromQuaternion(bullet.getQuaternionFromEuler(angles))).reshape(3, 3)[:, 2]
    return bullet.computeViewMatrix(eye, link[:3], up), bullet.computeProjectionMatrixFOV(
        geometry.CAM_FOV, 1.0, geometry.CAM_NEAR, geometry.CAM_FAR
    )


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


def tiny_texture(tex, clip, face, uv, textured, tid, resolution):
    """Sample ``tex`` at TinyRenderer's texel for each textured pixel; other pixels get ones.

    TinyRenderer takes double barycentrics of the unclipped triangle from float32
    screen points at integer pixel positions, sums float32 UVs in reverse vertex
    order, wraps them with modf and truncates them to a texel.
    """
    out = torch.ones(tid.shape + (3,), device=tex.device, dtype=tex.dtype)
    hit = (tid > 0) & textured[(tid - 1).clamp_min(0)]
    b, y, x = torch.nonzero(hit, as_tuple=True)
    if not len(b):
        return out
    corners = face[tid[b, y, x] - 1]
    c = clip[b[:, None], corners].to(torch.float32)
    half = torch.tensor(resolution / 2, dtype=torch.float32, device=tex.device)
    w = c[..., 3]
    sx = (((half * w) + (half * c[..., 0])) / w).double()
    sy = (((half * w) + (half * c[..., 1])) / w).double()
    s0 = (sx[:, 2] - sx[:, 0], sx[:, 1] - sx[:, 0], sx[:, 0] - x.double())
    s1 = (sy[:, 2] - sy[:, 0], sy[:, 1] - sy[:, 0], sy[:, 0] - y.double())
    ux = s0[1] * s1[2] - s0[2] * s1[1]
    uy = s0[2] * s1[0] - s0[0] * s1[2]
    uz = s0[0] * s1[1] - s0[1] * s1[0]
    wd = w.double()
    p = ((1.0 - (ux + uy) / uz) / wd[:, 0], (uy / uz) / wd[:, 1], (ux / uz) / wd[:, 2])
    total = (p[0] + p[1]) + p[2]
    bar = [(q / total).to(torch.float32)[:, None] for q in p]
    t = uv[corners].to(torch.float32)
    coord = torch.frac(((t[:, 2] * bar[2]) + (t[:, 1] * bar[1])) + (t[:, 0] * bar[0]))
    coord = torch.where(coord < 0, coord + 1, coord)
    h, wid = tex.shape[0], tex.shape[1]
    iu = (coord[:, 0] * wid).long().clamp(0, wid - 1)
    iv = (coord[:, 1] * h).long().clamp(0, h - 1)
    out[b, y, x] = tex[iv, iu]
    return out


class _Renderer(ND.BatchRenderer):
    def _rasterize(self, positions, resolution):
        from .raster import rasterize

        self._clip = positions
        raster = rasterize(positions, self.scene.face, resolution)
        # Rocket uses nearest textures without mipmaps. Nvdiffrast's shared
        # interpolation API still requires the four-channel derivative buffer.
        return raster, torch.zeros_like(raster)

    def _texture(self, S, rast, db, gid):
        (spec,) = S.textures.values()
        textured = (S.texid >= 0)[S.fgeom]
        return tiny_texture(
            spec["tex"], self._clip, S.face.long(), S.uv[:, :2], textured, rast[..., 3].long(), rast.shape[2]
        )


def _gpu_context(env, size, ss):
    cache = env.__dict__.setdefault("_rocket_pixel_gpu", {})
    key = (size, ss)
    if key not in cache:
        cache[key] = geometry.nvd_renderer(size, ss, env.device, _Scene, _Renderer)
    return cache[key]


def tiny_scene():
    """Open a PyBullet client with the ground, the rocket and the pad for TinyRenderer drawing."""
    from pathlib import Path

    import pybullet as bullet
    import pybullet_data
    import PyFlyt

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
        painted = []
        for i in range(bullet.getNumJoints(rocket, physicsClientId=client)):
            name = bullet.getJointInfo(rocket, i, physicsClientId=client)[12].decode()
            slot = next((slot for part, slot in LINK_COLORS if part in name), None)
            if slot is not None:
                painted.append((i, slot))
    except BaseException:
        bullet.disconnect(client)
        raise
    return client, (ground, rocket, pad), painted


def draw_tiny(scene, body, links, colors, size, ss, out=None):
    """Draw ``(B, 3, 7)`` body poses, ``(B, 7)`` camera link poses and ``(B, 6, 4)`` colours to uint8 frames."""
    import pybullet as bullet

    client, bodies, painted = scene
    images = np.empty((len(body), size, size, 3), np.uint8) if out is None else out
    for i in range(len(body)):
        for j, uid in enumerate(bodies):
            bullet.resetBasePositionAndOrientation(uid, body[i, j, :3], body[i, j, 3:], physicsClientId=client)
        bullet.changeVisualShape(bodies[1], -1, rgbaColor=colors[i, BODY_COLOR], physicsClientId=client)
        bullet.changeVisualShape(bodies[2], -1, rgbaColor=colors[i, PAD_COLOR], physicsClientId=client)
        for index, color in painted:
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
        # PyBullet built without NumPy returns a tuple of ints; ``bytes`` reads it faster than ``np.asarray``.
        pixels = raw if isinstance(raw, np.ndarray) else np.frombuffer(bytes(raw), np.uint8)
        image = np.asarray(pixels, np.uint8).reshape(size * ss, size * ss, 4)[..., :3]
        if ss > 1:
            image = np.rint(image.reshape(size, ss, size, ss, 3).mean((1, 3))).astype(np.uint8)
        images[i] = image
    return images


class TinyDrawer:
    """Picklable factory of a render worker's TinyRenderer drawer."""

    def __call__(self):
        scene = tiny_scene()

        def draw(args, frames):
            body, links, colors, size, ss = args
            draw_tiny(scene, body, links, colors, size, ss, frames)

        return draw


def draw_cpu(env, body, links, colors, size, ss, check=None):
    """Draw with TinyRenderer: in render workers on Linux, else in this process.

    A frame depends only on its row, so the split does not change the frames. ``check`` runs in this
    process while the workers draw; an error it raises discards the frames.
    """
    from ..._execution import render_pool

    workers = render_pool.pool(
        "rocket-tiny",
        TinyDrawer(),
        len(body),
        "rocket-render",
        size,
        min_rows=MIN_WORKER_ROWS,
        name="PyFlytRocketLanding render",
    )
    if workers is None:
        if check is not None:
            check()
        return env._simulator.draw(body, links, colors, size, ss)

    with workers.lock:
        return workers.render(
            len(body), (size, size, 3), lambda rows: (body[rows], links[rows], colors[rows], size, ss), during=check
        )


def render_cpu(env, body, links, colors, size, ss, check=None):
    """Draw host body poses, camera link poses and colours on CPU; ``check`` must pass before any frame returns."""
    env._check_open("render")
    with env.__dict__.setdefault("_rocket_pixel_lock", threading.RLock()):
        return torch.from_numpy(draw_cpu(env, body, links, colors, size, ss, check))


def render(env, state, size, ss, chunk_size=None, check=None):
    """Draw a RenderState with explicit poses; ``check`` must pass before any frame returns."""
    if chunk_size is not None and (isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0):
        raise ValueError("chunk_size must be a positive integer")
    import pybullet as bullet

    env._check_open("render")
    host = {k: state["renderer", k].detach().cpu().numpy() for k in FIELDS}
    body, links, colors = host["source_pose"], host["camera_pose"], host["rgba"]
    if env.device.type == "cpu":
        return render_cpu(env, body, links, colors, size, ss, check)
    lock = env.__dict__.setdefault("_rocket_pixel_lock", threading.RLock())
    with lock:
        if env.device.type != "cuda":
            raise ValueError("Rocket rendering supports CPU TinyRenderer or CUDA nvdiffrast")
        if check is not None:
            check()
        tuning.on_render("rocket-nvdiff", size, len(state), env)
        renderer, per_world = _gpu_context(env, size, ss)
        limit = geometry.nvd_worlds(per_world, env.device, size, len(state))
        limit = min(limit, chunk_size) if chunk_size is not None else limit
        positions = np.zeros((len(state), geometry.N_GEOMS, 3), np.float64)
        quaternions = np.zeros((len(state), geometry.N_GEOMS, 4), np.float64)
        quaternions[..., 3] = 1
        for source, geom in enumerate((GROUND_GEOM, BODY_GEOM, PAD_GEOM)):
            positions[:, geom], quaternions[:, geom] = body[:, source, :3], body[:, source, 3:]
        local_pos, local_quat = geometry.link_local_frames()
        rotation = geometry.quat_to_mat(quaternions[:, BODY_GEOM])
        for j in range(1, len(local_pos)):
            positions[:, BODY_GEOM + j] = positions[:, BODY_GEOM] + np.einsum("bij,j->bi", rotation, local_pos[j])
            quaternions[:, BODY_GEOM + j] = geometry.quat_mul(quaternions[:, BODY_GEOM], local_quat[j : j + 1])
        rotations = geometry.quat_to_mat(quaternions)
        geometry.cull_ground(positions, rotations, geometry.camera_pose(links[:, :3], links[:, 3:])[0])
        model = np.zeros((len(state), geometry.N_GEOMS, 4, 4), np.float32)
        model[..., :3, :3], model[..., :3, 3], model[..., 3, 3] = rotations, positions, 1
        views, projections = [], []
        for link in links:
            view, projection = camera_matrices(link, bullet)
            views.append(np.asarray(view, np.float32).reshape(4, 4).T)
            projections.append(np.asarray(projection, np.float32).reshape(4, 4).T)
        mvp = _mm(np.stack(projections)[:, None], _mm(np.stack(views)[:, None], model))
        origin = positions[:, BODY_GEOM].copy()
        positions -= origin[:, None]
        eye, camera = geometry.camera_pose(links[:, :3] - origin, links[:, 3:])
        material = np.ones((len(state), geometry.N_GEOMS, 3), np.float64)
        material[:, BODY_GEOM] = colors[:, BODY_COLOR, :3]
        material[:, BOOSTER_GEOM] = colors[:, BOOSTER_COLOR, :3]
        material[:, FIN_GEOMS] = colors[:, FIN_COLOR : FIN_COLOR + 1, :3]
        material[:, LEG_GEOMS] = colors[:, LEG_COLOR : LEG_COLOR + 1, :3]
        material[:, FLAME_GEOM] = colors[:, FLAME_COLOR, :3]
        material[:, PAD_GEOM] = colors[:, PAD_COLOR, :3]
        output = []
        for start in range(0, len(state), limit):
            stop = start + limit
            renderer.scene.source_mvp = torch.as_tensor(mvp[start:stop], device=env.device)
            image = renderer.render(
                torch.as_tensor(positions[start:stop], dtype=torch.float32, device=env.device),
                torch.as_tensor(rotations[start:stop], dtype=torch.float32, device=env.device),
                ND.Camera(eye[start:stop], camera[start:stop], geometry.CAM_FOV, geometry.CAM_NEAR, geometry.CAM_FAR),
                mat_rgb=torch.as_tensor(material[start:stop], dtype=torch.float32, device=env.device),
            )
            output.append(image)
        return torch.cat(output)


def close(env):
    for renderer, _ in env.__dict__.pop("_rocket_pixel_gpu", {}).values():
        renderer.close()
