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
from tensordict import TensorDict

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
#: ``{key: (shape, sliced)}`` for each full State leaf ``project`` reads. All are float64. The frame reads the
#: leading ``shape`` part of a leaf after the batch axis. A ``sliced`` leaf can be larger on each axis; the
#: other leaves have exactly this shape.
STATE_READS = {
    **{("current", "physics", f"body_{i}", "position"): ((3,), False) for i in range(3)},
    **{("current", "physics", f"body_{i}", "quaternion"): ((4,), False) for i in range(3)},
    ("current", "diagnostic", "cached_links"): ((1, 7), True),
    ("render", "rgba"): ((6, 4), False),
}
#: Fewest CPU frames one ``render`` draws in render workers when the ``render_worker_min_rows`` knob is not set.
MIN_WORKER_ROWS = 2
#: CPU renderers. ``"pybullet"`` (also ``"tiny"``) draws each frame with PyBullet's TinyRenderer; ``"numba"`` draws
#: all frames in one call of the Numba port in ``tiny_raster``. ``"auto"`` selects ``CPU_AUTO``.
CPU_BACKENDS = ("auto", "pybullet", "tiny", "numba")
CPU_AUTO = "numba"
#: CUDA renderers; ``"auto"`` is ``"nvdiffrast"``.
CUDA_BACKENDS = ("auto", "nvdiffrast")


def check_backend(backend, device):
    """Raise ValueError unless ``backend`` names a Rocket renderer on ``device``."""
    kind = torch.device(device).type
    allowed = CPU_BACKENDS if kind == "cpu" else CUDA_BACKENDS
    if backend not in allowed:
        raise ValueError(f"Rocket rendering on {kind} uses a backend in {allowed}, got {backend!r}")


def cpu_backend(backend):
    """``"pybullet"`` or ``"numba"`` for a CPU ``backend`` name."""
    check_backend(backend, "cpu")
    backend = CPU_AUTO if backend == "auto" else backend
    return "pybullet" if backend == "tiny" else backend


def host_leaves(state, reads, batch):
    """Check the parts of the full State leaves in ``reads`` and return them on the host, with one transfer.

    It checks presence, shape, dtype and finiteness of the part a frame reads, and that colours lie in [0, 1].
    The replay, history and RNG checks of a restore do not apply to a frame; ``set_state`` makes them.
    """
    if not isinstance(state, TensorDict) or state.batch_size != torch.Size([batch]):
        raise ValueError("Rocket full State requires batch [num_envs]")
    parts = {}
    for key, (shape, sliced) in reads.items():
        value = state.get(key, None)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Rocket full State lacks the render leaf {key}")
        if (
            value.dtype != torch.float64
            or value.ndim != 1 + len(shape)
            or any(n < m if sliced else n != m for n, m in zip(value.shape[1:], shape, strict=True))
        ):
            raise ValueError(f"Invalid Rocket State leaf: {key}")
        parts[key] = value.detach()[(slice(None), *(slice(0, n) for n in shape))] if sliced else value.detach()
    if any(value.device.type != "cpu" for value in parts.values()):
        flat = torch.cat([value.reshape(-1) for value in parts.values()]).cpu()
        offset = 0
        for key, value in parts.items():
            parts[key] = flat[offset : offset + value.numel()].view(value.shape)
            offset += value.numel()
    for key, value in parts.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Invalid Rocket State leaf: {key}")
    rgba = parts.get(("render", "rgba"))
    if rgba is not None and ((rgba < 0) | (rgba > 1)).any():
        raise ValueError("Rocket RenderState rgba must lie in [0, 1]")
    return parts


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


def pose_row_fields(rows, style):
    """``(body, links, rgba)`` of ``(B, 13)`` observation rows and ``(B, 16)`` renderer style rows.

    The rocket is at the row's pose and its camera link at ``legacy.lagged_pose``'s previous pose; the colours
    are ``legacy.material_table``'s with the alphas the reference bridge paints.
    """
    rows = np.asarray(rows, np.float64).reshape(-1, geometry.PyFlytRocketLanding.state_dim)
    position, quaternion, previous, previous_quaternion, _, _ = geometry.lagged_pose(rows)
    batch = len(rows)
    body = np.zeros((batch, 3, 7))
    body[..., 6] = 1.0
    body[:, 1, :3], body[:, 1, 3:] = position, quaternion
    body[:, 2, :3] = geometry.PAD_POS
    links = np.concatenate([previous, previous_quaternion], -1)
    material = geometry.material_table(style)
    rgba = np.ones((batch, 6, 4))
    for row, geom in (
        (BODY_COLOR, BODY_GEOM),
        (FIN_COLOR, FIN_GEOMS.start),
        (LEG_COLOR, LEG_GEOMS.start),
        (BOOSTER_COLOR, BOOSTER_GEOM),
        (PAD_COLOR, PAD_GEOM),
        (FLAME_COLOR, FLAME_GEOM),
    ):
        rgba[:, row, :3] = material[:, geom]
    offset, _ = geometry.style_layout()["flame.ignited"]
    ignited = np.asarray(style, np.float64).reshape(batch, -1)[:, offset] > 0.5
    rgba[:, FLAME_COLOR, 3] = np.where(ignited, geometry.IGNITION_RGBA[3], geometry.FLAME_RGBA[3])
    return body, links, rgba


def camera_view(link, bullet):
    """PyBullet's view matrix for camera link pose ``link``, as its tuple of 16 column-major floats."""
    rotation = np.array(bullet.getMatrixFromQuaternion(link[3:7])).reshape(3, 3)
    eye = link[:3] + rotation @ geometry.CAM_OFFSET
    angles = np.array(bullet.getEulerFromQuaternion(link[3:7]))
    angles[1] += math.radians(geometry.CAM_PITCH_DEG)
    up = np.array(bullet.getMatrixFromQuaternion(bullet.getQuaternionFromEuler(angles))).reshape(3, 3)[:, 2]
    return bullet.computeViewMatrix(eye, link[:3], up)


def camera_projection(bullet):
    return bullet.computeProjectionMatrixFOV(geometry.CAM_FOV, 1.0, geometry.CAM_NEAR, geometry.CAM_FAR)


def camera_matrices(link, bullet):
    return camera_view(link, bullet), camera_projection(bullet)


def gl_matrix(matrix):
    """A PyBullet column-major matrix tuple as a float32 ``(4, 4)`` row-major array."""
    return np.asarray(matrix, np.float32).reshape(4, 4).T


def frame_views(links, bullet):
    """Float32 ``(B, 4, 4)`` view matrices of ``(B, 7)`` camera link poses."""
    views = np.empty((len(links), 4, 4), np.float32)
    for i, link in enumerate(links):
        views[i] = gl_matrix(camera_view(link, bullet))
    return views


def _mm(a, b):
    out = np.zeros(np.broadcast_shapes(a.shape[:-2], b.shape[:-2]) + (4, 4), np.float32)
    for k in (3, 2, 1, 0):
        out += a[..., :, k : k + 1] * b[..., k : k + 1, :]
    return out


class _Scene(ND.Scene):
    def clip(self, xpos, xmat, view, proj):
        """Clip positions as TinyRenderer's ``mvp * (x, y, z, 1)``: ``((m3 + m2 z) + m1 y) + m0 x`` in float32."""
        _, normals, world = super().clip(xpos, xmat, view, proj)
        m = self.source_mvp[:, self.vgeom]
        x, y, z = (self.vert[:, i].reshape(1, -1, 1) for i in range(3))
        positions = ((m[..., 3] + m[..., 2] * z) + m[..., 1] * y) + m[..., 0] * x
        return positions.contiguous(), normals, world


def tiny_texture(tex, clip, face, uv, textured, tid, resolution, fill=1):
    """Sample ``tex`` at TinyRenderer's texel for each textured pixel; other pixels get ``fill``.

    TinyRenderer takes double barycentrics of the unclipped triangle from float32
    screen points at integer pixel positions, sums float32 UVs in reverse vertex
    order, wraps them with modf and truncates them to a texel.
    """
    out = torch.full(tid.shape + (3,), fill, device=tex.device, dtype=tex.dtype)
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

    def _texels(self, S, rast):
        """TinyRenderer's ``diffuse(uv)`` per pixel as float32 bytes: the ground texel, else 255."""
        texture = self.__dict__.get("_texture8")
        if texture is None:
            from .tiny_raster import scene

            texture = torch.as_tensor(scene().texture, dtype=torch.float32, device=self.device)
            self._texture8 = texture
        textured = (S.texid >= 0)[S.fgeom]
        tid = rast[..., 3].long()
        return tiny_texture(texture, self._clip, S.face.long(), S.uv[:, :2], textured, tid, rast.shape[2], 255.0)

    def render(self, xpos, xmat, cam, out=None, mat_rgb=None):
        """Draw with TinyRenderer's colour law on the interpolated normals.

        ``c = int(texel * rgba)`` per channel, then ``int(A*c + (D*diffuse + S*specular)*c)`` with float32 operations
        in Shader::fragment's order. Supersampled frames average the uint8 samples to the nearest integer.
        """
        import nvdiffrast.torch as dr

        self._check_open("render")
        S = self.scene
        R = self.res * self.ss
        B = int(xpos.shape[0])
        view_np, proj_np = cam.matrices(aspect=1.0)
        view = torch.as_tensor(view_np, dtype=torch.float32, device=self.device)
        proj = torch.as_tensor(proj_np, dtype=torch.float32, device=self.device)
        pos, nrm, _ = S.clip(xpos, xmat, view, proj)
        rast, _ = self._rasterize(pos, R)
        normal, _ = dr.interpolate(nrm, rast, S.face)
        tid = rast[..., 3].long()
        hit = tid > 0
        gid = torch.where(hit, S.fgeom[(tid - 1).clamp_min(0)], torch.zeros_like(tid))
        rgba = ND.gather_material(S.mat_rgb if mat_rgb is None else mat_rgb, gid)
        c8 = torch.floor(self._texels(S, rast) * rgba)
        del rgba
        k = tiny_light(normal, self.device)
        del normal
        f32 = lambda v: torch.tensor(v, dtype=torch.float32, device=self.device)  # noqa: E731
        value = f32(0.6) * c8 + k * c8
        del c8, k
        value = torch.where(torch.isnan(value), f32(0.0), torch.floor(value)).clamp_max_(255.0)
        img = torch.where(hit[..., None], value, f32(255.0))
        del value
        if self.ss > 1:
            img = img.view(B, self.res, self.ss, self.res, self.ss, 3).sum((2, 4)) / (self.ss * self.ss)
            img = torch.round(img)
        img = img.to(torch.uint8).flip(1)  # nvdiffrast row 0 is the bottom
        if out is not None:
            out.view(B, self.res, self.res, 3).copy_(img)
            return out
        return img


def tiny_light(normal, device):
    """``D*diffuse + S*specular`` of TinyRenderer's Shader::fragment for interpolated ``(..., 3)`` normals."""
    from .tiny_raster import scene

    f32 = lambda v: torch.tensor(float(v), dtype=torch.float32, device=device)  # noqa: E731
    lx, ly, lz = (f32(v) for v in scene().light)
    nx, ny, nz = normal[..., 0:1], normal[..., 1:2], normal[..., 2:3]
    inv = f32(1.0) / torch.sqrt((nx * nx + ny * ny) + nz * nz)
    nx, ny, nz = nx * inv, ny * inv, nz * inv
    d = (nz * lz + ny * ly) + nx * lx
    twice = d * f32(2.0)
    rx, ry, rz = nx * twice - lx, ny * twice - ly, nz * twice - lz
    rz = rz * (f32(1.0) / torch.sqrt((rx * rx + ry * ry) + rz * rz))
    del rx, ry, nx, ny, nz, twice
    r = torch.where(rz > 0, rz, f32(0.0))
    diffuse = torch.where(d < 0, f32(0.0), d)
    return f32(0.35) * diffuse + f32(0.05) * (r * r)


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


def draw_cpu(env, body, links, colors, size, ss):
    """Draw with TinyRenderer: in render workers on Linux, else in this process.

    A frame depends only on its row, so the split does not change the frames.
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
        return env._simulator.draw(body, links, colors, size, ss)

    with workers.lock:
        return workers.render(
            len(body), (size, size, 3), lambda rows: (body[rows], links[rows], colors[rows], size, ss)
        )


def render_cpu(env, body, links, colors, size, ss, backend=None):
    """Draw host body poses, camera link poses and colours on CPU with ``backend`` (default: the env's)."""
    env._check_open("render")
    backend = cpu_backend(env.backend if backend is None else backend)
    with env.__dict__.setdefault("_rocket_pixel_lock", threading.RLock()):
        if backend == "numba":
            from . import tiny_raster

            return torch.from_numpy(tiny_raster.draw(body, links, colors, size, ss))
        return torch.from_numpy(draw_cpu(env, body, links, colors, size, ss))


def render(env, state, size, ss, chunk_size=None, backend=None):
    """Draw a RenderState with explicit poses."""
    if chunk_size is not None and (isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0):
        raise ValueError("chunk_size must be a positive integer")
    env._check_open("render")
    host = {k: state["renderer", k].detach().cpu().numpy() for k in FIELDS}
    body, links, colors = host["source_pose"], host["camera_pose"], host["rgba"]
    if env.device.type == "cpu":
        return render_cpu(env, body, links, colors, size, ss, backend)
    if env.device.type != "cuda":
        raise ValueError("Rocket rendering supports CPU TinyRenderer or CUDA nvdiffrast")
    return draw_gpu(env, body, links, colors, size, ss, chunk_size)


def draw_gpu(env, body, links, colors, size, ss, chunk_size=None, views=None):
    """Draw host body poses, camera link poses and colours on CUDA; ``views`` are their ``frame_views`` if known.

    The clip positions use TinyRenderer's float32 model, view and projection matrices (``tiny_raster``).
    """
    from . import tiny_raster

    lock = env.__dict__.setdefault("_rocket_pixel_lock", threading.RLock())
    with lock:
        batch = len(body)
        tuning.on_render("rocket-nvdiff", size, batch, env)
        renderer, per_world = _gpu_context(env, size, ss)
        limit = geometry.nvd_worlds(per_world, env.device, size, batch)
        limit = min(limit, chunk_size) if chunk_size is not None else limit
        positions = np.zeros((batch, geometry.N_GEOMS, 3), np.float64)
        quaternions = np.zeros((batch, geometry.N_GEOMS, 4), np.float64)
        quaternions[..., 3] = 1
        for source, geom in enumerate((GROUND_GEOM, BODY_GEOM, PAD_GEOM)):
            positions[:, geom], quaternions[:, geom] = body[:, source, :3], body[:, source, 3:]
        local_pos, local_quat = geometry.link_local_frames()
        rotation = geometry.quat_to_mat(quaternions[:, BODY_GEOM])
        for j in range(1, len(local_pos)):
            positions[:, BODY_GEOM + j] = positions[:, BODY_GEOM] + np.einsum("bij,j->bi", rotation, local_pos[j])
            quaternions[:, BODY_GEOM + j] = geometry.quat_mul(quaternions[:, BODY_GEOM], local_quat[j : j + 1])
        rotations = geometry.quat_to_mat(quaternions)
        ground = positions[:, GROUND_GEOM].copy()
        geometry.cull_ground(positions, rotations, geometry.camera_pose(links[:, :3], links[:, 3:])[0])
        model = tiny_raster.model_matrices(body)
        culled = (positions[:, GROUND_GEOM] != ground).any(-1)
        model[culled, GROUND_GEOM, :3, 3] = positions[culled, GROUND_GEOM]
        views = tiny_raster.view_matrices(links) if views is None else views
        mvp = _mm(tiny_raster.scene().projection, _mm(views[:, None], model))
        origin = positions[:, BODY_GEOM].copy()
        positions -= origin[:, None]
        eye, camera = geometry.camera_pose(links[:, :3] - origin, links[:, 3:])
        material = np.ones((batch, geometry.N_GEOMS, 3), np.float64)
        material[:, BODY_GEOM] = colors[:, BODY_COLOR, :3]
        material[:, BOOSTER_GEOM] = colors[:, BOOSTER_COLOR, :3]
        material[:, FIN_GEOMS] = colors[:, FIN_COLOR : FIN_COLOR + 1, :3]
        material[:, LEG_GEOMS] = colors[:, LEG_COLOR : LEG_COLOR + 1, :3]
        material[:, FLAME_GEOM] = colors[:, FLAME_COLOR, :3]
        material[:, PAD_GEOM] = colors[:, PAD_COLOR, :3]
        output = []
        for start in range(0, batch, limit):
            stop = start + limit
            renderer.scene.source_mvp = torch.as_tensor(mvp[start:stop], device=env.device)
            image = renderer.render(
                torch.as_tensor(positions[start:stop], dtype=torch.float32, device=env.device),
                torch.as_tensor(rotations[start:stop], dtype=torch.float32, device=env.device),
                ND.Camera(eye[start:stop], camera[start:stop], geometry.CAM_FOV, geometry.CAM_NEAR, geometry.CAM_FAR),
                mat_rgb=torch.as_tensor(material[start:stop], dtype=torch.float32, device=env.device),
            )
            output.append(image)
        return output[0] if len(output) == 1 else torch.cat(output)


def close(env):
    for renderer, _ in env.__dict__.pop("_rocket_pixel_gpu", {}).values():
        renderer.close()
