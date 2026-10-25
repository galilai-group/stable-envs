"""Fixed-order visibility transforms without per-vertex/per-face matrix gathers.

Each thread reads its object's tiny transform directly, retaining float64 world
vertices/canonical planes and camera multiplication order. Shading attributes
keep the existing sparse float32 arithmetic independently of visibility.
"""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _dot4(M, base, row: tl.constexpr, x, y, z, w):
    return (
        x * tl.load(M + base + 4 * row)
        + y * tl.load(M + base + 4 * row + 1)
        + z * tl.load(M + base + 4 * row + 2)
        + w * tl.load(M + base + 4 * row + 3)
    )


@tr.jit
def _vertices(
    Vert,
    Geom,
    Position,
    Rotation,
    View,
    Proj,
    World,
    Clip,
    V: tl.constexpr,
    G: tl.constexpr,
    VIEWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.program_id(1)
    valid = q < V
    g = tl.load(Geom + q, valid, 0)
    r = (batch * G + g) * 9
    p = (batch * G + g) * 3
    x = tl.load(Vert + q * 3, valid, 0).to(tl.float64)
    y = tl.load(Vert + q * 3 + 1, valid, 0).to(tl.float64)
    z = tl.load(Vert + q * 3 + 2, valid, 0).to(tl.float64)
    wx = (
        x * tl.load(Rotation + r)
        + y * tl.load(Rotation + r + 1)
        + z * tl.load(Rotation + r + 2)
        + tl.load(Position + p)
    )
    wy = (
        x * tl.load(Rotation + r + 3)
        + y * tl.load(Rotation + r + 4)
        + z * tl.load(Rotation + r + 5)
        + tl.load(Position + p + 1)
    )
    wz = (
        x * tl.load(Rotation + r + 6)
        + y * tl.load(Rotation + r + 7)
        + z * tl.load(Rotation + r + 8)
        + tl.load(Position + p + 2)
    )
    base = batch * 16 if VIEWS != 1 else 0
    vx = _dot4(View, base, 0, wx, wy, wz, 1.0)
    vy = _dot4(View, base, 1, wx, wy, wz, 1.0)
    vz = _dot4(View, base, 2, wx, wy, wz, 1.0)
    vw = _dot4(View, base, 3, wx, wy, wz, 1.0)
    tl.store(World + (batch * V + q) * 3, wx, valid)
    tl.store(World + (batch * V + q) * 3 + 1, wy, valid)
    tl.store(World + (batch * V + q) * 3 + 2, wz, valid)
    for c in tl.static_range(4):
        tl.store(Clip + (batch * V + q) * 4 + c, _dot4(Proj, 0, c, vx, vy, vz, vw), valid)


@tr.jit
def _planes(Local, Geom, Position, Inverse, Out, F: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.program_id(1)
    valid = q < F
    g = tl.load(Geom + q, valid, 0)
    r = (batch * G + g) * 9
    p = (batch * G + g) * 3
    x = tl.load(Local + q * 4, valid, 0)
    y = tl.load(Local + q * 4 + 1, valid, 0)
    z = tl.load(Local + q * 4 + 2, valid, 0)
    nx = x * tl.load(Inverse + r) + y * tl.load(Inverse + r + 3) + z * tl.load(Inverse + r + 6)
    ny = x * tl.load(Inverse + r + 1) + y * tl.load(Inverse + r + 4) + z * tl.load(Inverse + r + 7)
    nz = x * tl.load(Inverse + r + 2) + y * tl.load(Inverse + r + 5) + z * tl.load(Inverse + r + 8)
    # Match CUDA's three-element Torch reduction: lanes 0+2, then lane 1.
    dot = (nx * tl.load(Position + p) + nz * tl.load(Position + p + 2)) + ny * tl.load(Position + p + 1)
    offset = tl.load(Local + q * 4 + 3, valid, 0) - dot
    scale = tl.where(
        (tl.abs(nx) >= tl.abs(ny)) & (tl.abs(nx) >= tl.abs(nz)), nx, tl.where(tl.abs(ny) >= tl.abs(nz), ny, nz)
    )
    base = (batch * F + q) * 4
    tl.store(Out + base, tl.where(scale != 0, nx / scale, 0), valid)
    tl.store(Out + base + 1, tl.where(scale != 0, ny / scale, 0), valid)
    tl.store(Out + base + 2, tl.where(scale != 0, nz / scale, 0), valid)
    tl.store(Out + base + 3, tl.where(scale != 0, offset / scale, 0), valid)


def transform(scene, positions, rotations, inverse, local_planes, view, projection):
    """Return exact world vertices, canonical planes and clip-space vertices."""
    b, g, _ = positions.shape
    v, f = scene.V, len(scene.face)
    positions, rotations, inverse = (x.double().contiguous() for x in (positions, rotations, inverse))
    view = torch.as_tensor(view, dtype=torch.float64, device=positions.device).reshape(-1, 4, 4).contiguous()
    projection = torch.as_tensor(projection, dtype=torch.float64, device=positions.device).contiguous()
    if len(view) not in (1, b) or projection.shape != (4, 4):
        raise ValueError("visibility transforms need shared projection and shared/per-row view")
    vertices = torch.empty((b, v, 3), device=positions.device, dtype=torch.float64)
    clip = torch.empty((b, v, 4), device=positions.device, dtype=torch.float64)
    planes = torch.empty((b, f, 4), device=positions.device, dtype=torch.float64)
    with torch.cuda.device(positions.device):
        _vertices[(tr.cdiv(v, 128), b)](
            scene.vert,
            scene.vgeom,
            positions,
            rotations,
            view,
            projection,
            vertices,
            clip,
            v,
            g,
            len(view),
            128,
            enable_fp_fusion=False,
        )
        _planes[(tr.cdiv(f, 128), b)](
            local_planes,
            scene.fgeom,
            positions,
            inverse,
            planes,
            f,
            g,
            128,
            enable_fp_fusion=False,
        )
    return vertices, planes, clip


def shading_attributes(scene, positions, rotations):
    """Keep nvdiff's original sparse float32 shading arithmetic, omit unused MVP."""
    positions, rotations = positions.float(), rotations.float()
    b, g, _ = positions.shape
    matrix = rotations.permute(1, 3, 0, 2).reshape(3 * g, 3 * b)
    normal = torch.sparse.mm(scene.S3, matrix).reshape(scene.V, b, 3).permute(1, 0, 2).contiguous()
    world = torch.sparse.mm(scene.S3p, matrix).reshape(scene.V, b, 3).permute(1, 0, 2) + positions[:, scene.vgeom]
    return normal, world.contiguous()
