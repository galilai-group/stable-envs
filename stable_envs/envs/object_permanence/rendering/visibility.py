"""Versioned opaque visibility contract; legacy rasterization stays opt-in separate."""

from fractions import Fraction

import numpy as np

CONTRACT = "stable-envs/opaque-pixel-rays-v1"
CONTRACT_V2 = "stable-envs/opaque-pixel-rays-v2"
CONTRACTS = (CONTRACT, CONTRACT_V2)
SPEC = {
    "id": CONTRACT,
    "camera": "fixed perspective; recorded OpenGL projection and world matrix in binary64",
    "samples": "top-left image; pixel centers (x+0.5,y+0.5); no jitter or multisampling",
    "ray": "world @ ((ndc_x+P02)/P00,(ndc_y+P12)/P11,-1,0); no normalization",
    "clipping": "near <= camera-axis ray parameter <= far; two-sided closed triangles",
    "depth": "positive camera-axis ray parameter; float32 output; zero for no hit",
    "planes": "binary64 normalized plane coefficients; largest absolute normal component becomes +1",
    "ties": "exactly equal binary64 plane-ray depth: lowest positive semantic object ID",
    "within_object": "lowest original triangle index; no depth or silhouette epsilon",
    "scope": "opaque rigid triangle meshes; fixed perspective camera; no depth-plane crossing",
}
SPEC_V2 = {
    **SPEC,
    "id": CONTRACT_V2,
    "camera": "fixed perspective or orthographic; explicit orthographic projection_type; binary64 matrices",
    "orthographic_ray": "origin=world@((ndc_x-P03)/P00,(ndc_y-P13)/P11,0,1); direction=world@(0,0,-1,0)",
    "scope": "opaque rigid meshes; orthographic interval clipping; perspective retains no depth-plane crossing",
}


def local_planes(vertices, faces):
    """Canonical local planes from represented vertices, independent of triangulation.

    Rational construction avoids independently rounded cross products on the same
    plane. Conversion to binary64 happens only after coefficient normalization.
    Degenerate triangles receive a zero plane and cannot cover any sample.
    """
    result = []
    for face in faces:
        a, b, c = [[Fraction(float(x)) for x in vertices[i]] for i in face]
        u, v = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
        n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        k = max(range(3), key=lambda i: abs(n[i]))
        result.append([float(x / n[k]) for x in [*n, -sum(n[i] * a[i] for i in range(3))]] if n[k] else [0] * 4)
    return np.asarray(result, np.float64)


def validate_camera(camera, *, contract=CONTRACT):
    """Reject projection conventions not represented by this contract version."""
    p = np.asarray(camera["projection"], np.float64)
    world = np.asarray(camera["world"], np.float64)
    if p.shape != (4, 4) or world.shape != (4, 4) or not np.isfinite([p, world]).all():
        raise ValueError("contract requires finite camera matrices")
    if contract not in CONTRACTS:
        raise ValueError("unsupported visibility contract")
    kind = camera.get("projection_type", "perspective")
    permitted = np.zeros((4, 4), bool)
    if kind == "orthographic" and contract == CONTRACT_V2:
        permitted[0, [0, 3]], permitted[1, [1, 3]] = True, True
        permitted[2, [2, 3]], permitted[3, 3] = True, True
        valid = p[3, 3] == 1 and p[2, 2] < 0
    elif kind == "perspective":
        permitted[0, [0, 2]], permitted[1, [1, 2]] = True, True
        permitted[2, [2, 3]], permitted[3, 2] = True, True
        valid = p[3, 2] == -1
    else:
        raise ValueError("contract does not support this camera projection type")
    if not valid or np.any(p[~permitted] != 0) or p[0, 0] <= 0 or p[1, 1] <= 0:
        raise ValueError("contract requires a standard camera projection")
    if (
        not np.isfinite([camera["near"], camera["far"]]).all()
        or camera["near"] <= 0
        or camera["far"] <= camera["near"]
    ):
        raise ValueError("contract requires 0 < near < far")
    if not np.array_equal(world[3], [0, 0, 0, 1]) or abs(np.linalg.det(world[:3, :3])) < 1e-12:
        raise ValueError("contract requires an invertible affine camera")
