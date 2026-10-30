"""Independent NumPy world-space oracle for sampled optical rays.

No runtime modules, exported assets, GPU kernels or local-pose helpers are used.
A recursive world-space BVH and vectorized linear solves implement intersections.
The reference may cache static source paths, after checking source geometry on
all frames; it never uses a candidate renderer to construct expected geometry.
"""

from typing import TypedDict

import numpy as np

from .object_permanence_ray_reference import plane


class OpticalPolicy(TypedDict):
    samples: int
    seed: int
    max_events: int
    ray_start: float


CONTRACT = "stable-envs/sampled-optical-rays-v1"
MATERIAL = "stable-envs/constant-dielectric-surfaces-v1"
POLICY: OpticalPolicy = {"samples": 8, "seed": 0, "max_events": 64, "ray_start": 1e-9}


def uniform(sample, event):
    h = (sample.astype(np.uint64) * np.uint64(747796405) + np.uint64(event) * np.uint64(2891336453)) & np.uint64(
        0xFFFFFFFF
    )
    h = ((h ^ (h >> np.uint64(16))) * np.uint64(2246822519)) & np.uint64(0xFFFFFFFF)
    h = ((h ^ (h >> np.uint64(13))) * np.uint64(3266489917)) & np.uint64(0xFFFFFFFF)
    return ((h ^ (h >> np.uint64(16))).astype(np.float64) + 0.5) / 4294967296


class WorldTree:
    """World-coordinate recursive triangle BVH; original source order breaks ties."""

    def __init__(self, meshes, names):
        triangles = []
        labels = []
        for g, name in enumerate(names):
            mesh = meshes.get(name, np.empty((0, 3, 3)))
            triangles.extend(mesh)
            labels.extend([g] * len(mesh))
        self.tri = np.array(triangles, dtype=np.float64).reshape(-1, 3, 3)
        self.labels = np.array(labels)
        self.planes = np.array([plane(t) for t in self.tri])
        self.normals = np.cross(self.tri[:, 1] - self.tri[:, 0], self.tri[:, 2] - self.tri[:, 0])
        lengths = np.linalg.norm(self.normals, axis=-1)
        self.normals = np.divide(
            self.normals, lengths[:, None], out=np.zeros_like(self.normals), where=lengths[:, None] > 0
        )
        centres = self.tri.mean(1)

        def build(faces):
            lo, hi = self.tri[faces].min((0, 1)), self.tri[faces].max((0, 1))
            if len(faces) <= 12:
                return lo, hi, faces, None
            axis = np.argmax(hi - lo)
            faces = faces[np.argsort(centres[faces, axis], kind="stable")]
            middle = len(faces) // 2
            return lo, hi, build(faces[:middle]), build(faces[middle:])

        self.root = build(np.arange(len(self.tri))) if len(self.tri) else None

    def hit(self, origin, direction):
        distance = np.full(len(origin), np.inf)
        winner = np.full(len(origin), -1, dtype=np.int64)
        if self.root is None:
            return distance, winner, np.zeros_like(origin), np.full(len(origin), -1, dtype=np.int64)

        def visit(node, indices):
            if not len(indices):
                return
            lo, hi, left, right = node
            o, d = origin[indices], direction[indices]
            with np.errstate(divide="ignore", invalid="ignore"):
                a, b = (lo - o) / d, (hi - o) / d
            low, high = np.minimum(a, b), np.maximum(a, b)
            zero = d == 0
            low = np.where(zero, np.where((o >= lo) & (o <= hi), -np.inf, np.inf), low)
            high = np.where(zero, np.inf, high)
            valid = np.maximum(low.max(1), 1e-9) <= np.minimum(high.min(1), distance[indices] + 1e-12)
            indices = indices[valid]
            if right is not None:
                visit(left, indices)
                visit(right, indices)
                return
            if not len(indices):
                return
            # A distinct plane/barycentric construction, not the CUDA Moller code.
            a = self.tri[left, 0]
            e = self.tri[left, 1] - a
            f = self.tri[left, 2] - a
            o = origin[indices]
            d = direction[indices]
            with np.errstate(divide="ignore", invalid="ignore"):
                coeff = self.planes[left]
                t = -(
                    o[:, None, 0] * coeff[None, :, 0]
                    + o[:, None, 1] * coeff[None, :, 1]
                    + o[:, None, 2] * coeff[None, :, 2]
                    + coeff[None, :, 3]
                ) / (
                    d[:, None, 0] * coeff[None, :, 0]
                    + d[:, None, 1] * coeff[None, :, 1]
                    + d[:, None, 2] * coeff[None, :, 2]
                )
                p = o[:, None] + t[..., None] * d[:, None] - a
                ee = (e * e).sum(-1)
                ff = (f * f).sum(-1)
                ef = (e * f).sum(-1)
                pe = (p * e).sum(-1)
                pf = (p * f).sum(-1)
                den = ee * ff - ef * ef
                u = (pe * ff - pf * ef) / den
                v = (pf * ee - pe * ef) / den
            good = (t > 1e-9) & (u >= 0) & (v >= 0) & (u + v <= 1) & (den != 0)
            t = np.where(good, t, np.inf)
            # Explicit lexicographic (distance, source object, source face).
            best = t.min(1)
            face = np.min(np.where(t == best[:, None], left[None], len(self.tri)), axis=1)
            change = np.isfinite(best) & (
                (best < distance[indices]) | ((best == distance[indices]) & (face < winner[indices]))
            )
            distance[indices[change]] = best[change]
            winner[indices[change]] = face[change]

        visit(self.root, np.arange(len(origin)))
        hit = winner >= 0
        normals = np.zeros_like(origin)
        labels = np.full(len(origin), -1, dtype=np.int64)
        normals[hit] = self.normals[winner[hit]]
        labels[hit] = self.labels[winner[hit]]
        return distance, winner, normals, labels


def draw(meshes, materials, camera, height, width, *, segments=None, save=False):
    """Trace fixed counter samples; return geometry fields and optional static paths."""
    names = sorted(materials)
    count = len(names)
    pixels = height * width
    samples = POLICY["samples"]
    tree = WorldTree(meshes, names)
    material = np.array([[materials[n][k] for k in ("alpha", "transmission", "ior")] for n in names])
    indices = {m["ior"] for m in materials.values() if m["transmission"] > 0}
    if len(indices) != 1:
        raise ValueError("reference optical v1 requires one common dielectric IOR")
    ior = indices.pop()
    yy, xx = np.mgrid[:height, :width]
    p = np.array(camera["projection"])
    world = np.array(camera["world"])
    local = np.stack(
        (
            (2 * (xx + 0.5) / width - 1 + p[0, 2]) / p[0, 0],
            (1 - 2 * (yy + 0.5) / height + p[1, 2]) / p[1, 1],
            -np.ones_like(xx),
        ),
        -1,
    ).reshape(-1, 3)
    rays = local @ world[:3, :3].T
    factor = 1 / np.linalg.norm(rays, axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    direction = np.repeat(rays, samples, axis=0)
    origin = np.broadcast_to(world[:3, 3], direction.shape).copy()
    sample = np.arange(len(direction))
    travelled = np.zeros(len(direction))
    membership = np.zeros(len(direction), np.int64)
    depths = np.zeros(len(direction))
    ids = np.zeros(len(direction), np.int16)
    paths = np.full((count, pixels), np.inf)
    saved = []
    for event in range(POLICY["max_events"]):
        if not len(sample):
            break
        t, face, normal, label = tree.hit(origin, direction)
        if segments is not None:
            old = segments[event]
            index = np.searchsorted(old[0], sample)
            st, sn, sl = (old[k][index] for k in (1, 2, 3))
            # Source object label is sufficient when equal-depth surfaces belong
            # to different source partitions; same-object normals stay unchanged.
            use = (st < t) | ((st == t) & (sl < label))
            t = np.where(use, st, t)
            normal = np.where(use[:, None], sn, normal)
            label = np.where(use, sl, label)
        if save:
            saved.append((sample.copy(), t.copy(), normal.copy(), label.copy()))
        hit = label >= 0
        sample, origin, direction, travelled, membership, t, normal, label = (
            a[hit] for a in (sample, origin, direction, travelled, membership, t, normal, label)
        )
        if not len(sample):
            break
        alpha, tau, _ = material[label].T
        visible = alpha > 0
        np.minimum.at(paths, (label[visible], sample[visible] // samples), (travelled + t)[visible])
        if event == 0:
            depths[sample] = t * factor[sample // samples]
            ids[sample] = (label + 1).astype(np.int16)
        choice = uniform(sample, 2 * event)
        null = choice < 1 - alpha
        keep = null | (choice >= 1 - alpha + alpha * (1 - tau))
        sample, origin, direction, travelled, membership, t, normal, label, null = (
            a[keep] for a in (sample, origin, direction, travelled, membership, t, normal, label, null)
        )
        if not len(sample):
            break
        next_membership = membership ^ np.where(material[label, 1] > 0, np.left_shift(np.int64(1), label), 0)
        n1 = np.where(membership != 0, ior, 1.0)
        n2 = np.where(next_membership != 0, ior, 1.0)
        normal = np.where(((direction * normal).sum(1) > 0)[:, None], -normal, normal)
        cosine = np.clip(-(direction * normal).sum(1), 0, 1)
        ratio = n1 / n2
        radical = 1 - ratio**2 * (1 - cosine**2)
        ct = np.sqrt(np.maximum(radical, 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            fresnel = (
                ((n1 * cosine - n2 * ct) / (n1 * cosine + n2 * ct)) ** 2
                + ((n1 * ct - n2 * cosine) / (n1 * ct + n2 * cosine)) ** 2
            ) / 2
        fresnel = np.where(n1 == n2, 0, np.where(radical <= 0, 1, fresnel))
        reflect = uniform(sample, 2 * event + 1) < fresnel
        reflected = direction + 2 * cosine[:, None] * normal
        refracted = ratio[:, None] * direction + (ratio * cosine - ct)[:, None] * normal
        bent = np.where(reflect[:, None], reflected, refracted)
        bent /= np.linalg.norm(bent, axis=1, keepdims=True)
        origin = origin + t[:, None] * direction
        direction = np.where(null[:, None], direction, bent)
        travelled += t
        membership = np.where(null | ~reflect, next_membership, membership)
    else:
        if len(sample):
            raise ValueError("independent source paths exceed optical event capacity")
    result = {
        "depth": depths.reshape(pixels, samples)[:, 0].astype(np.float32).reshape(height, width),
        "ids": ids.reshape(pixels, samples)[:, 0].reshape(height, width),
        "path_depth": np.where(np.isfinite(paths), paths, 0).astype(np.float32).reshape(count, height, width),
        "masks": np.isfinite(paths).reshape(count, height, width),
    }
    return result, saved
