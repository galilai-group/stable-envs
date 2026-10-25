"""Versioned rigid-scene recordings: shared local meshes and sampled world poses.

Numeric I/O uses NumPy. Loading does not construct an engine or renderer.
Pose files stay memory mapped; selecting frames returns an owned CPU array.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from copy import deepcopy
from pathlib import Path

import numpy as np

from ...render._resources import Closeable

SCHEMA = "stable-envs/object-permanence-rigid-v1"
SCHEMA_V2 = "stable-envs/object-permanence-rigid-v2"
SCHEMA_V3 = "stable-envs/object-permanence-rigid-v3"
SCHEMA_V4 = "stable-envs/object-permanence-rigid-v4"
SCALE_CONVENTION = "stable-envs/asset-relative-positive-diagonal-v1"
CONVENTIONS = {"world": "right-handed-z-up", "quaternion": "wxyz", "units": "source-scene-units"}


def array_digest(arrays):
    """Hash names, shapes, dtypes and values, independently of ZIP compression."""
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        stream = io.BytesIO()
        np.save(stream, array, allow_pickle=False)
        digest.update(name.encode() + b"\0" + stream.getvalue())
    return digest.hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _poses(value, shape):
    if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"poses must be finite float32 {shape}")
    if not np.allclose(np.linalg.norm(value[..., 3:], axis=-1), 1, atol=2e-6, rtol=0):
        raise ValueError("pose quaternions must have unit length")


class RigidTrajectory(Closeable):
    """Read one immutable clip under ``root/clips/<clip>/manifest.json``.

    A dynamic node is one sampled world pose, optionally with positive diagonal
    asset-relative scale in v3. Objects reference its
    index, or -1 for their constant pose. Several mesh objects may reference the
    same node. Node indices and semantic IDs are different namespaces.

    ``identity`` binds metadata, assets, camera and pose/frame file hashes. It can
    identify a future replay checkpoint together with a frame index; it is not
    an engine checkpoint. Existing G34Replay State and rendering are unchanged.
    """

    resource_kind = "object_permanence.RigidTrajectory"

    def __init__(self, root, clip):
        if not isinstance(clip, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", clip):
            raise ValueError("clip must be a plain identifier")
        self.root = Path(root)
        directory = self.root / "clips" / clip
        self._manifest = json.loads((directory / "manifest.json").read_text())
        m = self._manifest
        if m.get("schema") not in (SCHEMA, SCHEMA_V2, SCHEMA_V3, SCHEMA_V4) or m.get("conventions") != CONVENTIONS:
            raise ValueError("unsupported rigid-scene schema or conventions")
        if m.get("clip") != clip:
            raise ValueError("clip identity differs")
        if "license_ref" in m:
            key = m["license_ref"]["sha256"]
            if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
                raise ValueError("invalid license reference")
            if _sha256(self.root / "licenses" / f"{key}.txt") != key:
                raise ValueError("license hash differs")
        self.identity = hashlib.sha256(json.dumps(m, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if not isinstance(m.get("fps"), (int, float)) or not np.isfinite(m["fps"]) or m["fps"] <= 0:
            raise ValueError("fps must be finite and positive")
        self._arrays = {}
        scaled = "scales" in m["arrays"]
        if scaled and m["schema"] not in (SCHEMA_V3, SCHEMA_V4):
            raise ValueError("scale arrays require rigid-scene v3")
        if (m["schema"] == SCHEMA_V3 or scaled) and m.get("scale_convention") != SCALE_CONVENTION:
            raise ValueError("v3 requires the asset-relative positive scale convention")
        for name in ("poses", "frames", *(["scales"] if scaled else [])):
            description = m["arrays"][name]
            path = directory / f"{name}.npy"
            if _sha256(path) != description["sha256"]:
                raise ValueError(f"{name} file hash differs")
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(value.shape) != description["shape"] or str(value.dtype) != description["dtype"]:
                raise ValueError(f"{name} shape/dtype differs")
            self._arrays[name] = value
        self.frames, self._poses = self._arrays["frames"], self._arrays["poses"]
        self._scales = self._arrays.get("scales")
        if self.frames.ndim != 1 or not len(self.frames) or self.frames.dtype != np.int64:
            raise ValueError("frames must be nonempty int64 [T]")
        if np.any(np.diff(self.frames) != 1):
            raise ValueError("v1 requires consecutive source frames")
        nodes = m["dynamic_nodes"]
        if len(set(nodes)) != len(nodes) or any(not isinstance(n, str) or not n for n in nodes):
            raise ValueError("dynamic node names must be unique nonempty strings")
        # Validate in bounded chunks without materializing a copy of the clip.
        if self._poses.shape != (len(self.frames), len(nodes), 7):
            raise ValueError("pose shape differs from frames/nodes")
        for start in range(0, len(self.frames), 4096):
            rows = self._poses[start : start + 4096]
            _poses(rows, (len(rows), len(nodes), 7))
        if scaled:
            value = self._scales
            assert value is not None
            if value.dtype != np.float32 or value.shape != (len(self.frames), len(nodes), 3):
                raise ValueError("scales must be float32 [T,D,3]")
            for start in range(0, len(self.frames), 4096):
                rows = value[start : start + 4096]
                if not np.isfinite(rows).all() or np.any(rows <= 0):
                    raise ValueError("scales must be finite and strictly positive")
            if not np.array_equal(value[0], np.ones_like(value[0])):
                raise ValueError("asset-relative scales must start at one")
        objects = m["objects"]
        transparent = m["schema"] == SCHEMA_V4
        if not transparent and "material_convention" in m:
            raise ValueError("alpha material policies require rigid-scene v4")
        if transparent and (
            m.get("material_convention") != "stable-envs/constant-alpha-surfaces-v1"
            or m.get("visibility_contract") != "stable-envs/transparent-pixel-rays-v1"
            or m["camera"].get("projection_type") not in ("perspective", "orthographic")
        ):
            raise ValueError("v4 requires explicit alpha material, visibility and camera conventions")
        if not objects or len({o["name"] for o in objects}) != len(objects):
            raise ValueError("objects must have unique names")
        if len({o["id"] for o in objects}) != len(objects):
            raise ValueError("object IDs must be unique")
        used = set()
        for obj in objects:
            node = obj["node"]
            if type(node) is not int or node < -1 or node >= len(nodes):
                raise ValueError("object references an invalid dynamic node")
            if type(obj["id"]) is not int or not 0 < obj["id"] < 32768:
                raise ValueError("object ID must be a positive int16")
            _poses(np.asarray(obj["pose"], np.float32), (7,))
            rgba = np.asarray(obj["rgba"])
            if (
                rgba.shape != (4,)
                or not np.isfinite(rgba).all()
                or (not 0 <= rgba[3] <= 1 if transparent else rgba[3] != 1)
            ):
                raise ValueError("v1 requires finite opaque RGBA")
            if node >= 0:
                used.add(node)
        if used != set(range(len(nodes))):
            raise ValueError("unreferenced dynamic node")
        self._meshes = {}
        for key, description in m["assets"].items():
            if not re.fullmatch(r"[0-9a-f]{64}", key):
                raise ValueError("asset key must be a SHA256 digest")
            relative = description.get("file", f"assets/{key}.npz")
            if not re.fullmatch(r"assets/(?:[A-Za-z0-9_-]+/)?" + key + r"\.npz", relative):
                raise ValueError("invalid asset path")
            path = self.root / relative
            if _sha256(path) != description["sha256"]:
                raise ValueError("asset file hash differs")
            with np.load(path, allow_pickle=False) as archive:
                mesh = {name: archive[name] for name in archive.files}
            if set(mesh) != {"vertices", "normals", "triangles"} or array_digest(mesh) != key:
                raise ValueError("asset content hash differs")
            v, n, f = (mesh[name] for name in ("vertices", "normals", "triangles"))
            if (
                v.dtype != np.float32
                or v.ndim != 2
                or v.shape[1] != 3
                or not len(v)
                or n.dtype != np.float32
                or n.shape != v.shape
                or not np.isfinite(v).all()
                or not np.isfinite(n).all()
                or f.dtype != np.int32
                or f.ndim != 2
                or f.shape[1] != 3
                or not len(f)
                or f.min() < 0
                or f.max() >= len(v)
            ):
                raise ValueError("invalid local triangle mesh")
            for array in mesh.values():
                array.flags.writeable = False
            self._meshes[key] = mesh
        if {o["asset"] for o in objects} != set(self._meshes):
            raise ValueError("objects/assets differ")
        camera = m["camera"]
        if m["schema"] in (SCHEMA_V2, SCHEMA_V3) and (
            camera.get("projection_type") not in ("perspective", "orthographic")
            or m.get("visibility_contract")
            not in (
                ("stable-envs/opaque-pixel-rays-v2",)
                if m["schema"] == SCHEMA_V2 or camera.get("projection_type") == "orthographic"
                else ("stable-envs/opaque-pixel-rays-v1", "stable-envs/opaque-pixel-rays-v2")
            )
        ):
            raise ValueError("v2/v3 recordings require an explicit projection type and visibility contract")
        for key in ("world", "projection"):
            value = np.asarray(camera[key])
            if value.shape != (4, 4) or not np.isfinite(value).all():
                raise ValueError("camera matrices must be finite [4,4]")
        if any(type(camera[k]) is not int or camera[k] <= 0 for k in ("width", "height")):
            raise ValueError("camera dimensions must be positive integers")
        self._opened()

    @property
    def manifest(self):
        """Return an independent description; callers cannot mutate the reader."""
        self._check_open("manifest")
        return deepcopy(self._manifest)

    def __len__(self):
        self._check_open("len")
        return len(self.frames)

    def _indices(self, indices):
        indices = np.asarray(indices)
        if indices.dtype.kind not in "iu" or indices.ndim != 1 or not len(indices):
            raise ValueError("indices must be a nonempty integer vector")
        if (indices < 0).any() or (indices >= len(self)).any():
            raise ValueError("index outside trajectory")
        return indices

    def poses(self, indices):
        """Return owned CPU float32 [B,D,7] using zero-based clip indices."""
        self._check_open("poses")
        indices = self._indices(indices)
        # Integer-array indexing already creates independent storage.
        return np.asarray(self._poses[indices])

    @property
    def has_scales(self):
        """Whether this clip stores optional per-node relative diagonal scales."""
        self._check_open("has_scales")
        return self._scales is not None

    def scales(self, indices):
        """Return owned CPU float32 [B,D,3]; old clips return unit scales."""
        self._check_open("scales")
        indices = self._indices(indices)
        if self._scales is None:
            return np.ones((len(indices), len(self._manifest["dynamic_nodes"]), 3), np.float32)
        return np.asarray(self._scales[indices])

    def object_scales(self, indices):
        """Expand relative node scales to owned CPU float32 [B,G,3]."""
        rows = self.scales(indices)
        objects = self._manifest["objects"]
        out = np.ones((len(rows), len(objects), 3), np.float32)
        for index, obj in enumerate(objects):
            if obj["node"] >= 0:
                out[:, index] = rows[:, obj["node"]]
        return out

    def object_poses(self, indices):
        """Expand dynamic-node poses to owned CPU float32 [B,G,7] world poses."""
        rows = self.poses(indices)
        objects = self._manifest["objects"]
        out = np.broadcast_to(
            np.asarray([o["pose"] for o in objects], np.float32), (len(rows), len(objects), 7)
        ).copy()
        for index, obj in enumerate(objects):
            if obj["node"] >= 0:
                out[:, index] = rows[:, obj["node"]]
        return out

    def mesh(self, key):
        """Return independent local mesh arrays for one content-addressed asset."""
        self._check_open("mesh")
        return {name: value.copy() for name, value in self._meshes[key].items()}

    def triangles(self):
        """Return NumPy inputs for nvdiff.Scene.from_triangles, in object order.

        File-level assets are shared. This assembled view duplicates an asset
        for each geometry instance, as the existing nvdiff interface requires.
        Its one-based geometry IDs map through manifest.objects to semantic IDs.
        """
        self._check_open("triangles")
        vertices, normals, triangles, owners, colors = [], [], [], [], []
        offset = 0
        for index, obj in enumerate(self._manifest["objects"]):
            mesh = self._meshes[obj["asset"]]
            v = mesh["vertices"]
            vertices.append(v)
            normals.append(mesh["normals"])
            triangles.append(mesh["triangles"] + offset)
            owners.append(np.full(len(v), index, np.int64))
            colors.append(obj["rgba"])
            offset += len(v)
        return {
            "vert": np.concatenate(vertices),
            "norm": np.concatenate(normals),
            "face": np.concatenate(triangles),
            "vert_geom": np.concatenate(owners),
            "rgba": np.asarray(colors, np.float32),
        }

    def _release(self):
        self._arrays.clear()
        self.frames = self._poses = self._scales = None
        self._meshes.clear()
