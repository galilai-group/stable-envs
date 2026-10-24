"""Static pixel visibility seeds for the versioned opaque ray contract.

No scene names, motion formulas or color-layer assumptions. Cache keys include
all inputs to static ray visibility; a varying static partition falls back to a
complete pass. Depth and semantic/face keys retain their original int64 bits.
"""

import torch


class StaticRayCoverage:
    """Bind one immutable topology/partition and reuse owned static pixel seeds."""

    def __init__(self, faces, static_faces):
        if (
            static_faces.dtype != torch.bool
            or static_faces.device != faces.device
            or static_faces.shape != (len(faces),)
        ):
            raise ValueError("static ray faces must be a boolean CUDA face mask")
        self.faces = faces
        self._face_version = faces._version
        self.static = static_faces.nonzero().flatten().to(torch.int32).contiguous()
        self.dynamic = (~static_faces).nonzero().flatten().to(torch.int32).contiguous()
        self.vertices = faces[self.static.long()].flatten().unique().long()
        self.builds = self.fallbacks = 0
        self._snapshot = self._key = self._depth = self._winner = None

    def seed(self, faces, positions, vertices, planes, rays, origin, labels, resolution, near, far, draw):
        """Clone batch seeds, or return None when full coverage is necessary.

        ``draw`` evaluates the supplied subset using the same ray visibility
        passes as a full draw. Snapshots are owned; caller outputs never alias
        the stored seed. Within-batch static variation never poisons the cache.
        """
        if faces is not self.faces or self.faces._version != self._face_version:
            raise ValueError("static ray cache requires immutable face topology")
        if not len(self.static):
            return None
        varied = (positions[:, self.vertices], vertices[:, self.vertices], planes[:, self.static.long()])
        if any(not torch.equal(value, value[:1].expand_as(value)) for value in varied):
            self.fallbacks += 1
            return None
        current = (*(value[:1] for value in varied), rays, origin, labels)
        key = (tuple(resolution), near, far)
        if (
            self._key != key
            or self._snapshot is None
            or any(
                a.dtype != b.dtype or a.device != b.device or not torch.equal(a, b)
                for a, b in zip(current, self._snapshot, strict=True)
            )
        ):
            h, w = resolution
            depth = torch.full((1, h, w), torch.iinfo(torch.int64).max, dtype=torch.int64, device=positions.device)
            winner = torch.full_like(depth, torch.iinfo(torch.int64).max)
            draw(positions[:1], vertices[:1], planes[:1], depth, winner, self.static)
            self._depth, self._winner = depth, winner
            self._snapshot = tuple(value.clone() for value in current)
            self._key = key
            self.builds += 1
        assert self._depth is not None and self._winner is not None
        return (
            self._depth.expand(len(positions), -1, -1).clone(),
            self._winner.expand(len(positions), -1, -1).clone(),
            self.dynamic,
            self._depth,
        )

    def clear(self):
        """Release all owned CUDA tensors, including retained validation inputs."""
        self.faces = self.static = self.dynamic = self.vertices = None
        self._snapshot = self._key = self._depth = self._winner = None
