from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Union

import torch

# Shared scalars/vectors or tensors carrying per-environment values.
Batched = Union[float, int, Sequence[float], torch.Tensor]

KIND_NONE = 0
KIND_DISC = 1
KIND_POLY = 2
KIND_SEGMENT = 3


@dataclass
class Scene2D:
    """Batched 2-D primitives in draw order.

    world=(x0, y0, x1, y1) maps native coordinates onto the image; flip_y handles
    upward-positive world axes. KIND_NONE disables a primitive independently
    per row. Floating arrays share device and dtype; kind remains int32.
    """

    kind: torch.Tensor  # (B,N) int32
    geom: torch.Tensor  # (B,N,G) float32
    fill: torch.Tensor  # (B,N,3) float32, 0-255
    line: torch.Tensor  # (B,N,3) float32, 0-255
    line_w: torch.Tensor  # (B,N)   float32, world units (half-width); 0 = no outline
    background: torch.Tensor  # (B,3)   float32, 0-255
    world: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    flip_y: bool = False  # True when the env's +y points up but the image's +y points down
    # Requested bakeable prefix; the rasterizer verifies batch-uniform values before sharing a plate.
    static_prefix: int = 0

    @property
    def batch(self) -> int:
        return int(self.kind.shape[0])

    @property
    def n_prims(self) -> int:
        """Return primitive slots per environment, including disabled slots."""
        return int(self.kind.shape[1])

    @property
    def vertices(self) -> int:
        """Return polygon capacity in (x, y) pairs."""
        return int(self.geom.shape[2]) // 2

    def to(self, device: Any = None, dtype: Any = None) -> Scene2D:
        """Move arrays and optionally convert floating dtype, preserving kind dtype."""
        f = lambda t, d=None: t.to(device=device, dtype=d or t.dtype)
        return Scene2D(
            kind=f(self.kind),
            geom=f(self.geom, dtype),
            fill=f(self.fill, dtype),
            line=f(self.line, dtype),
            line_w=f(self.line_w, dtype),
            background=f(self.background, dtype),
            world=self.world,
            flip_y=self.flip_y,
            static_prefix=self.static_prefix,
        )

    def validate(self) -> Scene2D:
        """Validate shapes, int32 kinds, and positive world extents; return self."""
        if self.kind.ndim != 2 or self.kind.dtype != torch.int32:
            raise ValueError("kind must have shape (B, N) and dtype int32")
        B, N = self.kind.shape
        if self.geom.ndim != 3 or self.geom.shape[:2] != (B, N) or self.geom.shape[2] % 2:
            raise ValueError("geom must have shape (B, N, 2 * vertices)")
        for name, shape in (("fill", (B, N, 3)), ("line", (B, N, 3)), ("line_w", (B, N)), ("background", (B, 3))):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if len(self.world) != 4:
            raise ValueError("world must contain (x0, y0, x1, y1)")
        x0, y0, x1, y1 = self.world
        if not (x1 > x0 and y1 > y0):
            raise ValueError(f"degenerate world extent {self.world}")
        return self


class SceneBuilder:
    """Append primitives in draw order and build a batched Scene2D.

    Arguments accept shared values or values per environment. Plural add methods
    also accept a primitive axis. A width-sized vector is shared; use (B, 1) for
    per-row scalar values. In plural methods, (B, width) is per environment;
    use (1, n, width) for per-primitive values shared across environments.
    """

    def __init__(
        self,
        batch: int,
        device: Any = "cpu",
        vertices: int = 4,
        dtype: torch.dtype = torch.float32,
        style: Any = None,
    ) -> None:
        self.B = int(batch)
        self.V = int(vertices)
        self.device = torch.device(device)
        self.dtype = dtype
        self._kind: list[torch.Tensor] = []
        self._geom: list[torch.Tensor] = []
        self._fill: list[torch.Tensor] = []
        self._line: list[torch.Tensor] = []
        self._lw: list[torch.Tensor] = []
        self._static = 0
        # Count primitive slots, not appended chunks.
        self._n = 0
        # A non-shared style can reject baking early; otherwise the rasterizer checks primitive values.
        self._style = style

    def _b(self, v: Batched, width: int) -> torch.Tensor:
        """Broadcast scalar, shared vector, or per-row values to (B, width)."""
        t = torch.as_tensor(v, device=self.device, dtype=self.dtype)
        if t.ndim == 0:
            t = t.reshape(1, 1).expand(self.B, width)
        elif t.ndim == 1:
            t = t.reshape(1, -1).expand(self.B, -1) if t.shape[0] == width else t.reshape(-1, 1).expand(self.B, width)
        return t.reshape(self.B, width).contiguous()

    def _enabled(self, enabled: Batched | None) -> torch.Tensor:
        """Broadcast visibility to (B,) bool; disabled slots receive KIND_NONE."""
        if enabled is None:
            return torch.ones(self.B, device=self.device, dtype=torch.bool)
        t = torch.as_tensor(enabled, device=self.device)
        return (t.reshape(1).expand(self.B) if t.ndim == 0 else t.reshape(self.B)).bool()

    def _bn(self, v: Batched, n: int, width: int) -> torch.Tensor:
        """Broadcast values to (B, n, width), preserving a supplied primitive axis."""
        t = torch.as_tensor(v, device=self.device, dtype=self.dtype)
        if t.ndim == 0:
            t = t.reshape(1, 1, 1).expand(self.B, n, width)
        elif t.ndim == 1:
            t = (
                t.reshape(1, 1, -1).expand(self.B, n, -1)
                if t.shape[0] == width
                else t.reshape(-1, 1, 1).expand(self.B, n, width)
            )
        elif t.ndim == 2:
            t = t.reshape(self.B, 1, width).expand(self.B, n, width)
        elif t.ndim == 3:
            t = t.expand(self.B, n, width)
        return t.reshape(self.B, n, width).contiguous()

    def _enabled_n(self, enabled: Batched | None, n: int) -> torch.Tensor:
        """Broadcast scalar or per-row visibility to (B, n)."""
        if enabled is None:
            return torch.ones((self.B, n), device=self.device, dtype=torch.bool)
        t = torch.as_tensor(enabled, device=self.device)
        if t.ndim == 0:
            t = t.reshape(1, 1).expand(self.B, n)
        elif t.ndim == 1:
            t = t.reshape(self.B, 1).expand(self.B, n)
        return t.reshape(self.B, n).bool()

    def _push(
        self,
        kind_id: int,
        geom: torch.Tensor,
        fill: Batched,
        line: Batched | None,
        line_w: Batched,
        enabled: Batched | None,
        n: int = 1,
    ) -> None:
        """Append n ordered primitive slots, padding geometry to the configured vertex capacity."""
        b = self.B
        sh = (b,) if n == 1 else (b, n)
        k = torch.where(
            self._enabled(enabled) if n == 1 else self._enabled_n(enabled, n),
            torch.full(sh, kind_id, device=self.device, dtype=torch.int32),
            torch.zeros(sh, device=self.device, dtype=torch.int32),
        )
        pad = 2 * self.V - geom.shape[-1]
        if pad > 0:
            geom = torch.cat([geom, torch.zeros(*sh, pad, device=self.device, dtype=self.dtype)], -1)
        elif pad < 0:
            raise ValueError(f"geometry exceeds vertex capacity {self.V}")
        if n == 1:
            fill_t = self._b(fill, 3)
            line_t = self._b(line if line is not None else fill, 3)
            lw_t = self._b(line_w, 1).squeeze(1)
            k, geom = k.unsqueeze(1), geom.unsqueeze(1)
            fill_t, line_t = fill_t.unsqueeze(1), line_t.unsqueeze(1)
            lw_t = lw_t.unsqueeze(1)
        else:
            fill_t = self._bn(fill, n, 3)
            line_t = self._bn(line if line is not None else fill, n, 3)
            lw_t = self._bn(line_w, n, 1).squeeze(-1)
        self._kind.append(k)
        self._geom.append(geom)
        self._fill.append(fill_t)
        self._line.append(line_t)
        self._lw.append(lw_t)
        self._n += n

    def add_disc(
        self,
        center: Batched,
        radius: Batched,
        color: Batched,
        inner_radius: Batched = 0.0,
        inner_color: Batched | None = None,
        enabled: Batched | None = None,
    ) -> None:
        """Add a filled disc with an optional concentric inner disc and independent color."""
        c = self._b(center, 2)
        r = self._b(radius, 1)
        ir = self._b(inner_radius, 1)
        geom = torch.cat([c, r, ir], dim=1)
        self._push(KIND_DISC, geom, color, inner_color if inner_color is not None else color, 0.0, enabled)

    def add_poly(
        self,
        verts: Batched,
        color: Batched,
        outline_color: Batched | None = None,
        outline_w: Batched = 0.0,
        enabled: Batched | None = None,
    ) -> None:
        """Add a convex CCW polygon from (V, 2) or (B, V, 2) vertices.

        Repeat the last vertex when padding; exceeding vertex capacity raises ValueError.
        """
        t = torch.as_tensor(verts, device=self.device, dtype=self.dtype)
        if t.ndim == 2:
            t = t.unsqueeze(0).expand(self.B, -1, -1)
        v = t.shape[1]
        if not 3 <= v <= self.V:
            raise ValueError(f"polygon requires 3..{self.V} vertices, got {v}")
        if v < self.V:  # repeat the last vertex; the kernel drops degenerate edges
            t = torch.cat([t, t[:, -1:].expand(-1, self.V - v, -1)], dim=1)
        self._push(KIND_POLY, t.reshape(self.B, -1).contiguous(), color, outline_color, outline_w, enabled)

    def add_segment(
        self, a: Batched, b: Batched, half_w: Batched, color: Batched, enabled: Batched | None = None
    ) -> None:
        """Add a capsule from endpoints and a half-width in world units."""
        geom = torch.cat([self._b(a, 2), self._b(b, 2), self._b(half_w, 1)], dim=1)
        self._push(KIND_SEGMENT, geom, color, color, 0.0, enabled)

    @staticmethod
    def _corners(s):
        return torch.stack(
            [
                torch.stack([-s[:, 0], -s[:, 1]], -1),
                torch.stack([s[:, 0], -s[:, 1]], -1),
                torch.stack([s[:, 0], s[:, 1]], -1),
                torch.stack([-s[:, 0], s[:, 1]], -1),
            ],
            dim=1,
        )  # (B,4,2) CCW

    def add_rect(
        self,
        center: Batched,
        size: Batched,
        color: Batched,
        angle: Batched | None = None,
        outline_color: Batched | None = None,
        outline_w: Batched = 0.0,
        enabled: Batched | None = None,
    ) -> None:
        """Add a rectangle with full width/height and optional rotation in radians."""
        c = self._b(center, 2)
        s = self._b(size, 2) * 0.5
        base = self._corners(s)
        if angle is not None:
            th = self._b(angle, 1).squeeze(1)
            ct, st = torch.cos(th), torch.sin(th)
            rot = torch.stack([torch.stack([ct, -st], -1), torch.stack([st, ct], -1)], -2)
            base = torch.einsum("bij,bkj->bki", rot, base)
        self.add_poly(base + c.unsqueeze(1), color, outline_color, outline_w, enabled)

    # Flatten batch and primitive axes to reuse the scalar-form expressions and their rounding.

    def add_segments(
        self, a: torch.Tensor, b: torch.Tensor, half_w: Batched, color: Batched, enabled: Batched | None = None
    ) -> None:
        """Add n capsules per row with endpoints (B, n, 2) and shared or per-primitive widths."""
        n = int(a.shape[1])
        hw = (
            half_w.reshape(self.B, n, 1).to(device=self.device, dtype=self.dtype)
            if torch.is_tensor(half_w) and half_w.ndim >= 2
            else self._bn(half_w, n, 1)
        )
        geom = torch.cat([self._bn(a, n, 2), self._bn(b, n, 2), hw], dim=-1)
        self._push(KIND_SEGMENT, geom, color, color, 0.0, enabled, n=n)

    def add_polys(
        self,
        verts: torch.Tensor,
        color: Batched,
        outline_color: Batched | None = None,
        outline_w: Batched = 0.0,
        enabled: Batched | None = None,
    ) -> None:
        """Add convex CCW polygons (B, n, V, 2), rejecting excess vertices."""
        n, v = int(verts.shape[1]), int(verts.shape[2])
        if not 3 <= v <= self.V:
            raise ValueError(f"polygon requires 3..{self.V} vertices, got {v}")
        t = verts.to(device=self.device, dtype=self.dtype)
        if v < self.V:
            t = torch.cat([t, t[:, :, -1:].expand(-1, -1, self.V - v, -1)], dim=2)
        self._push(
            KIND_POLY,
            t.reshape(self.B, n, -1).contiguous(),
            color,
            outline_color,
            outline_w,
            enabled,
            n=n,
        )

    def add_rects(
        self,
        center: torch.Tensor,
        size: torch.Tensor,
        color: Batched,
        angle: Batched | None = None,
        outline_color: Batched | None = None,
        outline_w: Batched = 0.0,
        enabled: Batched | None = None,
    ) -> None:
        """Add rectangles with centers and full sizes (B, n, 2), optionally rotated per rectangle."""
        n = int(center.shape[1])
        M = self.B * n
        c = center.to(device=self.device, dtype=self.dtype).reshape(M, 2)
        s = size.to(device=self.device, dtype=self.dtype).reshape(M, 2) * 0.5
        base = self._corners(s)
        if angle is not None:
            th = torch.as_tensor(angle, device=self.device, dtype=self.dtype).reshape(M)
            ct, st = torch.cos(th), torch.sin(th)
            rot = torch.stack([torch.stack([ct, -st], -1), torch.stack([st, ct], -1)], -2)
            base = torch.einsum("bij,bkj->bki", rot, base)
        self.add_polys((base + c.unsqueeze(1)).reshape(self.B, n, 4, 2), color, outline_color, outline_w, enabled)

    def mark_static(self) -> SceneBuilder:
        """Offer all primitives appended so far as a state-independent prefix for caching.

        The rasterizer checks that prefix values are uniform across the batch before
        baking. A supplied non-shared style declines the request early. Strides alone
        cannot establish equal values.
        """
        self._static = self._n
        return self

    def _requested_prefix(self) -> int:
        K = int(self._static)
        if K <= 0 or self.B <= 1 or self._style is None:
            return K
        return K if self._style.shared else 0

    def build(self, background: Batched, world: Sequence[float], flip_y: bool = False) -> Scene2D:
        """Build a validated nonempty scene; world is (x0, y0, x1, y1) in native units."""
        if not self._kind:
            raise ValueError("scene has no primitives")
        cat = lambda xs: (xs[0] if len(xs) == 1 else torch.cat(xs, dim=1)).contiguous()  # noqa: E731
        return Scene2D(
            kind=cat(self._kind),
            geom=cat(self._geom),
            fill=cat(self._fill),
            line=cat(self._line),
            line_w=cat(self._lw),
            background=self._b(background, 3),
            world=tuple(float(w) for w in world),
            flip_y=bool(flip_y),
            static_prefix=self._requested_prefix(),
        ).validate()
