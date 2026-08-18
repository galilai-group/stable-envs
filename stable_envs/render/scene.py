"""The 2-D scene IR: the one thing every 2-D env emits, and the one thing the rasterizer consumes.

The PushT prototype that motivated this repo was a kernel written for PushT — it knew about a T-block
and a disc. That does not scale to a zoo. What actually generalises is the observation that the 2-D
research envs draw from a tiny vocabulary: convex polygons, discs, and thick line segments, composited
back-to-front with flat colours. CartPole is two polygons and a disc; Pendulum is a polygon and two
discs; a MiniGrid tile is a rectangle; PushT is two quads, a disc and four segments. So the env's job is
to turn ``(state, variation)`` into a ``Scene2D``, and exactly one kernel rasterizes it.

Everything is batched on the leading dim, and that is the point: primitive *geometry and colour are
per-environment tensors*, so a batch can hold 4096 environments with different block colours, different
agent radii and different backgrounds, and still render in one launch. Domain randomisation stops being
a for-loop.

Layout. Primitives are packed into fixed-width arrays so the kernel does flat, predictable loads:

    kind  (B,N) int32   0 = disabled, 1 = disc, 2 = convex polygon, 3 = segment
    geom  (B,N,G) f32   disc:    [cx, cy, r, inner_r, ...]
                        poly:    [x0,y0, x1,y1, ... xV-1,yV-1]      (CCW, convex)
                        segment: [x0,y0, x1,y1, half_w, ...]
    fill  (B,N,3) f32   interior colour, 0-255
    line  (B,N,3) f32   outline colour, 0-255
    line_w(B,N)   f32   outline half-width in world units; 0 disables the outline

``kind == 0`` is a real feature, not padding hygiene: a batch is one rectangular tensor, so an env whose
primitive count varies with its state (a maze with a variable number of walls, a scene where the goal
marker is toggled off by ``variation['rendering']['render_goal']``) expresses that by disabling slots
rather than by changing shape. Shape stays static, which is what lets the whole step be captured in a
CUDA graph.

Draw order is index order, back to front. Painter's algorithm with flat colours has no depth to resolve,
so ordering the array *is* the z-buffer — and it makes the order explicit and diffable, which matters:
in PushT, whether the agent or the block paints last is not documented anywhere and had to be recovered
by pixel-diffing both against the reference renderer.

Polygons are padded to ``V`` vertices by repeating the last one. Degenerate (zero-length) edges are
detected in the kernel and excluded from the SDF; they are harmless for the inside test (their cross
product is 0, which passes) but would otherwise pin the outline distance to 0 and paint the whole plane.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

KIND_NONE = 0
KIND_DISC = 1
KIND_POLY = 2
KIND_SEGMENT = 3


@dataclass
class Scene2D:
    """A batch of flat 2-D scenes, ready to rasterize.

    ``world`` is the extent of the coordinate system the primitives live in, as ``(x0, y0, x1, y1)``.
    The rasterizer maps it onto the output image, so an env keeps its own native units (PushT's 512-unit
    board, MiniGrid's tile grid, CartPole's metres) and never hand-converts to pixels. Getting that
    conversion wrong in each env separately is precisely the class of bug that produced a y-flipped
    PushT render.
    """

    kind: torch.Tensor          # (B,N) int32
    geom: torch.Tensor          # (B,N,G) float32
    fill: torch.Tensor          # (B,N,3) float32, 0-255
    line: torch.Tensor          # (B,N,3) float32, 0-255
    line_w: torch.Tensor        # (B,N)   float32, world units (half-width); 0 = no outline
    background: torch.Tensor    # (B,3)   float32, 0-255
    world: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    flip_y: bool = False        # True when the env's +y points up but the image's +y points down
    #: Number of LEADING primitives that are identical across the batch and across calls (given the
    #: same variation). They are rasterized once into a cached base image and skipped every frame
    #: afterwards. Most scenes are mostly static — PushT's goal T and four walls are 6 of its 9
    #: primitives.
    #:
    #: This is a REQUEST, not a promise. The rasterizer paints one shared base image over the whole
    #: batch, so a prefix that is not batch-uniform would give every environment environment 0's goal,
    #: walls and background — and rather than trust the field, the rasterizer measures the prefix and
    #: declines to bake when it differs across the batch (see ``raster2d._static_base``). Asking for a
    #: prefix that is not shared therefore costs the speedup and never the picture, which is what lets
    #: an env declare it once and still render per-environment appearances correctly.
    static_prefix: int = 0

    @property
    def batch(self) -> int:
        return int(self.kind.shape[0])

    @property
    def n_prims(self) -> int:
        return int(self.kind.shape[1])

    @property
    def vertices(self) -> int:
        return int(self.geom.shape[2]) // 2

    def to(self, device=None, dtype=None) -> "Scene2D":
        f = lambda t, d=None: t.to(device=device, dtype=d or t.dtype)
        return Scene2D(kind=f(self.kind), geom=f(self.geom, dtype), fill=f(self.fill, dtype),
                       line=f(self.line, dtype), line_w=f(self.line_w, dtype),
                       background=f(self.background, dtype), world=self.world, flip_y=self.flip_y,
                       static_prefix=self.static_prefix)

    def validate(self) -> "Scene2D":
        """Shape/dtype contract, checked once at build time rather than misrendered silently."""
        B, N = self.kind.shape
        assert self.geom.shape[:2] == (B, N), f"geom {tuple(self.geom.shape)} vs kind {(B, N)}"
        assert self.geom.shape[2] % 2 == 0, "geom width must be even (x,y pairs)"
        assert self.fill.shape == (B, N, 3), f"fill {tuple(self.fill.shape)}"
        assert self.line.shape == (B, N, 3), f"line {tuple(self.line.shape)}"
        assert self.line_w.shape == (B, N), f"line_w {tuple(self.line_w.shape)}"
        assert self.background.shape == (B, 3), f"background {tuple(self.background.shape)}"
        assert self.kind.dtype == torch.int32, f"kind dtype {self.kind.dtype}, expected int32"
        x0, y0, x1, y1 = self.world
        assert x1 > x0 and y1 > y0, f"degenerate world extent {self.world}"
        return self


class SceneBuilder:
    """Accumulates primitives into a ``Scene2D``.

    Envs build their scene declaratively — ``add_disc``, ``add_poly``, ``add_segment`` in draw order —
    and every argument accepts either a scalar (shared by the whole batch) or a ``(B,...)`` tensor (one
    value per environment). That is what makes appearance variation free: an env passes
    ``variation['block']['color'].value`` whether it is one colour or 4096 different ones, and the
    builder broadcasts.
    """

    def __init__(self, batch: int, device="cuda", vertices: int = 4, dtype=torch.float32,
                 style=None):
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
        #: primitives appended so far. Not ``len(self._kind)`` any more: a chunk holds n of them.
        self._n = 0
        #: Optional ``StyleTensor`` this scene is being drawn from. A static prefix depends on the
        #: variation and nothing else — that is what ``mark_static`` means — so a batch-uniform
        #: appearance gives a batch-uniform prefix, and ``StyleTensor.shared`` settles for free (cached
        #: bool) a question the rasterizer would otherwise settle with a device-to-host round trip.
        #: Optional because an env may build a scene from values that never passed through a style
        #: tensor; the rasterizer then measures the assembled primitives instead, and the two agree.
        self._style = style

    # broadcasting helpers

    def _b(self, v, width: int) -> torch.Tensor:
        """Scalar / (width,) / (B,width) -> (B,width) float32 on device."""
        t = torch.as_tensor(v, device=self.device, dtype=self.dtype)
        if t.ndim == 0:
            t = t.reshape(1, 1).expand(self.B, width)
        elif t.ndim == 1:
            t = (t.reshape(1, -1).expand(self.B, -1) if t.shape[0] == width
                 else t.reshape(-1, 1).expand(self.B, width))
        return t.reshape(self.B, width).contiguous()

    def _enabled(self, enabled) -> torch.Tensor:
        """Per-primitive on/off, broadcast to ``(B,)`` bool.

        ``enabled`` lands in ``kind``, and the static bake takes ``kind[:1]``, so a prefix whose only
        per-environment quantity is a visibility flag (``variation['rendering']['render_goal']`` set
        per environment, with a shared colour) is exactly as unbakeable as one with a per-environment
        colour. The rasterizer compares ``kind`` alongside the other four arrays before it bakes, so
        that case needs no separate watch here.
        """
        if enabled is None:
            return torch.ones(self.B, device=self.device, dtype=torch.bool)
        t = torch.as_tensor(enabled, device=self.device)
        return (t.reshape(1).expand(self.B) if t.ndim == 0 else t.reshape(self.B)).bool()

    def _bn(self, v, n: int, width: int) -> torch.Tensor:
        """``_b`` with a primitive axis: scalar / ``(width,)`` / ``(B,width)`` / ``(B,n,width)``."""
        t = torch.as_tensor(v, device=self.device, dtype=self.dtype)
        if t.ndim == 0:
            t = t.reshape(1, 1, 1).expand(self.B, n, width)
        elif t.ndim == 1:
            t = (t.reshape(1, 1, -1).expand(self.B, n, -1) if t.shape[0] == width
                 else t.reshape(-1, 1, 1).expand(self.B, n, width))
        elif t.ndim == 2:
            t = t.reshape(self.B, 1, width).expand(self.B, n, width)
        return t.reshape(self.B, n, width).contiguous()

    def _enabled_n(self, enabled, n: int) -> torch.Tensor:
        if enabled is None:
            return torch.ones((self.B, n), device=self.device, dtype=torch.bool)
        t = torch.as_tensor(enabled, device=self.device)
        if t.ndim == 0:
            t = t.reshape(1, 1).expand(self.B, n)
        elif t.ndim == 1:
            t = t.reshape(self.B, 1).expand(self.B, n)
        return t.reshape(self.B, n).bool()

    def _push(self, kind_id, geom, fill, line, line_w, enabled, n: int = 1):
        """Append ``n`` primitives. ``n == 1`` takes ``(B, ...)``; otherwise ``(B, n, ...)``.

        The batched form exists because dispatch, not arithmetic, is what a scene of a few hundred
        primitives costs: highway-env's roundabout issued 53,481 torch calls to build 436 of them and
        spent 315 ms doing it, the same 315 ms at batch 16 as at batch 1024. One call per primitive
        is the wrong granularity for a builder whose whole point is that a batch is free.
        """
        b = self.B
        sh = (b,) if n == 1 else (b, n)
        k = torch.where(self._enabled(enabled) if n == 1 else self._enabled_n(enabled, n),
                        torch.full(sh, kind_id, device=self.device, dtype=torch.int32),
                        torch.zeros(sh, device=self.device, dtype=torch.int32))
        pad = 2 * self.V - geom.shape[-1]
        if pad > 0:
            geom = torch.cat([geom, torch.zeros(*sh, pad, device=self.device, dtype=self.dtype)], -1)
        geom = geom[..., :2 * self.V]
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

    # primitives

    def add_disc(self, center, radius, color, inner_radius=0.0, inner_color=None, enabled=None):
        """A filled disc, optionally with a concentric inner disc drawn on top (a ring).

        The ring is one primitive rather than two because that is how the reference renderers draw it —
        pymunk's debug draw paints ``circle(r)`` then ``circle(r-4)`` in a lighter shade — and splitting
        it would let the two halves drift apart under a variation that changes only one of them.
        """
        c = self._b(center, 2)
        r = self._b(radius, 1)
        ir = self._b(inner_radius, 1)
        geom = torch.cat([c, r, ir], dim=1)
        self._push(KIND_DISC, geom, color, inner_color if inner_color is not None else color,
                   0.0, enabled)

    def add_poly(self, verts, color, outline_color=None, outline_w=0.0, enabled=None):
        """A convex polygon, vertices CCW, ``(B,V,2)`` or ``(V,2)``. Padded to the builder's V."""
        t = torch.as_tensor(verts, device=self.device, dtype=self.dtype)
        if t.ndim == 2:
            t = t.unsqueeze(0).expand(self.B, -1, -1)
        v = t.shape[1]
        if v < self.V:              # repeat the last vertex; the kernel drops degenerate edges
            t = torch.cat([t, t[:, -1:].expand(-1, self.V - v, -1)], dim=1)
        self._push(KIND_POLY, t[:, :self.V].reshape(self.B, -1).contiguous(),
                   color, outline_color, outline_w, enabled)

    def add_segment(self, a, b, half_w, color, enabled=None):
        """A thick line segment with round caps — a capsule, i.e. distance-to-segment <= half_w."""
        geom = torch.cat([self._b(a, 2), self._b(b, 2), self._b(half_w, 1)], dim=1)
        self._push(KIND_SEGMENT, geom, color, color, 0.0, enabled)

    def add_rect(self, center, size, color, angle=None, outline_color=None, outline_w=0.0,
                 enabled=None):
        """Axis-aligned or rotated rectangle, as a convex polygon. The most common primitive by far."""
        c = self._b(center, 2)
        s = self._b(size, 2) * 0.5
        base = torch.stack([torch.stack([-s[:, 0], -s[:, 1]], -1),
                            torch.stack([s[:, 0], -s[:, 1]], -1),
                            torch.stack([s[:, 0], s[:, 1]], -1),
                            torch.stack([-s[:, 0], s[:, 1]], -1)], dim=1)      # (B,4,2) CCW
        if angle is not None:
            th = self._b(angle, 1).squeeze(1)
            ct, st = torch.cos(th), torch.sin(th)
            rot = torch.stack([torch.stack([ct, -st], -1), torch.stack([st, ct], -1)], -2)
            base = torch.einsum("bij,bkj->bki", rot, base)
        self.add_poly(base + c.unsqueeze(1), color, outline_color, outline_w, enabled)

    # batched primitives: one call, n primitives per environment, in the order of the n axis
    #
    # These are the same three primitives with a primitive axis folded in, and they exist for
    # dispatch rather than for expressiveness -- see ``_push``. Everything below folds ``(B, n)``
    # into a single leading axis and reuses the SAME expressions the one-primitive form uses, rather
    # than rewriting them with an extra index: an elementwise op does not care how many rows it has,
    # but ``einsum`` and ``norm`` over an extra axis are each entitled to a different last bit, and
    # one ulp on a vertex is a pixel wherever that vertex lands on a boundary.

    def add_segments(self, a, b, half_w, color, enabled=None) -> None:
        """``n`` capsules per environment: ``a``/``b`` ``(B,n,2)``, ``half_w`` ``(B,n)`` or scalar."""
        n = int(a.shape[1])
        hw = (half_w.reshape(self.B, n, 1).to(device=self.device, dtype=self.dtype)
              if torch.is_tensor(half_w) and half_w.ndim >= 2 else self._bn(half_w, n, 1))
        geom = torch.cat([self._bn(a, n, 2), self._bn(b, n, 2), hw], dim=-1)
        self._push(KIND_SEGMENT, geom, color, color, 0.0, enabled, n=n)

    def add_polys(self, verts, color, outline_color=None, outline_w=0.0, enabled=None) -> None:
        """``n`` convex polygons per environment, ``verts`` ``(B,n,v,2)``, padded to the builder's V."""
        n, v = int(verts.shape[1]), int(verts.shape[2])
        t = verts.to(device=self.device, dtype=self.dtype)
        if v < self.V:
            t = torch.cat([t, t[:, :, -1:].expand(-1, -1, self.V - v, -1)], dim=2)
        self._push(KIND_POLY, t[:, :, :self.V].reshape(self.B, n, -1).contiguous(),
                   color, outline_color, outline_w, enabled, n=n)

    def add_rects(self, center, size, color, angle=None, outline_color=None, outline_w=0.0,
                  enabled=None) -> None:
        """``n`` rectangles per environment: ``center``/``size`` ``(B,n,2)``, ``angle`` ``(B,n)``."""
        n = int(center.shape[1])
        M = self.B * n
        c = center.to(device=self.device, dtype=self.dtype).reshape(M, 2)
        s = size.to(device=self.device, dtype=self.dtype).reshape(M, 2) * 0.5
        base = torch.stack([torch.stack([-s[:, 0], -s[:, 1]], -1),
                            torch.stack([s[:, 0], -s[:, 1]], -1),
                            torch.stack([s[:, 0], s[:, 1]], -1),
                            torch.stack([-s[:, 0], s[:, 1]], -1)], dim=1)      # (M,4,2) CCW
        if angle is not None:
            th = torch.as_tensor(angle, device=self.device, dtype=self.dtype).reshape(M)
            ct, st = torch.cos(th), torch.sin(th)
            rot = torch.stack([torch.stack([ct, -st], -1), torch.stack([st, ct], -1)], -2)
            base = torch.einsum("bij,bkj->bki", rot, base)
        self.add_polys((base + c.unsqueeze(1)).reshape(self.B, n, 4, 2), color, outline_color,
                       outline_w, enabled)

    def mark_static(self) -> "SceneBuilder":
        """Everything added so far depends on the variation and not on the state; offer it for baking.

        The env declares the prefix; whether it is *baked* depends on whether it is actually identical
        across the batch, and that is decided by measuring the values rather than by trusting the
        declaration. A prefix that is not shared would be rasterized once from environment 0 and
        painted over every environment — the second of the three collapses recorded in
        ``core/style.py`` — so the case has to be caught, and the case is common: per-environment
        backgrounds and goal colours are the domain randomisation the repo advertises.

        Where that measurement happens is a cost decision, and it is made once. If this builder was
        handed the ``StyleTensor`` the scene is drawn from, a non-shared appearance settles it here
        for free (a static prefix is a function of the variation alone, so a per-environment
        appearance means a per-environment prefix, and ``StyleTensor.shared`` is a cached bool).
        Otherwise the declaration is passed on and ``raster2d._static_base`` — which must read the
        prefix's bytes anyway to key its cache — measures it there and declines to bake if it differs.
        Measuring in both places was tried and cost 0.25 ms per call for a second copy of the same
        answer.

        The refusal used to be decided by ``stride(0) != 0`` on the incoming values, which never fired
        correctly. A ``(B, w)`` slice of a style tensor has a non-zero row stride whether or not its
        rows differ, and ``add_rect`` hands ``add_poly`` freshly computed vertices whatever it was
        given, so EVERY prefix was called per-environment as soon as ``B > 1``: the bake was dead code
        outside batch 1, in a repo whose entire subject is batches of thousands. Sharedness is a
        property of the values, so it is now decided from the values.
        """
        self._static = self._n
        return self

    def _requested_prefix(self) -> int:
        """The ``static_prefix`` to put on the scene: the declared prefix, or 0 if it cannot be shared."""
        K = int(self._static)
        if K <= 0 or self.B <= 1 or self._style is None:
            return K
        return K if self._style.shared else 0

    def build(self, background, world, flip_y: bool = False) -> Scene2D:
        assert self._kind, "scene has no primitives"
        cat = lambda xs: (xs[0] if len(xs) == 1 else torch.cat(xs, dim=1)).contiguous()  # noqa: E731
        return Scene2D(
            kind=cat(self._kind),
            geom=cat(self._geom),
            fill=cat(self._fill),
            line=cat(self._line),
            line_w=cat(self._lw),
            background=self._b(background, 3),
            world=tuple(float(w) for w in world), flip_y=bool(flip_y),
            static_prefix=self._requested_prefix(),
        ).validate()
