"""Compiled appearance rows: a ``(B, style_dim)`` float32 tensor addressed by variation path.

``StyleTensor.for_batch`` is the only operation that changes the batch size of a compiled row.
"""

from __future__ import annotations

import numpy as np
import torch


def rows_are_equal(*tensors: torch.Tensor) -> bool:
    """Test equality across batch rows, reducing all comparisons before one host read."""
    ok = None
    for t in tensors:
        if t is None or t.ndim == 0 or t.shape[0] <= 1 or t.numel() == 0:
            continue
        eq = (t == t[:1]).all()
        ok = eq if ok is None else (ok & eq)
    return True if ok is None else bool(ok)


def _as_tensor(v, device, dtype=torch.float32) -> torch.Tensor:
    if torch.is_tensor(v):
        return v.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(v), device=device, dtype=dtype)


class StyleTensor:
    """Appearance values of shape ``(B, style_dim)``, indexed by variation path.

    ``layout`` lists ``(path, offset, width)`` for each leaf. ``shared`` records whether all
    rows are known to be equal; None defers the check until :attr:`shared` is read.
    """

    def __init__(self, values: torch.Tensor, layout: list[tuple[str, int, int]], shared: bool | None = None):
        self.values = values
        self.layout = {p: (o, w) for p, o, w in layout}
        self.paths = [p for p, _, _ in layout]
        self._shared = shared

    @property
    def batch(self) -> int:
        return int(self.values.shape[0])

    @property
    def shared(self) -> bool:
        """Whether all rows match; the first query may synchronize with the device.

        The result is cached. Do not mutate ``values`` after this property is read.
        """
        if self._shared is None:
            self._shared = rows_are_equal(self.values)
        return self._shared

    def __contains__(self, path: str) -> bool:
        return path in self.layout

    def __getitem__(self, path: str) -> torch.Tensor:
        """Return a leaf view: ``(B, width)``, or ``(B,)`` for scalar leaves."""
        if path not in self.layout:
            raise KeyError(f"unknown style path {path!r}; known: {self.paths}")
        o, w = self.layout[path]
        v = self.values[:, o : o + w]
        return v if w > 1 else v.reshape(-1)

    def get(self, path: str, default=None):
        """Return the leaf view for ``path``, or ``default`` when the layout has no such leaf."""
        return self[path] if path in self.layout else default

    def _layout_list(self) -> list[tuple[str, int, int]]:
        return [(p, o, w) for p, (o, w) in self.layout.items()]

    def to(self, device=None, dtype=None) -> StyleTensor:
        """Return a copy on ``device`` with ``dtype``, keeping the layout and known sharedness."""
        return StyleTensor(
            self.values.to(device=device, dtype=dtype or self.values.dtype), self._layout_list(), shared=self._shared
        )

    def for_batch(self, batch: int, owner: str = "render") -> StyleTensor:
        """Return this style at ``batch`` rows.

        A matching batch returns ``self``. A shared style broadcasts its row; a style
        with differing rows raises ValueError, since no row selection is implied.
        """
        b = int(batch)
        if self.batch == b:
            return self
        if not self.shared:
            raise ValueError(
                f"{owner}: PER-ENVIRONMENT style has {self.batch} rows; render requests {b} environment(s). "
                "Only one shared appearance can broadcast to another batch size. "
                "Slice the style with the states: env.render(states[i:j], style=env.get_style()[i:j])."
            )
        return StyleTensor(
            self.values[:1].expand(b, self.values.shape[1]).contiguous(), self._layout_list(), shared=True
        )

    @classmethod
    def from_space(cls, space, batch: int, device="cuda", overrides: dict | None = None, values=None) -> StyleTensor:
        """Build float32 appearance rows from space defaults or supplied values.

        Values must contain one shared row or exactly ``batch`` rows. Overrides map
        paths to either one leaf value or one value per batch row; they apply last.
        """
        layout = space.flat_layout()
        dim = space.flat_dim
        dev = torch.device(device)
        b = int(batch)
        shared: bool | None
        if values is None:
            base = torch.as_tensor(space.pack(), dtype=torch.float32, device=dev)
            vals = base.reshape(1, dim).repeat(b, 1)
            shared = True
        elif dim == 0:
            # Avoid the ambiguous reshape(-1, 0) for environments without variations.
            vals = torch.zeros((b, 0), dtype=torch.float32, device=dev)
            shared = True
        else:
            v = _as_tensor(values, dev)
            # Check the last axis before reshaping; a transpose can have the right element count.
            if v.ndim >= 2 and v.shape[-1] != dim:
                raise ValueError(
                    f"style rows are {dim} number(s) wide, got shape {tuple(v.shape)}. "
                    f"Use (batch, {dim}) or one shared ({dim},) row in env.style_names order."
                )
            if v.ndim <= 1 and v.numel() % dim:
                raise ValueError(
                    f"a flat style must be a whole number of {dim}-wide rows, and {v.numel()} is not "
                    f"a multiple of {dim}. One row is env.style_dim numbers, in env.style_names order."
                )
            vals = v.reshape(-1, dim)
            shared = True if vals.shape[0] == 1 else None
            if vals.shape[0] == 1 and b != 1:
                vals = vals.repeat(b, 1)
            elif vals.shape[0] != b:
                raise ValueError(
                    f"style has {vals.shape[0]} rows but {b} environment(s) are being rendered. A "
                    f"style is one row shared by the batch or one row per environment, in the order "
                    f"of the states; slice the style together with the states."
                )
        if overrides:
            lut = {p: (o, w) for p, o, w in layout}
            for path, v in overrides.items():
                if path not in lut:
                    raise KeyError(f"unknown style path {path!r}; known: {sorted(lut)}")
                o, w = lut[path]
                t = _as_tensor(v, dev)
                n = t.numel()
                if t.ndim >= 2 and t.shape[-1] != w and n == b * w:
                    raise ValueError(
                        f"variation {path!r}: expected one value per environment as ({b}, {w}) and "
                        f"got {tuple(t.shape)}, which looks like a transpose. Pass ({b}, {w}), or "
                        f"({w},) to share one value across the batch."
                    )
                if n == b * w and b != 1:
                    t = t.reshape(b, w)  # one value per environment
                    if shared:
                        shared = None  # may still be uniform; measure it if asked
                elif n == w:
                    t = t.reshape(1, w).expand(b, w)  # one value shared by the batch
                else:
                    raise ValueError(
                        f"variation {path!r}: expected {w} numbers shared by the batch or "
                        f"{b * w} ({b} environments × {w}); got {n} with shape {tuple(t.shape)}."
                    )
                vals = vals.clone()
                vals[:, o : o + w] = t
        return cls(vals.contiguous(), layout, shared=shared)
