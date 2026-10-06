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
    """Appearance values of shape ``(B, style_dim)``, indexed by variation path."""

    def __init__(self, values: torch.Tensor, layout: list[tuple[str, int, int]], shared: bool | None = None):
        self.values = values
        self.layout = {p: (o, w) for p, o, w in layout}
        self.paths = [p for p, _, _ in layout]
        # Defer the device-to-host equality check until sharedness is needed.
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
        return self[path] if path in self.layout else default

    def _layout_list(self) -> list[tuple[str, int, int]]:
        return [(p, o, w) for p, (o, w) in self.layout.items()]

    def to(self, device=None, dtype=None) -> StyleTensor:
        return StyleTensor(
            self.values.to(device=device, dtype=dtype or self.values.dtype), self._layout_list(), shared=self._shared
        )

    def for_batch(self, batch: int, owner: str = "render") -> StyleTensor:
        """Return this style unchanged when its batch size matches.

        Otherwise, broadcast equal rows to the requested size. Different per-lane
        appearances cannot be resized without an explicit row selection.
        """
        b = int(batch)
        if self.batch == b:
            return self
        if not self.shared:
            raise ValueError(
                f"{owner}: the env carries a PER-ENVIRONMENT style of {self.batch} rows and this call "
                f"asks for {b} environment(s). A style with one row (or with every row equal) "
                f"broadcasts to any batch; {self.batch} different appearances cannot be spread over "
                f"{b} environments, because nothing in the states says which of the {self.batch} rows "
                f"they are. This used to substitute row 0's appearance for every row and say nothing. "
                f"Index the style the same way as the states, e.g. "
                f"env.render(states[i:j], style=env.get_style()[i:j])."
            )
        return StyleTensor(
            self.values[:1].expand(b, self.values.shape[1]).contiguous(), self._layout_list(), shared=True
        )

    @classmethod
    def from_space(cls, space, batch: int, device="cuda", overrides: dict | None = None, values=None) -> StyleTensor:
        """Build float32 appearance rows from space defaults or supplied values.

        Values must contain one shared row or exactly ``batch`` rows. Overrides map
        paths to either one leaf value or one value per lane; they apply last.
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
                    f"style rows are {dim} number(s) wide and the last axis of this one is "
                    f"{v.shape[-1]} (shape {tuple(v.shape)}). A style is (batch, style_dim) or one "
                    f"shared (style_dim,) row; anything else used to be flattened and regrouped, "
                    f"which for a transposed array succeeds and gives every environment a mixture of "
                    f"other environments' knobs. env.style_names lists the {dim} columns in order."
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
                    f"style must be one row (shared by the whole batch) or exactly one row per "
                    f"environment, in the same order as the states — anything else has no defined "
                    f"meaning, and the row count is exactly what a caller gets wrong when they slice "
                    f"the states and forget the style."
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
                        f"got {tuple(t.shape)}, which holds the right NUMBER of values in the wrong "
                        f"arrangement — almost always a transpose. Reshaping it by size would give "
                        f"environment i a mixture of the other environments' values and render "
                        f"without complaint, so it is refused. Pass ({b}, {w}), or ({w},) to share "
                        f"one value across the batch."
                    )
                if n == b * w and b != 1:
                    t = t.reshape(b, w)  # one value per environment
                    if shared:
                        shared = None  # may still be uniform; measure it if asked
                elif n == w:
                    t = t.reshape(1, w).expand(b, w)  # one value shared by the batch
                else:
                    raise ValueError(
                        f"variation {path!r}: expected {w} numbers (shared by the batch) or "
                        f"{b * w} ({b} environments x {w}), got {n} with shape "
                        f"{tuple(t.shape)}. A count that is neither used to be taken as 'the first "
                        f"{w} numbers, broadcast', i.e. environment 0's value for everybody."
                    )
                vals = vals.clone()
                vals[:, o : o + w] = t
        return cls(vals.contiguous(), layout, shared=shared)
