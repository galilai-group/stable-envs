"""Variation spaces: Gymnasium spaces that also hold a default and a current value."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
from gymnasium import spaces as gspaces

from ._core.config import log


class _Valued:
    """Default and current value of a leaf space, with an optional extra constraint."""

    def __init__(self, init_value=None, constrain_fn: Callable[[Any], bool] | None = None):
        self._init_value = init_value
        self.constrain_fn = constrain_fn or (lambda x: True)
        self._value = init_value

    @property
    def init_value(self):
        return self._init_value

    @property
    def value(self):
        return self._value

    def reset(self) -> None:
        self._value = self._init_value

    def set_init_value(self, value, reset_value: bool = True) -> None:
        if not super().contains(value):
            raise ValueError(f"{type(self).__name__}: {value!r} is not in the space")
        self._init_value = value
        if reset_value:
            self._value = value

    def set_value(self, value) -> None:
        if not super().contains(value):
            raise ValueError(f"{type(self).__name__}: {value!r} is not in the space")
        self._value = value

    def check(self) -> bool:
        return super().contains(self._value) and self.constrain_fn(self._value)


class Discrete(_Valued, gspaces.Discrete):
    """Gymnasium ``Discrete`` with a default and current value; the default ``init_value`` is 0."""

    def __init__(self, n: int, init_value: int | None = None, constrain_fn=None, **kw):
        gspaces.Discrete.__init__(self, n, **kw)
        _Valued.__init__(self, 0 if init_value is None else int(init_value), constrain_fn)

    def contains(self, x) -> bool:
        return gspaces.Discrete.contains(self, x) and self.constrain_fn(x)

    def sample(self, mask=None, max_tries: int = 1000, set_value: bool = True, **kw):
        """Draw a value satisfying ``constrain_fn``, store it unless ``set_value`` is False, and return it."""
        for _ in range(max_tries):
            v = int(gspaces.Discrete.sample(self, mask=mask))
            if self.constrain_fn(v):
                if set_value:
                    self._value = v
                return v
        raise RuntimeError(f"Discrete: no sample satisfied constrain_fn in {max_tries} tries")


class Box(_Valued, gspaces.Box):
    """Gymnasium ``Box`` with a default and current value.

    The default ``init_value`` is zeros clipped into the bounds. ``exact`` keeps init and set
    values as given (Python floats stay float64) instead of casting them to ``dtype``.
    """

    def __init__(self, low, high, shape=None, dtype=np.float32, init_value=None, constrain_fn=None, exact=False, **kw):
        gspaces.Box.__init__(self, low, high, shape=shape, dtype=dtype, **kw)
        self._exact = exact
        if init_value is None:
            init_value = np.zeros(self.shape, dtype=self.dtype)
            init_value = np.clip(init_value, self.low, self.high).astype(self.dtype)
        elif not exact:
            init_value = np.asarray(init_value, dtype=self.dtype).reshape(self.shape)
        _Valued.__init__(self, init_value, constrain_fn)

    def _coerce(self, value):
        return value if self._exact else np.asarray(value, dtype=self.dtype).reshape(self.shape)

    def contains(self, x) -> bool:
        return gspaces.Box.contains(self, np.asarray(x, dtype=self.dtype)) and self.constrain_fn(x)

    def set_value(self, value) -> None:
        _Valued.set_value(self, self._coerce(value))

    def set_init_value(self, value, reset_value: bool = True) -> None:
        _Valued.set_init_value(self, self._coerce(value), reset_value)

    def sample(self, mask=None, max_tries: int = 1000, set_value: bool = True, **kw):
        """Draw a value satisfying ``constrain_fn``, store it unless ``set_value`` is False, and return it."""
        for _ in range(max_tries):
            v = gspaces.Box.sample(self)
            if self.constrain_fn(v):
                if set_value:
                    self._value = v
                return v
        raise RuntimeError("Box: no sample satisfied constrain_fn")

    def batch_sample(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw (batch, *shape) uniform values within bounds using a NumPy generator.

        This vectorized path does not apply constrain_fn or update the stored value.
        """
        rng = rng or np.random.default_rng()
        lo = np.broadcast_to(self.low, self.shape).astype(np.float64)
        hi = np.broadcast_to(self.high, self.shape).astype(np.float64)
        return rng.uniform(lo, hi, size=(batch, *self.shape)).astype(self.dtype)


class RGBBox(Box):
    """Three uint8 color channels in [0, 255], distinguishable from other Box leaves."""

    def __init__(self, init_value=(0, 0, 0), low=0, high=255, **kw):
        super().__init__(
            low=low, high=high, shape=(3,), dtype=np.uint8, init_value=np.asarray(init_value, dtype=np.uint8), **kw
        )


class Dict(gspaces.Dict):
    """Nested variation spaces with current values and dotted leaf paths.

    Passing ``sampling_order`` makes ``sample`` and ``update`` draw top-level keys in that order
    (unlisted keys follow in insertion order); without
    it ``update`` resamples the requested names in the order given.
    """

    def __init__(self, spaces_dict: dict, sampling_order: Sequence[str] | None = None, **kw):
        super().__init__(spaces_dict, **kw)
        unknown = set(sampling_order or ()) - set(self.spaces)
        if unknown:
            raise ValueError(f"sampling_order contains keys not in spaces: {sorted(unknown)}")
        order = list(sampling_order or ())
        self._sampling_order = order + [k for k in self.spaces if k not in order]
        self._ordered = sampling_order is not None

    def ordered_names(self, prefix: str = ""):
        """Yield every dotted path, subtrees included, in sampling order."""
        for key in self._sampling_order:
            path = f"{prefix}{key}"
            yield path
            if isinstance(self.spaces[key], Dict):
                yield from self.spaces[key].ordered_names(path + ".")

    def sample(self, mask=None, set_value: bool = True, **kw) -> dict:
        """Sample every child in sampling order and return the values as a nested dict."""
        return {
            k: self.spaces[k].sample(mask=None if mask is None else mask.get(k), set_value=set_value)
            for k in self._sampling_order
        }

    @property
    def init_value(self) -> dict:
        return {k: (v.init_value if hasattr(v, "init_value") else None) for k, v in self.spaces.items()}

    @property
    def value(self) -> dict:
        return {k: v.value for k, v in self.spaces.items()}

    def reset(self) -> None:
        """Restore every leaf's current value to its default."""
        for v in self.spaces.values():
            v.reset()

    def check(self, debug: bool = False) -> bool:
        """Return whether every current value lies in its space; ``debug`` logs the failing leaves at DEBUG level."""
        ok = True
        for k, v in self.spaces.items():
            r = v.check(debug) if isinstance(v, Dict) else v.check()
            if not r and debug:
                log("DEBUG", f"Variation {k} = {getattr(v, 'value', None)!r} is out of space.")
            ok = ok and r
        return ok

    def names(self, prefix: str = "") -> list[str]:
        """Return all dotted leaf paths, optionally prefixed."""
        out = []
        for k, v in self.spaces.items():
            p = f"{prefix}{k}"
            out.extend(v.names(p + ".") if isinstance(v, Dict) else [p])
        return out

    def _walk(self, path: str):
        node = self
        for part in path.split("."):
            node = node[part]
        return node

    def update(self, keys: Iterable[str]) -> None:
        """Resample named leaves or all leaves of named subtrees, leaving other values unchanged."""
        if self._ordered:
            selected = set(keys)
            if selected == {"all"}:
                self.sample()
                return
            for name in self.ordered_names():
                if name in selected:
                    self._walk(name).sample()
            return
        for k in keys:
            node = self._walk(k)
            if isinstance(node, Dict):
                for leaf in node.names():
                    node._walk(leaf).sample()
            else:
                node.sample()

    def set_value(self, values: dict) -> None:
        """Set current values from ``{dotted path: value}``; values outside their space raise ValueError."""
        for k, v in values.items():
            self._walk(k).set_value(v)

    def set_init_value(self, values: dict, reset_value: bool = True) -> None:
        """Set defaults from ``{dotted path: value}``, and the current values too when ``reset_value``."""
        for k, v in values.items():
            self._walk(k).set_init_value(v, reset_value)

    def flat_layout(self) -> list[tuple[str, int, int]]:
        """Return (path, offset, width) entries sorted by leaf path.

        Sorting keeps stored configuration columns independent of dictionary insertion order.
        """
        out, off = [], 0
        for path in sorted(self.names()):
            node = self._walk(path)
            w = int(np.prod(node.shape)) if hasattr(node, "shape") else 1
            out.append((path, off, w))
            off += w
        return out

    @property
    def flat_dim(self) -> int:
        """Width of a packed row."""
        lay = self.flat_layout()
        return (lay[-1][1] + lay[-1][2]) if lay else 0

    def pack(self, values: dict | None = None) -> np.ndarray:
        """Pack current values into a float32 row, optionally overriding supplied dotted paths."""
        vec = np.zeros(self.flat_dim, np.float32)
        for path, off, w in self.flat_layout():
            v = values[path] if values is not None and path in values else self._walk(path).value
            vec[off : off + w] = np.asarray(v, np.float32).reshape(-1)[:w]
        return vec

    def unpack(self, vec) -> dict:
        """Split a flat float32 row into dotted-path values; vector values may share its storage."""
        vec = np.asarray(vec, np.float32).reshape(-1)
        return {p: (vec[o] if w == 1 else vec[o : o + w]) for p, o, w in self.flat_layout()}

    def to_str(self, indent: int = 0) -> str:
        """Return the current values as indented ``name = value`` lines."""
        lines = []
        for k, v in self.spaces.items():
            pad = "  " * indent
            if isinstance(v, Dict):
                lines.append(f"{pad}{k}:")
                lines.append(v.to_str(indent + 1))
            else:
                lines.append(f"{pad}{k} = {getattr(v, 'value', None)}")
        return "\n".join(lines)


def reset_variation_space(
    variation_space: Dict,
    seed: int | None = None,
    options: dict | None = None,
    default_variations: Sequence[str] | None = None,
) -> None:
    """Seed and reset defaults, resample selected leaves, then apply and validate explicit overrides."""
    variation_space.seed(seed)
    variation_space.reset()
    options = options or {}
    keys = options.get("variation", default_variations or ())
    if not isinstance(keys, Sequence) or isinstance(keys, str):
        raise ValueError("'variation' must be a Sequence of variation names")
    variation_space.update(keys)
    if "variation_values" in options:
        variation_space.set_value(options["variation_values"])
    assert variation_space.check(debug=True), "variation values must lie in the variation space"
