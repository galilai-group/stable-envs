"""Variation spaces: every appearance and layout knob an env has, declared as a space you can sample.

The API deliberately mirrors ``stable_worldmodel.spaces`` — same class names, same
``init_value`` / ``value`` / ``reset`` / ``sample`` / ``set_value`` / ``check`` surface, same
``reset_variation_space`` entry point — so an env written against one works against the other and
``stable_worldmodel`` envs can move here without a rewrite. It is vendored rather than imported so that
``stable_envs`` stays installable on its own; the compatibility is the point, the dependency is not.

What changes here is what the values are *for*. In ``stable_worldmodel`` a variation is read once at
reset and baked into a CPU renderer. Here the variation values flow into a ``Scene2D`` as per-environment
tensors, so a batch of 4096 environments can each carry a different block colour, agent radius or
background, and still render in a single kernel launch. Domain randomisation becomes a tensor, not a
loop over envs.

``batch_sample`` exists for exactly that: it draws B independent values for a subtree and returns them
stacked, which is the form the scene builder consumes.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
from gymnasium import spaces as gspaces


class _Valued:
    """Shared value/constraint machinery for the leaf spaces."""

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

    def set_init_value(self, value) -> None:
        if not super().contains(value):
            raise ValueError(f"{type(self).__name__}: {value!r} is not in the space")
        self._init_value = value
        self._value = value

    def set_value(self, value) -> None:
        if not super().contains(value):
            raise ValueError(f"{type(self).__name__}: {value!r} is not in the space")
        self._value = value

    def check(self) -> bool:
        return super().contains(self._value) and self.constrain_fn(self._value)


class Discrete(_Valued, gspaces.Discrete):
    def __init__(self, n: int, init_value: int | None = None, constrain_fn=None, **kw):
        gspaces.Discrete.__init__(self, n, **kw)
        _Valued.__init__(self, 0 if init_value is None else int(init_value), constrain_fn)

    def contains(self, x) -> bool:
        return gspaces.Discrete.contains(self, x) and self.constrain_fn(x)

    def sample(self, mask=None, max_tries: int = 1000, set_value: bool = True, **kw):
        for _ in range(max_tries):
            v = int(gspaces.Discrete.sample(self, mask=mask))
            if self.constrain_fn(v):
                if set_value:
                    self._value = v
                return v
        raise RuntimeError(f"Discrete: no sample satisfied constrain_fn in {max_tries} tries")


class Box(_Valued, gspaces.Box):
    def __init__(self, low, high, shape=None, dtype=np.float32, init_value=None,
                 constrain_fn=None, **kw):
        gspaces.Box.__init__(self, low, high, shape=shape, dtype=dtype, **kw)
        if init_value is None:
            init_value = np.zeros(self.shape, dtype=self.dtype)
            init_value = np.clip(init_value, self.low, self.high).astype(self.dtype)
        else:
            init_value = np.asarray(init_value, dtype=self.dtype).reshape(self.shape)
        _Valued.__init__(self, init_value, constrain_fn)

    def contains(self, x) -> bool:
        return gspaces.Box.contains(self, np.asarray(x, dtype=self.dtype)) and self.constrain_fn(x)

    def set_value(self, value) -> None:
        _Valued.set_value(self, np.asarray(value, dtype=self.dtype).reshape(self.shape))

    def set_init_value(self, value) -> None:
        _Valued.set_init_value(self, np.asarray(value, dtype=self.dtype).reshape(self.shape))

    def sample(self, mask=None, max_tries: int = 1000, set_value: bool = True, **kw):
        for _ in range(max_tries):
            v = gspaces.Box.sample(self)
            if self.constrain_fn(v):
                if set_value:
                    self._value = v
                return v
        raise RuntimeError("Box: no sample satisfied constrain_fn")

    def batch_sample(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """``(batch, *shape)`` independent draws — the form the GPU scene builder wants.

        Uses a plain uniform over [low, high] rather than B calls to ``sample``: for 4096 environments
        the per-call overhead of gymnasium's sampler dominates, and a per-env appearance draw is exactly
        the thing this repo is trying to make free.
        """
        rng = rng or np.random.default_rng()
        lo = np.broadcast_to(self.low, self.shape).astype(np.float64)
        hi = np.broadcast_to(self.high, self.shape).astype(np.float64)
        return rng.uniform(lo, hi, size=(batch, *self.shape)).astype(self.dtype)


class RGBBox(Box):
    """A colour, as 3 channels in 0-255. Separate from ``Box`` so a renderer can recognise it."""

    def __init__(self, init_value=(0, 0, 0), low=0, high=255, **kw):
        super().__init__(low=low, high=high, shape=(3,), dtype=np.uint8,
                         init_value=np.asarray(init_value, dtype=np.uint8), **kw)


class Dict(gspaces.Dict):
    """A tree of variation spaces, addressed by ``space['block']['color']``."""

    def __init__(self, spaces_dict: dict, **kw):
        super().__init__(spaces_dict, **kw)

    @property
    def init_value(self) -> dict:
        return {k: (v.init_value if hasattr(v, "init_value") else None) for k, v in self.spaces.items()}

    @property
    def value(self) -> dict:
        return {k: v.value for k, v in self.spaces.items()}

    def reset(self) -> None:
        for v in self.spaces.values():
            v.reset()

    def check(self, debug: bool = False) -> bool:
        ok = True
        for k, v in self.spaces.items():
            r = v.check(debug) if isinstance(v, Dict) else v.check()
            if not r and debug:
                print(f"[variation] {k} = {getattr(v, 'value', None)!r} is out of space")
            ok = ok and r
        return ok

    def names(self, prefix: str = "") -> list[str]:
        """Every leaf path, e.g. ``block.color``. This is the env's advertised knob list."""
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
        """Resample exactly the named leaves (dotted paths), leaving everything else alone."""
        for k in keys:
            node = self._walk(k)
            if isinstance(node, Dict):
                for leaf in node.names():
                    node._walk(leaf).sample()
            else:
                node.sample()

    def set_value(self, values: dict) -> None:
        for k, v in values.items():
            self._walk(k).set_value(v)

    def set_init_value(self, values: dict) -> None:
        for k, v in values.items():
            self._walk(k).set_init_value(v)

    def flat_layout(self) -> list[tuple[str, int, int]]:
        """``[(path, offset, width), ...]`` — the flat vector layout of every leaf, in name order.

        This is what makes style storable. A variation tree is convenient to author and impossible to
        put in a dataset column; the flat layout gives it a fixed width and a stable order so a row of
        floats round-trips back to the same appearance. Sorted by path so the layout depends only on
        the space's structure, never on dict insertion order — a layout that shifts between runs would
        silently reinterpret every stored style.
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
        lay = self.flat_layout()
        return (lay[-1][1] + lay[-1][2]) if lay else 0

    def pack(self, values: dict | None = None) -> np.ndarray:
        """Current (or supplied) leaf values -> a flat float32 vector of width ``flat_dim``."""
        vec = np.zeros(self.flat_dim, np.float32)
        for path, off, w in self.flat_layout():
            v = values[path] if values is not None and path in values else self._walk(path).value
            vec[off:off + w] = np.asarray(v, np.float32).reshape(-1)[:w]
        return vec

    def unpack(self, vec) -> dict:
        """Flat vector -> ``{path: value}``. Inverse of ``pack``."""
        vec = np.asarray(vec, np.float32).reshape(-1)
        return {p: (vec[o] if w == 1 else vec[o:o + w]) for p, o, w in self.flat_layout()}

    def to_str(self, indent: int = 0) -> str:
        lines = []
        for k, v in self.spaces.items():
            pad = "  " * indent
            if isinstance(v, Dict):
                lines.append(f"{pad}{k}:")
                lines.append(v.to_str(indent + 1))
            else:
                lines.append(f"{pad}{k} = {getattr(v, 'value', None)}")
        return "\n".join(lines)


def reset_variation_space(variation_space: Dict, seed: int | None = None, options: dict | None = None,
                          default_variations: Sequence[str] | None = None) -> None:
    """Reset, then resample the requested leaves, then apply explicit overrides. swm-compatible."""
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
