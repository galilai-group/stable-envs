from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict


def _as_np(v, shape: tuple[int, ...]) -> np.ndarray | None:
    if v is None:
        return None
    return np.broadcast_to(np.asarray(v, dtype=np.float64), shape).copy()


@dataclass(frozen=True)
class Spec:
    """Describe one lane's tensor shape, dtype, optional bounds, and column names.

    The batch axis is excluded from shape. Sampling uses a NumPy generator and
    returns a tensor on the configured device.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float32
    low: Any = None
    high: Any = None
    device: torch.device | None = None
    #: Optional labels, one per element of a one-dimensional spec.
    names: tuple[str, ...] = ()
    #: Integer cardinality; None uses continuous bounds when supplied.
    discrete_n: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))
        object.__setattr__(self, "names", tuple(self.names or ()))
        if self.device is not None and not isinstance(self.device, torch.device):
            object.__setattr__(self, "device", torch.device(self.device))
        if self.names and (len(self.shape) != 1 or len(self.names) != self.shape[0]):
            raise ValueError(
                f"Spec(shape={self.shape}, names={self.names}): names label the elements of a 1-D "
                f"spec, one each, so there must be exactly {self.shape} of them."
            )

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def bounded(self) -> bool:
        return self.discrete_n is not None or (self.low is not None and self.high is not None)

    def batch_shape(self, num_envs: int) -> tuple[int, ...]:
        return (int(num_envs), *self.shape)

    def to(self, device) -> Spec:
        return replace(self, device=torch.device(device))

    def zeros(self, num_envs: int, device=None) -> torch.Tensor:
        return torch.zeros(
            self.batch_shape(num_envs), dtype=self.dtype, device=device if device is not None else self.device
        )

    def sample(self, num_envs: int, rng: np.random.Generator | None = None, device=None) -> torch.Tensor:
        """Sample (num_envs, *shape) values with the supplied NumPy generator.

        Discrete values use [0, discrete_n); bounded values use a uniform draw followed
        by conversion to the declared dtype. Unbounded specs cannot be sampled.
        """
        rng = rng if rng is not None else np.random.default_rng()
        dev = device if device is not None else self.device
        n = int(num_envs)
        if self.discrete_n is not None:
            v = rng.integers(0, int(self.discrete_n), size=(n, *self.shape))
        elif self.low is not None and self.high is not None:
            lo, hi = _as_np(self.low, self.shape), _as_np(self.high, self.shape)
            v = rng.uniform(lo, hi, size=(n, *self.shape))
        else:
            raise ValueError(
                f"Spec(shape={self.shape}) declares no bounds, so it cannot be sampled from. Whatever "
                f"this describes, the distribution lives with whoever knows the reachable set — for a "
                f"state that is the environment's own reset distribution, env.sample_states(n)."
            )
        return torch.as_tensor(np.asarray(v), dtype=self.dtype, device=dev)

    def validate(self, x, owner: str = "value") -> torch.Tensor:
        """Check trailing shape, cast to the declared dtype, and validate bounds/finiteness.

        Return the converted tensor on its original device. Batch size is unrestricted;
        dtype conversion occurs before value checks.
        """
        t = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))
        if tuple(t.shape[1:]) != self.shape:
            raise ValueError(
                f"{owner}: expected (batch, {', '.join(str(s) for s in self.shape) or ''}) and got "
                f"{tuple(t.shape)}"
                + (f"; the {len(self.names)} column(s) are {list(self.names)}" if self.names else "")
            )
        t = t.to(self.dtype)
        if t.is_floating_point() and not bool(torch.isfinite(t).all()):
            raise ValueError(f"{owner}: holds NaN or inf.")
        if self.discrete_n is not None:
            if bool(((t < 0) | (t >= self.discrete_n)).any()):
                raise ValueError(f"{owner}: values must lie in [0, {self.discrete_n}).")
        elif self.low is not None and self.high is not None:
            lo = torch.as_tensor(_as_np(self.low, self.shape), dtype=torch.float64, device=t.device)
            hi = torch.as_tensor(_as_np(self.high, self.shape), dtype=torch.float64, device=t.device)
            v = t.to(torch.float64)
            if bool((v < lo).any() or (v > hi).any()):
                raise ValueError(
                    f"{owner}: values outside the declared bounds "
                    f"[{np.min(_as_np(self.low, self.shape))}, "
                    f"{np.max(_as_np(self.high, self.shape))}]."
                )
        return t

    def contains(self, x) -> bool:
        try:
            self.validate(x)
        except ValueError:
            return False
        return True

    def __repr__(self) -> str:
        bits = [f"shape={self.shape}", f"dtype={str(self.dtype).replace('torch.', '')}"]
        if self.discrete_n is not None:
            bits.append(f"n={self.discrete_n}")
        elif self.bounded:
            bits.append(
                f"bounds=[{np.min(_as_np(self.low, self.shape)):g}, {np.max(_as_np(self.high, self.shape)):g}]"
            )
        if self.names:
            bits.append(f"names={list(self.names)}")
        return f"Spec({', '.join(bits)})"


@dataclass
class CompositeSpec:
    """Nested tensor specs addressed by entity paths, such as spec["agent", "position"]."""

    specs: dict[str, Spec | CompositeSpec] = field(default_factory=dict)
    device: torch.device | None = None

    def __init__(self, specs: dict | None = None, device=None, **kw):
        merged = dict(specs or {})
        merged.update(kw)
        self.specs = {k: (v if isinstance(v, (Spec, CompositeSpec)) else CompositeSpec(v)) for k, v in merged.items()}
        self.device = torch.device(device) if device is not None else None

    def __getitem__(self, key) -> Spec | CompositeSpec:
        if isinstance(key, tuple):
            node: Any = self
            for part in key:
                node = node[part]
            return node
        if key not in self.specs:
            raise KeyError(f"{key!r} is not a key of this spec; it has {sorted(self.specs)}")
        return self.specs[key]

    def __contains__(self, key) -> bool:
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[str]:
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    def keys(self, nested: bool = False):
        if not nested:
            return list(self.specs)
        return [k for k, _ in self.items(nested=True)]

    def items(self, nested: bool = False):
        if not nested:
            return list(self.specs.items())
        out: list[tuple[tuple[str, ...], Spec]] = []
        for k, v in self.specs.items():
            if isinstance(v, CompositeSpec):
                out.extend(((k, *sub), s) for sub, s in v.items(nested=True))
            else:
                out.append(((k,), v))
        return out

    def to(self, device) -> CompositeSpec:
        return CompositeSpec({k: v.to(device) for k, v in self.specs.items()}, device=device)

    def zeros(self, num_envs: int, device=None) -> TensorDict:
        dev = device if device is not None else self.device
        return TensorDict(
            {k: v.zeros(num_envs, device=dev) for k, v in self.specs.items()}, batch_size=[int(num_envs)], device=dev
        )

    def sample(self, num_envs: int, rng: np.random.Generator | None = None, device=None) -> TensorDict:
        dev = device if device is not None else self.device
        rng = rng if rng is not None else np.random.default_rng()
        return TensorDict(
            {k: v.sample(num_envs, rng, device=dev) for k, v in self.specs.items()},
            batch_size=[int(num_envs)],
            device=dev,
        )

    def validate(self, x, owner: str = "value") -> TensorDict:
        """Validate exact nested keys and each leaf, then return the original input.

        Leaf conversion is used for checking only; this does not replace input values.
        """
        if not hasattr(x, "keys"):
            raise TypeError(
                f"{owner}: expected a TensorDict/mapping with keys {sorted(self.specs)}, got {type(x).__name__}."
            )
        have = set(x.keys())
        want = set(self.specs)
        if have != want:
            missing, extra = sorted(want - have), sorted(have - want)
            raise ValueError(
                f"{owner}: keys do not match the spec"
                + (f"; missing {missing}" if missing else "")
                + (f"; unexpected {extra}" if extra else "")
            )
        for k, v in self.specs.items():
            v.validate(x[k], owner=f"{owner}[{k!r}]")
        return x

    def contains(self, x) -> bool:
        try:
            self.validate(x)
        except (TypeError, ValueError):
            return False
        return True

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}: {v!r}" for k, v in self.specs.items())
        return f"CompositeSpec({inner})"


__all__ = ["Spec", "CompositeSpec"]
