"""Bounded SWM Rocket Variation behavior, including scalar types and sampling order.

Derived from stable-worldmodel (MIT), commit
176d54532b3c0a251a4f509b74f2fdacc143a63d, by Lucas Maes,
Quentin Le Lidec, Randall Balestriero and contributors. This module covers only
Rocket's unconstrained Box/RGBBox/Dict surface; shared spaces are unchanged.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from ..spaces import Dict as PublicDict


class Box(spaces.Box):
    def __init__(self, low, high, shape=None, init_value=None, **kwargs):
        super().__init__(low, high, shape, **kwargs)
        # Source preserves Python float 0.9, rather than rounding it to float32.
        self._init_value = init_value
        self._value = init_value

    @property
    def value(self):
        return self._value

    @property
    def init_value(self):
        return self._init_value

    def reset(self):
        self._value = self._init_value

    def check(self):
        return self.contains(self._value)

    def sample(self, mask=None, set_value=True, **kwargs):
        value = super().sample(mask=mask)
        if set_value:
            self._value = value
        return value

    def set_init_value(self, value):
        if not self.contains(value):
            raise ValueError("Value outside Rocket Variation space")
        self._init_value = value

    def set_value(self, value):
        if not self.contains(value):
            raise ValueError("Value outside Rocket Variation space")
        self._value = value


class RGBBox(Box):
    def __init__(self, init_value):
        super().__init__(0, 255, (3,), init_value=init_value, dtype=np.uint8)


class Dict(PublicDict):
    def __init__(self, spaces_dict, sampling_order=None):
        super().__init__(spaces_dict)
        self._sampling_order = list(self.spaces) if sampling_order is None else sampling_order

    def ordered_names(self, prefix=""):
        for key in self._sampling_order:
            path = prefix + key
            yield path
            if isinstance(self[key], Dict):
                yield from self[key].ordered_names(path + ".")

    def sample(self, mask=None, set_value=True, **kwargs):
        return {
            k: self[k].sample(mask=None if mask is None else mask.get(k), set_value=set_value)
            for k in self._sampling_order
        }

    def update(self, keys):
        selected = set(keys)
        if selected == {"all"}:
            self.sample()
        else:
            for name in self.ordered_names():
                if name in selected:
                    self._walk(name).sample()
        assert self.check(), "Values must be within space!"


def nodes(space, path=""):
    yield path, space
    for key, value in getattr(space, "spaces", {}).items():
        yield from nodes(value, path + "/" + key)
