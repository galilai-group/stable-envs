"""Per-family ``VariationMap`` modules, installed into ``variation.VARIATION_MAPS``.

Each family module declares ``MAPS: dict[str, VariationMap]``, keyed by stable-envs class name as
``VARIATION_MAPS`` is. ``install`` refuses to overwrite an existing key, so two modules cannot both
claim one cell.
"""

from __future__ import annotations

from collections.abc import Mapping

#: Family modules, in install order.
_MODULES = ("dmc_control", "fetch", "newcells")


def _collect() -> dict:
    import importlib

    out: dict = {}
    for name in _MODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        for cell, vmap in mod.MAPS.items():
            if cell in out:
                raise RuntimeError(
                    f"variation_maps: {cell!r} is declared by both {out[cell][1]} and {name}; a cell has one mapping"
                )
            out[cell] = (vmap, name)
    return out


def install(registry: Mapping) -> dict:
    """Merge every family module's maps into ``registry`` and return ``cell -> module name``.

    Raises if a cell is already in ``registry``.
    """
    where = {}
    for cell, (vmap, name) in _collect().items():
        if cell in registry:
            raise RuntimeError(
                f"variation_maps.{name} declares {cell!r}, which variation.VARIATION_MAPS already "
                f"carries; remove one of the two"
            )
        registry[cell] = vmap
        where[cell] = name
    return where
