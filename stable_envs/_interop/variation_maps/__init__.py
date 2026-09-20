"""One ``VariationMap`` module per family, installed into ``worldmodel.VARIATION_MAPS``.

``stable_envs.interop`` carries the four maps that predate the certified maps inline. The certified maps
adds lives here instead, one module per family, for a reason that is about ownership rather than
tidiness: the maps are written by different workstreams against different evidence, and a single
dict literal in one file is a merge conflict per workstream and no way to say which measurement a
given classification rests on.

A family module declares ``MAPS: dict[str, VariationMap]``, keyed by the stable-envs CLASS name — the
same key ``VARIATION_MAPS`` uses, so ``variation_map(env)`` finds it with no extra lookup. Adding a
family is one import and one entry in ``_MODULES`` below; nothing else in the package changes, and
nothing in ``worldmodel`` changes at all.

``install`` is called once, from the bottom of ``worldmodel``, and REFUSES to overwrite an existing
key. Two modules claiming one cell is a coordination bug and the import should say so rather than let
whichever module imported last win.
"""

from __future__ import annotations

from collections.abc import Mapping

#: Family modules, in the order they are installed. One line per workstream.
_MODULES = ("dmc_control", "fetch", "newcells")


def _collect() -> dict:
    import importlib

    out: dict = {}
    for name in _MODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        for cell, vmap in mod.MAPS.items():
            if cell in out:
                raise RuntimeError(
                    f"variation_maps: {cell!r} is declared by both {out[cell][1]} and {name}; one "
                    f"cell has one appearance mapping — ask the coordinator which module owns it"
                )
            out[cell] = (vmap, name)
    return out


def install(registry: Mapping) -> dict:
    """Merge every family module's maps into ``registry`` (``worldmodel.VARIATION_MAPS``).

    Returns ``cell -> module name`` so a caller can say where a map came from. Refuses to replace a
    map already in ``registry``: the four earlier maps are authoritative for their cells.
    """
    where = {}
    for cell, (vmap, name) in _collect().items():
        if cell in registry:
            raise RuntimeError(
                f"variation_maps.{name} declares {cell!r}, which worldmodel.VARIATION_MAPS already "
                f"carries; remove one of the two rather than shadowing"
            )
        registry[cell] = vmap
        where[cell] = name
    return where
