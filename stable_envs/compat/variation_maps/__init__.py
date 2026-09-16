"""Per-family ``VariationMap`` tables, split out of ``compat/worldmodel.py`` so several families can
be written independently without three workstreams editing one dict.

Each submodule exports a ``dict[str, VariationMap]`` keyed by OUR class name, and
``compat.worldmodel`` merges them into ``VARIATION_MAPS``. A family that is absent is simply a
family with no map yet, which ``check_mapping`` already reports as such.
"""

__all__ = ["newcells"]
