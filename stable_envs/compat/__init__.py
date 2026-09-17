"""Compatibility shim: ``stable_envs.compat.worldmodel`` is now ``stable_envs.interop``.

Step 9 replaced ``compat/`` with one public ``interop.py`` over a private ``_interop`` package (plan
§2). The old module paths keep resolving TO THE SAME MODULE OBJECTS — they are aliases installed in
``sys.modules``, not copies — so ``VARIATION_MAPS`` is one dict, a monkeypatch through either path is
seen by both, and ``VariationMap`` is one class. Removed in Wave 3.
"""
from __future__ import annotations

import sys

from .._interop import variation_maps as _variation_maps
from .._interop import worldmodel as _worldmodel
from .._interop.variation_maps import dmc_control as _dmc_control
from .._interop.variation_maps import fetch as _fetch
from .._interop.variation_maps import newcells as _newcells

worldmodel = _worldmodel
variation_maps = _variation_maps

for _old, _mod in (("stable_envs.compat.worldmodel", _worldmodel),
                   ("stable_envs.compat.variation_maps", _variation_maps),
                   ("stable_envs.compat.variation_maps.dmc_control", _dmc_control),
                   ("stable_envs.compat.variation_maps.fetch", _fetch),
                   ("stable_envs.compat.variation_maps.newcells", _newcells)):
    sys.modules.setdefault(_old, _mod)

__all__ = ["worldmodel", "variation_maps"]
