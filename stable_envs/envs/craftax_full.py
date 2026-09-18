"""Deprecated import path. ``stable_envs.envs.craftax_full`` IS ``stable_envs.envs._craftax_full``.

Plan §3's target tree names one ``craftax.py`` for this family, and Step 9 took the brief's stated
fallback instead: full Craftax lives in a PRIVATE module beside it rather than merged into it. The
reason is measured, not stylistic. The two modules share 34 module-level names, of which **15 hold
different values** — ``N_BLOCK`` is 17 for Classic and 37 for Full, ``N_STRIP`` 16 and 41, ``STRIP_H``
2 and 4, ``TILE_PX_AGENT`` 7 and 10, plus ``N_TILE``, ``N_COUNT``, ``OBS_H``, ``OBS_W``,
``REFERENCE_TILE_PX``, the three functions ``strip_geometry``, ``night_vignette`` and ``_unpack``, and
the three per-device caches — so one module
object cannot serve both spellings, and a merge would have to RENAME constants that tests and scratch
scripts import by name (``tests/unit/test_craftax_full.py`` alone reaches for 48 of them). That is not
behaviour-neutral, which is the condition the brief attached to the merge.

This file is a ``sys.modules`` ALIAS rather than a set of re-exports, so ``from stable_envs.envs
import craftax_full as CF`` keeps working for every name the old module had, including the private
ones the tests reach for, and a monkeypatch applied through either path is seen by both because there
is only one module object. Auto-discovery imports THIS module (it skips ``_``-prefixed ones), so
``CraftaxFull`` registers exactly as before and a missing ``craftax`` is still recorded under the key
``craftax_full``. Wave 3 decides whether the private module keeps its underscore.
"""
import sys

from . import _craftax_full

sys.modules[__name__] = _craftax_full
