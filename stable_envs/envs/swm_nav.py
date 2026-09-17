"""Deprecated import path. ``stable_envs.envs.swm_nav`` IS ``stable_envs.envs.navigation``.

Step 9 merged the three ``stable_worldmodel`` navigation ports into one module (plan §3). This file
is a ``sys.modules`` ALIAS rather than a set of re-exports, so ``from stable_envs.envs import
swm_nav as SN`` keeps working for every name the old module had — including the private ones the
render tests reach for (``SN._alpha_blend``, ``SN._ALPHA_FLOOR``, ``SN._gaussian_dot``, ``SN.IMG``)
— and a monkeypatch applied through either path is seen by both, because there is only one module
object. Wave 3 removes this file; nothing new should import it.
"""
import sys

from . import navigation as _navigation

sys.modules[__name__] = _navigation
