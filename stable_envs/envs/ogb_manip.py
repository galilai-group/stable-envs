"""Deprecated import path. ``stable_envs.envs.ogb_manip`` IS ``stable_envs.envs.ogbench.manipulation``.

Step 9 collected the OGBench family into one package (plan §3): the mazes are
``stable_envs.envs.ogbench.maze`` and the two ``stable_worldmodel`` manipulation cells are
``stable_envs.envs.ogbench.manipulation``. Nothing claims the name ``ogb_manip``, so this file stays
as a ``sys.modules`` ALIAS rather than a set of re-exports, which is what Workstream A's ``swm_nav``
shim does and what the module-level facts of this family need: ``tests/unit/test_ogb_manip.py`` and
``tests/unit/test_variation_maps_newcells.py`` reach for ``_bundle``, ``_reference_env``,
``_button_tables``, ``_appearance_digest``, ``_context`` (by ``inspect.getsource``), ``SPEC``,
``VISIBLE_GROUPS``, ``IDENTITY_QUAT``, ``MJW_CHUNK`` and ``OGBManipBase``, and there is exactly ONE
module object, so every one of them resolves and a monkeypatch applied through either path is seen by
both.

``OGBCube.__module__`` and ``OGBScene.__module__`` are still this path, which is what keeps the
registry row, ``spec(name)["family"]`` and ``list_envs(family="ogb_manip")`` byte-identical to
Step 8; see ``manipulation._MODULE``. Wave 3 removes this file and moves the family name with it.
Nothing new should import this path.
"""
import sys

from .ogbench import manipulation as _manipulation

sys.modules[__name__] = _manipulation
