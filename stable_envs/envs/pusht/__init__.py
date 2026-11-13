"""PushT: push a T-shaped block into a target pose, with continuous or discrete actions (``pusht`` extra)."""

import sys

from ...registry import _import_env_module

# Optional import failures are isolated and reported under these keys.
# Order matters: ``legacy`` registers its PushT first, then ``semantics`` replaces it with the
# public class. Each module is imported once per process, so a later ``import ...pusht.legacy``
# cannot register over the public class again.
_import_env_module("pusht.legacy", "pusht")
_import_env_module("pusht.semantics", "pusht_semantics")

_semantics = sys.modules.get(f"{__name__}.semantics")
if _semantics is not None:
    PushT = _semantics.PushT
    PushTDiscrete = _semantics.PushTDiscrete
