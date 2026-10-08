import sys

from ...registry import _import_env_module

# Keep optional import failures isolated and report them under their existing keys.
# Order matters: ``legacy`` registers its own PushT first, then ``semantics`` replaces it with the
# public class. Each module is imported once per process, so a later ``import ...pusht.legacy``
# cannot register again over the public class.
_import_env_module("pusht.legacy", "pusht")
_import_env_module("pusht.semantics", "pusht_semantics")

_semantics = sys.modules.get(f"{__name__}.semantics")
if _semantics is not None:
    PushT = _semantics.PushT
    PushTDiscrete = _semantics.PushTDiscrete
