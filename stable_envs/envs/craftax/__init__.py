from ...registry import _import_env_module

# Keep optional import failures isolated and report them under their existing keys.
_import_env_module("craftax.classic", "craftax")
_import_env_module("craftax.full", "craftax_full")
_import_env_module("craftax.semantics", "craftax_semantics")
