from ...registry import _import_env_module

# Isolate optional import failures; each is reported under its own key.
_import_env_module("craftax.classic", "craftax")
_import_env_module("craftax.full", "craftax_full")
_import_env_module("craftax.semantics", "craftax_semantics")
