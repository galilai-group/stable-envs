from ...registry import _import_env_module

# Keep optional import failures isolated and report them under their existing keys.
_import_env_module("rocket.legacy", "rocket_landing")
_import_env_module("rocket.render", "rocket_render")
_import_env_module("rocket.semantics", "rocket_semantics")


def __getattr__(name: str):
    # The public class needs the PyFlyt extra, so it is resolved on first use instead of at package import.
    if name == "PyFlytRocketLanding":
        from .semantics import PyFlytRocketLanding

        return PyFlytRocketLanding
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
