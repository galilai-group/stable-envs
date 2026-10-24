"""DM Control classes combining renderers with simulation adapters."""

import importlib

from ...registry import _import_env_module

# The domain classes need dm_control; isolate that failure so the simulation subpackage stays importable.
_import_env_module("dmcontrol._family", "dmcontrol")


def __getattr__(name):
    if name.startswith("__") or name == "_family":
        raise AttributeError(name)
    return getattr(importlib.import_module(f"{__name__}._family"), name)
