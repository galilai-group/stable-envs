"""Renderer-only ports of environment families that are not part of the released package.

These modules re-render other projects' environments from a state vector. They are not checked
against their upstream State, most of them cannot ``step()``, and they are not shipped in the
``stable-envs`` wheel or sdist. They are kept in the repository for in-lab use; see
``contrib/README.md``.

Importing one module (``import contrib.minigrid``) registers its classes in
``stable_envs.registry.REGISTRY``. ``load()`` imports all of them and returns the import failures,
keyed by module name, the same way ``stable_envs.import_errors()`` reports the packaged families.
"""

from __future__ import annotations

import importlib

from stable_envs._core.config import log as _log

#: Every contrib environment module, by the module name its import failures are recorded under.
MODULES = (
    "carracing",
    "crafter",
    "griddly",
    "highway",
    "humanoidbench",
    "jumanji",
    "lunarlander",
    "maze2d",
    "metaworld",
    "minigrid",
    "minihack",
    "mujoco_locomotion",
    "overcooked",
    "pgx",
    "procgen",
    "robocasa",
    "sokoban",
)

#: module name -> the exception that stopped it importing
_IMPORT_ERRORS: dict[str, str] = {}


def _import_module(mod_name: str, error_key: str) -> None:
    """Import one contrib module and record a failure, as ``stable_envs.registry._import_env_module`` does."""
    try:
        importlib.import_module(f"contrib.{mod_name}")
    except (ImportError, ModuleNotFoundError) as e:
        _IMPORT_ERRORS[error_key] = f"{type(e).__name__}: {e}"
        _log("DEBUG", f"contrib.{mod_name} did not import (missing dependency): {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001 — recorded, never fatal
        _IMPORT_ERRORS[error_key] = f"{type(e).__name__}: {e} (unexpected; this is not an optional-dependency miss)"
        _log(
            "DEBUG",
            f"contrib.{mod_name} did not import (unexpected, not an optional-dep miss): {type(e).__name__}: {e}",
        )


def load() -> dict[str, str]:
    """Import every contrib module, registering its classes, and return ``{module: error}`` for failures."""
    from stable_envs._core.mujoco_gl import configure_mujoco_gl

    configure_mujoco_gl()
    for name in MODULES:
        _import_module(name, name)
    return dict(_IMPORT_ERRORS)


def import_errors() -> dict[str, str]:
    """Return a copy of the module-name to import-error mapping recorded by ``load()``."""
    return dict(_IMPORT_ERRORS)


__all__ = ["MODULES", "import_errors", "load"]
