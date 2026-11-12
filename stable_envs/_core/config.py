"""Global verbosity and default-device settings, and the package logger."""

from __future__ import annotations

from typing import Any

from loguru import logger as _loguru_logger

from . import environ as _environ

# Records carry extra["package"] == "stable_envs" for application-side filtering.
logger = _loguru_logger.bind(package="stable_envs")

_LEVELS = {
    "TRACE": 5,
    "DEBUG": 10,
    "INFO": 20,
    "SUCCESS": 25,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
    "SILENT": 100,
}

_DEFAULTS: dict[str, Any] = {"verbose": "WARNING", "default_device": "auto"}


def _initial() -> dict[str, Any]:
    return {**_DEFAULTS, "verbose": _environ.read("STABLE_ENVS_VERBOSE")}


_CONFIG: dict[str, Any] = _initial()


def set(**kwargs: Any) -> dict[str, Any]:  # noqa: A001 - the public spelling is ``snv.set``
    """Update the log level or default device and return a copy of the settings.

    Args:
        **kwargs: ``verbose``, the lowest level logged (``"TRACE"``, ``"DEBUG"``, ``"INFO"``,
            ``"SUCCESS"``, ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"`` or ``"SILENT"``), and
            ``default_device``, used when a constructor gets ``device=None``.

    Raises:
        KeyError: If any other key is passed.
        ValueError: If ``verbose`` is not a known level.
    """
    unknown = [k for k in kwargs if k not in _DEFAULTS]
    if unknown:
        raise KeyError(
            f"stable_envs.set: unknown config key(s) {unknown}; the keys are {sorted(_DEFAULTS)}. "
            f"device, size, supersample and backend are environment constructor arguments."
        )
    if "verbose" in kwargs:
        v = str(kwargs["verbose"]).upper()
        if v not in _LEVELS:
            raise ValueError(f"stable_envs.set: verbose must be one of {sorted(_LEVELS)}, got {kwargs['verbose']!r}")
        kwargs["verbose"] = v
    if "default_device" in kwargs and kwargs["default_device"] is not None:
        kwargs["default_device"] = str(kwargs["default_device"])
    _CONFIG.update(kwargs)
    return dict(_CONFIG)


def get_config() -> dict[str, Any]:
    """Return a copy of the global settings, ``verbose`` and ``default_device``."""
    return dict(_CONFIG)


def reset_config() -> dict[str, Any]:
    """Restore and return the default global settings, with verbosity from ``STABLE_ENVS_VERBOSE``."""
    _CONFIG.clear()
    _CONFIG.update(_initial())
    return dict(_CONFIG)


def log(level: str, message: str) -> None:
    """Log ``message`` through the package logger when ``level`` meets the configured verbosity.

    Args:
        level: A level name such as ``"DEBUG"`` or ``"WARNING"``.
        message: The text to log.
    """
    if _LEVELS[level.upper()] >= _LEVELS[_CONFIG["verbose"]]:
        logger.opt(depth=1).log(level.upper(), message)
