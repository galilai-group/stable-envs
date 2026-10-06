from __future__ import annotations

from typing import Any

from loguru import logger as _loguru_logger

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
_CONFIG: dict[str, Any] = dict(_DEFAULTS)


def set(**kwargs: Any) -> dict[str, Any]:  # noqa: A001 - the public spelling is ``snv.set``
    """Update global verbosity or default device and return a configuration copy.

    Unknown keys and unrecognized verbosity levels raise an error. Environment
    options such as image size belong on the environment constructor.
    """
    unknown = [k for k in kwargs if k not in _DEFAULTS]
    if unknown:
        raise KeyError(
            f"stable_envs.set: unknown config key(s) {unknown}. The whole configuration is "
            f"{sorted(_DEFAULTS)}; anything else belongs on the environment itself (device, size, "
            f"supersample and backend are constructor arguments)."
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
    """Return a copy of the current global settings."""
    return dict(_CONFIG)


def reset_config() -> dict[str, Any]:
    """Restore and return the default global settings."""
    _CONFIG.clear()
    _CONFIG.update(_DEFAULTS)
    return dict(_CONFIG)


def log(level: str, message: str) -> None:
    if _LEVELS[level.upper()] >= _LEVELS[_CONFIG["verbose"]]:
        logger.opt(depth=1).log(level.upper(), message)
