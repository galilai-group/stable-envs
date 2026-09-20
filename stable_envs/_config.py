"""Global configuration and logging — deliberately three keys and one logger.

``snv.set(verbose="INFO")`` / ``snv.get_config()`` mirror the sibling ``stable-*`` packages. The
temptation with a config object is to let it grow into a second way of configuring an environment; it
is kept to the three keys that genuinely cannot live on an env instance:

  * ``verbose`` — the level at or above which this package emits log records;
  * ``default_device`` — the device an env gets when the caller names none (``None`` keeps the
    historical default, which every env spells itself);
  * ``warn_on_deprecated`` — whether the deprecated aliases announce themselves.

THE LOGGER DOES NOT TOUCH GLOBAL LOGURU STATE. ``loguru.logger`` is a process-wide singleton owned by
the application, so adding or removing handlers here would silently re-configure logging for a program
that merely imported an environment. Instead every record this package emits is filtered against
``verbose`` before it is handed to loguru, and the record carries ``extra={"package": "stable_envs"}``
so an application that does want its own sink can filter for exactly ours.
"""

from __future__ import annotations

from typing import Any

from loguru import logger as _loguru_logger

#: Every record from this package carries ``extra["package"] == "stable_envs"``, so an application can
#: route or silence stable-envs alone: ``logger.add(sink, filter=lambda r: r["extra"].get("package")
#: == "stable_envs")``.
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

_DEFAULTS: dict[str, Any] = {"verbose": "WARNING", "default_device": None, "warn_on_deprecated": True}
_CONFIG: dict[str, Any] = dict(_DEFAULTS)

#: Deprecation keys already announced. One warning per alias per process: a repository with ~470 call
#: sites on the old names would otherwise print ~470 identical lines, which trains the reader to ignore
#: them.
_ANNOUNCED: set[str] = set()


def set(**kwargs: Any) -> dict[str, Any]:  # noqa: A001 - the public spelling is ``snv.set``
    """``snv.set(verbose="INFO")``. Unknown keys are an error, not a silently stored typo."""
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
    if "warn_on_deprecated" in kwargs:
        kwargs["warn_on_deprecated"] = bool(kwargs["warn_on_deprecated"])
    _CONFIG.update(kwargs)
    return dict(_CONFIG)


def get_config() -> dict[str, Any]:
    """The current configuration, as a copy — mutating the return value configures nothing."""
    return dict(_CONFIG)


def reset_config() -> dict[str, Any]:
    """Back to the defaults, and forget which deprecations have been announced. Used by the tests."""
    _CONFIG.clear()
    _CONFIG.update(_DEFAULTS)
    _ANNOUNCED.clear()
    return dict(_CONFIG)


def log(level: str, message: str) -> None:
    """Emit one record at ``level`` if ``verbose`` allows it. The only logging entry point."""
    if _LEVELS[level.upper()] >= _LEVELS[_CONFIG["verbose"]]:
        logger.opt(depth=1).log(level.upper(), message)


def deprecated(key: str, message: str) -> None:
    """Announce a deprecated alias ONCE per process, under ``warn_on_deprecated``."""
    if not _CONFIG["warn_on_deprecated"] or key in _ANNOUNCED:
        return
    _ANNOUNCED.add(key)
    log("WARNING", message)


def announced() -> set[str]:
    """Which deprecation keys have already been announced. For tests and for debugging.

    Spelled as a set literal because this module defines a function named ``set`` (the public
    ``snv.set``), which shadows the builtin inside it.
    """
    return {*_ANNOUNCED}
