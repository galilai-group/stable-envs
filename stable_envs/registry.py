"""The environment registry: registration checks, lookup, metadata and family discovery."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from ._core.config import log as _log
from .env import (
    FIDELITY_TIERS,
    MEASURED_WITH_KEY,
    REFERENCE_VERSIONS_KEY,
    REQUIRED_VISUAL_FIDELITY,
    Env,
    _check_reference_versions_shape,
)

#: Canonical Python class names are registry keys.
REGISTRY: dict[str, type[Env]] = {}


def check_fidelity_declaration(cls: type[Env]) -> None:
    """Validate the declared fidelity tier and its required metadata.

    Visual classes must include VISUAL TIER in their class docstring. This checks
    declaration completeness, not whether the measured acceptance gates pass.
    """
    tier = getattr(cls, "fidelity_tier", "parity")
    if tier not in FIDELITY_TIERS:
        raise ValueError(f"{cls.__name__}: fidelity_tier must be one of {FIDELITY_TIERS}, got {tier!r}")
    vf = dict(getattr(cls, "visual_fidelity", {}) or {})
    # Version metadata is optional, but malformed declarations cannot identify a reference.
    if REFERENCE_VERSIONS_KEY in cls.__dict__ or getattr(cls, REFERENCE_VERSIONS_KEY, None):
        _check_reference_versions_shape(cls, REFERENCE_VERSIONS_KEY, getattr(cls, REFERENCE_VERSIONS_KEY))
    if REFERENCE_VERSIONS_KEY in vf:
        _check_reference_versions_shape(
            cls, f"visual_fidelity[{REFERENCE_VERSIONS_KEY!r}]", vf[REFERENCE_VERSIONS_KEY]
        )
    if MEASURED_WITH_KEY in vf:
        _check_reference_versions_shape(cls, f"visual_fidelity[{MEASURED_WITH_KEY!r}]", vf[MEASURED_WITH_KEY])
    if tier == "parity":
        if vf:
            raise ValueError(
                f"{cls.__name__} declares fidelity_tier='parity' and also carries visual_fidelity; "
                f"drop the dict or declare the visual tier."
            )
        return
    missing = [k for k in REQUIRED_VISUAL_FIDELITY if k not in vf]
    if missing:
        raise ValueError(
            f"{cls.__name__} declares the VISUAL tier without {missing}; a visual declaration "
            f"carries its measured agreement and acceptance limits."
        )
    if not getattr(cls, "visual_note", ""):
        raise ValueError(
            f"{cls.__name__} declares the VISUAL tier with no visual_note; the note states the "
            f"residual difference and what the frames must not be used for."
        )
    if getattr(cls, "parity_interior_min", None) is not None:
        raise ValueError(
            f"{cls.__name__} declares the VISUAL tier but leaves parity_interior_min at "
            f"{cls.parity_interior_min!r}. Set it to None: the parity gate does not apply to this env."
        )
    doc = (cls.__doc__ or "").upper()
    if "VISUAL TIER" not in doc:
        raise ValueError(
            f"{cls.__name__} declares the VISUAL tier and does not say so in its class docstring. "
            f"Put the words 'VISUAL TIER' in it, with what the residual is."
        )


def register(cls: type[Env]) -> type[Env]:
    """Validate ``cls`` and register it under its class name; usable as a class decorator.

    ``family`` must be set on the class. An unset ``domain`` defaults to ``DOMAIN`` or the family.
    A different class with the same name raises ValueError.
    """
    key = cls.__name__
    if key in REGISTRY and REGISTRY[key] is not cls:
        raise ValueError(
            f"env name collision: {key} is already registered by "
            f"{REGISTRY[key].__module__}; two classes cannot share a lookup key"
        )
    family = str(getattr(cls, "family", "") or "")
    if not family:
        raise ValueError(
            f"{key}: set family= as an explicit class attribute; it is not inferred from the "
            "module path. Example: family = 'dmcontrol'."
        )
    domain = str(getattr(cls, "domain", "") or "")
    if not domain and getattr(cls, "DOMAIN", ""):
        cls.domain = str(cls.DOMAIN)
    elif not domain:
        cls.domain = family
    check_fidelity_declaration(cls)
    REGISTRY[key] = cls
    return cls


def registry_table() -> list[dict[str, Any]]:
    """Return one metadata row per registered class, sorted by name.

    ``render_state_dim`` is None when the width is only known on an instance.
    """
    return [
        {
            "class_name": n,
            "name": n,
            "tier": getattr(c, "fidelity_tier", "parity"),
            "family": getattr(c, "family", "") or None,
            "domain": getattr(c, "domain", "") or None,
            "render_state_dim": (int(c.state_dim) or None) if isinstance(getattr(c, "state_dim", None), int) else None,
            "reference_id": getattr(c, "reference_id", None),
            "module": c.__module__,
        }
        for n, c in sorted(REGISTRY.items())
    ]


def format_registry(width: int = 34) -> str:
    """Return a text listing of registered classes with visual-tier classes marked."""
    rows = registry_table()
    n_visual = sum(r["tier"] == "visual" for r in rows)
    out = [f"{len(rows)} environments; {len(rows) - n_visual} parity, {n_visual} visual"]
    for r in rows:
        mark = "   [VISUAL TIER — not the official pixels]" if r["tier"] == "visual" else ""
        out.append(f"  {r['name']:<{width}}{mark}")
    if n_visual:
        out.append(
            "  visual tier: fast re-render of the same scene by a different rasterizer. Good "
            "for live training data; NOT a substitute for a stored dataset when a policy will "
            "be evaluated on official pixels. See each env's visual_fidelity / visual_note."
        )
    return "\n".join(out)


def _resolve(name: str) -> type[Env]:
    cls = REGISTRY.get(name)
    if cls is not None:
        return cls
    raise KeyError(f"unknown env {name!r}; known: {sorted(REGISTRY)}")


def make(name: str, **kwargs) -> Env:
    """Construct an environment by canonical class name, forwarding constructor keywords."""
    return _resolve(name)(**kwargs)


def info(name: str) -> dict[str, Any]:
    """Return class metadata for ``name`` without constructing an environment.

    Dimensions that depend on the instance are None; full specs come from a constructed
    environment. Unknown names raise KeyError.
    """
    cls = _resolve(name)
    dim = getattr(cls, "state_dim", None)
    render_dim = int(dim) if isinstance(dim, int) and dim else None
    return {
        "class_name": cls.__name__,
        "id": getattr(cls, "name", "") or cls.__name__,
        "class": cls,
        "module": cls.__module__,
        "family": getattr(cls, "family", "") or None,
        "domain": getattr(cls, "domain", "") or None,
        "render_state_dim": render_dim,
        "render_state_names": list(getattr(cls, "state_names", ()) or ()),
        "default_size": getattr(cls, "default_size", None),
        "reference_id": getattr(cls, "reference_id", None),
        "fidelity_tier": getattr(cls, "fidelity_tier", "parity"),
        "action_dim": getattr(cls, "action_dim", None),
        "has_dynamics": getattr(cls, "_step", None) is not Env._step,
        "requires_cuda": bool(getattr(cls, "requires_cuda", False)),
    }


def list_envs(family: str | None = None, domain: str | None = None) -> list[str]:
    """Return sorted canonical class names, optionally filtered by family or domain."""
    names = sorted(REGISTRY)
    if family is not None:
        names = [n for n in names if REGISTRY[n].family == family]
    if domain is not None:
        names = [n for n in names if REGISTRY[n].domain == domain]
    return names


def list_families() -> list[str]:
    """Return sorted family IDs with at least one registered class."""
    return sorted({c.family for c in REGISTRY.values() if c.family})


#: module name -> the exception that stopped it importing
_IMPORT_ERRORS: dict[str, str] = {}
_DISCOVERED = False


def _import_env_module(mod_name: str, error_key: str) -> None:
    """Import a family module, recording an ImportError or any other exception under ``error_key``."""
    try:
        importlib.import_module(f"stable_envs.envs.{mod_name}")
    except (ImportError, ModuleNotFoundError) as e:
        _IMPORT_ERRORS[error_key] = f"{type(e).__name__}: {e}"
        _log("DEBUG", f"stable_envs.envs.{mod_name} did not import (missing extra): {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001 — recorded, never fatal
        _IMPORT_ERRORS[error_key] = f"{type(e).__name__}: {e} (unexpected; this is not an optional-dependency miss)"
        _log(
            "WARNING",
            f"stable_envs.envs.{mod_name} did not import (unexpected, not an "
            f"optional-dep miss): {type(e).__name__}: {e}",
        )


def discover(force: bool = False) -> dict[str, str]:
    """Import every public module of ``stable_envs.envs`` and return the recorded import failures.

    A failed family leaves the others available. ``force`` repeats discovery without
    clearing registered classes.
    """
    global _DISCOVERED
    if _DISCOVERED and not force:
        return dict(_IMPORT_ERRORS)
    from ._core.mujoco_gl import configure_mujoco_gl

    configure_mujoco_gl()
    pkg = importlib.import_module("stable_envs.envs")
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name.startswith("_"):
            continue
        _import_env_module(mod.name, mod.name)
    _DISCOVERED = True
    return dict(_IMPORT_ERRORS)


def import_errors() -> dict[str, str]:
    """Return ``{module name: error}`` for the environment modules that did not import."""
    return dict(_IMPORT_ERRORS)


__all__ = [
    "REGISTRY",
    "check_fidelity_declaration",
    "discover",
    "format_registry",
    "import_errors",
    "info",
    "list_envs",
    "list_families",
    "make",
    "register",
    "registry_table",
]
