"""Re-exports of the Step-8 names that used to live here. See ``stable_envs/env.py``.

Everything below is THE SAME OBJECT the new module defines — ``stable_envs.core.env.StableEnv is
stable_envs.Env`` and ``stable_envs.core.env.REGISTRY is stable_envs.registry.REGISTRY`` — so a test
that mutates the registry through this path mutates the one registry, as it always did.
"""
from __future__ import annotations

from ..env import (FIDELITY_TIERS, MEASURED_WITH_KEY, REFERENCE_VERSIONS_KEY,  # noqa: F401
                   REQUIRED_VISUAL_FIDELITY, RENDER_DECODERS, Env, State, _as_state_tensor,
                   _guard_render_out, _names_phrase, _nonfinite_report, check_states,
                   reference_versions_of, render_states_arg, require_contiguous_out)
from ..registry import (REGISTRY, check_decoder_declaration,  # noqa: F401
                        check_fidelity_declaration, format_registry, make, register, registry_table)

#: The Step-8 name for :class:`stable_envs.Env`. Not a subclass: the same class object, so
#: ``isinstance``, ``issubclass`` and every ``__init_subclass__`` hook behave identically.
StableEnv = Env

__all__ = ["FIDELITY_TIERS", "MEASURED_WITH_KEY", "REFERENCE_VERSIONS_KEY",
           "REQUIRED_VISUAL_FIDELITY", "RENDER_DECODERS", "REGISTRY", "Env", "StableEnv", "State",
           "check_decoder_declaration", "check_fidelity_declaration", "check_states",
           "format_registry", "make", "reference_versions_of", "register", "registry_table",
           "render_states_arg", "require_contiguous_out"]
