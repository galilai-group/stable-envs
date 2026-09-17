"""Compatibility shim. The real modules are ``stable_envs.env``, ``stable_envs.registry``.

Step 9 moved the package's vocabulary to the top level — ``stable_envs/env.py``, ``dynamics.py``,
``specs.py``, ``registry.py``, ``_config.py`` — which is where a contributor coming from
``stable-pretraining`` or ``stable-worldmodel`` looks for it, and which removes a ``core/`` package
that held two files and named nothing (plan §16). ``stable_envs.core.env`` and
``stable_envs.core.style`` keep working, re-exporting the same objects, until Workstream K migrates
the ~470 call sites; they are removed in Wave 3, never before the oracle comparison is green.
"""
from __future__ import annotations
