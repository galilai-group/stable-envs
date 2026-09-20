"""Private home of the cross-package glue. The public module is ``stable_envs/interop.py``.

Public interop lives in ``interop.py``. The material that had to move — the SWM facade, the
batched pool, ``swap_renderer`` and the four variation-map tables — is ~2,450 lines, and one module of
that size is not the "small, obvious module" the same plan asks for, so it lives in a PRIVATE package
and ``stable_envs/interop.py`` is the single public face of it. Nothing outside imports
``stable_envs._interop`` directly; the leading underscore says so.
"""

from __future__ import annotations
