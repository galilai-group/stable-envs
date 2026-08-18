"""Adapters that let other libraries consume a stable-envs renderer without knowing it is one.

A stable-envs env is a BATCHED renderer, not a ``gym.Env``: it has no ``reset``, its ``step`` takes
``(B, action_dim)``, and ``render`` is a pure function of a state stack. That is the right shape for
the thing it does and the wrong shape for every training loop already written against gymnasium. The
modules here are the translation, and they are kept out of the core on purpose — nothing in
``stable_envs.core`` may depend on another framework being installed.

    worldmodel   stable_worldmodel: a single-env gymnasium facade, a batched EnvPool replacement,
                 and the variation-space <-> StyleTensor mapping that lets an existing swm config
                 drive our renderer unchanged.
"""
from . import worldmodel  # noqa: F401

__all__ = ["worldmodel"]
