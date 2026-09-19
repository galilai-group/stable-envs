"""Package identity, in one place, so nothing has to import the package to learn its version.

``stable_envs/__init__.py`` re-exports ``__version__`` from here and packaging metadata reads it from
this file, which is the arrangement the sibling ``stable-*`` repositories use: a build backend can read
a literal out of a two-line module without importing torch.
"""

from __future__ import annotations

__title__ = "stable-envs"
__version__ = "0.1.0"
__summary__ = "Famous research environments as GPU-batched, state-driven renderers."
__author__ = "Randall Balestriero, Sami BuGhanem"
__license__ = "MIT"

__all__ = ["__title__", "__version__", "__summary__", "__author__", "__license__"]
