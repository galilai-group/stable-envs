"""Batched torch rendering for the public PushT, pixel-identical to its pygame renderer.

Not part of the released package; see ``contrib/pusht_torch/README.md``.
"""

from .renderer import TorchRenderer

__all__ = ["TorchRenderer"]
