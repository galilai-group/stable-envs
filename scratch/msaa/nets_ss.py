"""Correctors for the SAME-RESOLUTION problem: ``(B, N, N, 3) uint8`` at ss=1 -> the same shape at ss=2.

The two trunks are ``rendernet.nets``'s, imported rather than copied, because the architecture
question was already swept there and re-answering it would be re-doing measured work. What changes is
the wrapper, and the change is not cosmetic:

* ``rendernet``'s ``forward_float`` upsamples the input to a fixed ``out_size`` and adds a residual.
  Here input and output are the same size, so the "bilinear skip" is an IDENTITY -- the model at
  initialisation is not an approximation of the target, it is the ss=1 frame byte for byte. Every
  number is then a departure from a point that is itself one of the rows in the table.
* The wrapper is RESOLUTION-AGNOSTIC. Nothing is resized, so one checkpoint runs at 84, 240 or 256 px
  and the size lives in the data rather than in the architecture. (It is still trained per size: the
  residual is edge-localised and an edge is a different width in pixels at 84 px than at 256 px.)
* ``shrink`` is promoted from an experiment to the centrepiece. These are PARITY cells: interior
  pixels must be bit-identical to ``mujoco.Renderer`` and a residual of 0.2/255 spread over a flat
  surface destroys every one of them while improving nothing. Soft-shrinking the residual by half a
  level makes "the corrector has nothing to say here" produce a bit-identical pixel by construction,
  which is the only way an interior-exact claim can survive a learned decoder at all.

``unshuffle`` is what makes the budget work. At 256 px a 3x3 conv over C=32 channels moves 1024 x 256 x
256 x 32 x 2 bytes per layer; folding 4x of resolution into channels leaves the arithmetic identical
and cuts the traffic 16x, which is the difference between 20 us and 200 us a frame.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "altrender", "rendernet"))

import torch
import torch.nn as nn
import torch.nn.functional as F

import nets as RN  # noqa: E402  (rendernet's trunks: ESPCN, UNetTiny)


class SameSize(nn.Module):
    """Wraps one of rendernet's trunks so that output size == input size, at any input size.

    The trunk is built with ``in_size == out_size``, so its total ``PixelShuffle`` factor equals its
    ``PixelUnshuffle`` factor and the two cancel; nothing in either class depends on the absolute
    resolution, only on divisibility by ``unshuffle``.
    """

    def __init__(self, kind: str, width: int, depth: int, unshuffle: int, in_ch: int = 3):
        super().__init__()
        cls = {"espcn": RN.ESPCN, "unet": RN.UNetTiny}[kind]
        # in_size == out_size: any value works, it only sets the (cancelling) shuffle factors.
        self.trunk = cls(in_size=256, out_size=256, width=width, depth=depth, unshuffle=unshuffle,
                         in_ch=in_ch, pre_size=0, coord=False)
        self.unshuffle = int(unshuffle)
        self.kind, self.width, self.depth = kind, width, depth

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def residual_float(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk.residual(x)

    def forward_float(self, x: torch.Tensor, shrink: float = 0.0) -> torch.Tensor:
        """float NCHW in [0,1] -> float NCHW in [0,1]-ish, same spatial size. The training path."""
        res = self.residual_float(x)
        if shrink > 0:
            res = F.softshrink(res, shrink / 255.0)
        return x + res

    def forward(self, u8: torch.Tensor, shrink: float = 0.0) -> torch.Tensor:
        """uint8 NHWC -> uint8 NHWC. The tensor a consumer would actually receive.

        The unpack, the add and the repack all stay in the model's dtype: fp16 represents every
        integer to 2048 exactly, so the quantisation is exact, and promoting to fp32 first
        materialises a (B, 3, 256, 256) float tensor for nothing -- measured at 3.3 us/frame in
        ``rendernet``'s speed study, 18 % of that budget.
        """
        x = u8.permute(0, 3, 1, 2).to(self.dtype).mul_(1.0 / 255.0)
        x = x.contiguous(memory_format=torch.channels_last)
        y = self.forward_float(x, shrink).mul_(255.0).round_().clamp_(0, 255)
        return y.to(torch.uint8).permute(0, 2, 3, 1)


#: name -> (kind, width, depth, unshuffle). ``u4`` runs the trunk at a quarter of the output edge,
#: which is where the throughput is; ``u2`` is four times the spatial work and is here as the
#: capacity end of the sweep. ``u1`` is full resolution and is included on the 84 px cells only,
#: where the traffic is 9x smaller.
ZOO = {
    "espcn_w32_d4_u4":  ("espcn", 32, 4, 4),
    "espcn_w48_d4_u4":  ("espcn", 48, 4, 4),
    "espcn_w64_d4_u4":  ("espcn", 64, 4, 4),
    "espcn_w64_d6_u4":  ("espcn", 64, 6, 4),
    "espcn_w96_d6_u4":  ("espcn", 96, 6, 4),
    "espcn_w128_d6_u4": ("espcn", 128, 6, 4),
    "espcn_w32_d4_u2":  ("espcn", 32, 4, 2),
    "espcn_w64_d4_u2":  ("espcn", 64, 4, 2),
    "espcn_w64_d6_u2":  ("espcn", 64, 6, 2),
    "espcn_w96_d6_u2":  ("espcn", 96, 6, 2),
    "unet_w48_d2_u4":   ("unet", 48, 2, 4),
    "unet_w64_d3_u4":   ("unet", 64, 3, 4),
    "unet_w96_d3_u4":   ("unet", 96, 3, 4),
    "unet_w48_d2_u2":   ("unet", 48, 2, 2),
    "unet_w64_d3_u2":   ("unet", 64, 3, 2),
}


def compatible(name: str, size: int) -> bool:
    """Can this architecture run at this frame size at all?

    ESPCN needs ``size % unshuffle == 0``; the UNet needs another factor of 2 because its down path
    halves the already-unshuffled grid and the ``PixelShuffle(2)`` on the way back up has to land on
    the same edge as the skip. It does NOT raise on an odd grid, it raises on the ``torch.cat``, one
    layer later and with a shape error that names neither the size nor the architecture -- so the
    check is here, where it can be asked before anything is built. 84 px with ``u4`` is the case that
    hits it: 84/4 = 21, 21 -> 11 -> 22, and 22 != 21.
    """
    kind, _, _, u = ZOO[name]
    return int(size) % (u * (2 if kind == "unet" else 1)) == 0


def build(name: str, in_ch: int = 3) -> SameSize:
    kind, w, d, u = ZOO[name]
    return SameSize(kind, w, d, u, in_ch)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def gflops_per_frame(name: str, size: int = 256) -> float:
    """Multiply-accumulates x2 for one frame, counted from the ACTUAL output shapes of a forward.

    From a hook rather than a formula: the UNet's up path is exactly the kind of thing a hand-derived
    count gets wrong by 2x with nothing complaining.
    """
    m = build(name)
    total = [0]

    def hook(mod, inp, out):
        total[0] += (mod.in_channels // mod.groups * mod.kernel_size[0] * mod.kernel_size[1]
                     * out.numel() / out.shape[0])

    hs = [mod.register_forward_hook(hook) for mod in m.modules() if isinstance(mod, nn.Conv2d)]
    with torch.no_grad():
        m(torch.zeros(1, size, size, 3, dtype=torch.uint8))
    for h in hs:
        h.remove()
    return 2 * total[0] / 1e9
