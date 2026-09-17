"""The lit-surface law both tiers evaluate, and the light set it reads.

``_Lights`` is where the HEADLIGHT exists at all: it is not in the model, MuJoCo adds it when it
builds an ``mjvScene``, and a renderer that read ``m.light_*`` alone would be drawing the scene under
the wrong lights. ``_shade`` is the same Phong term the reference's fixed-function pipeline
evaluates, per pixel instead of per vertex, and ``SPECULAR_SCALE`` is the one constant in it that was
measured off the reference's own quantised colours rather than taken from the material.

Shared by both tiers: ``suite`` calls ``_shade`` for the mover pass and the reflection pass, and the
CPU trackcom backend evaluates the same law inside ``stable_envs.render.cpu3d``.
"""
from __future__ import annotations

import numpy as np
import torch

from ._model import LIGHT_DIRECTIONAL, SPECULAR_SCALE


class _Lights:
    """The lights as ``mjvScene`` reports them, which is the only place the headlight exists.

    Nothing is folded together. The headlight is directional along the camera axis with
    ambient .4 / diffuse .8 / specular .1 and casts no shadow; the model light is a spotlight
    (cutoff 45 degrees, exponent 10) with ambient 0 / diffuse .7 / specular .3 that does. Summing them
    into one direction is legitimate for OGBench, where both point straight down, and would erase the
    spotlight falloff that IS the picture in point_mass.
    """

    __slots__ = ("kind", "pos", "dir", "ambient", "diffuse", "specular", "attenuation", "cutoff",
                 "exponent", "castshadow")

    def __init__(self, scene):
        n = int(scene.nlight)
        g = lambda f: np.stack([np.asarray(getattr(scene.lights[i], f), np.float64)  # noqa: E731
                                for i in range(n)])
        self.pos, self.dir = g("pos"), g("dir")
        self.ambient, self.diffuse, self.specular = g("ambient"), g("diffuse"), g("specular")
        self.attenuation = g("attenuation")
        self.cutoff = np.array([float(scene.lights[i].cutoff) for i in range(n)])
        self.exponent = np.array([float(scene.lights[i].exponent) for i in range(n)])
        self.castshadow = np.array([int(scene.lights[i].castshadow) for i in range(n)])
        self.kind = np.array([int(getattr(scene.lights[i], "type",
                                          getattr(scene.lights[i], "directional", 1)))
                              for i in range(n)])

    def shadow_light(self) -> int:
        idx = np.flatnonzero(self.castshadow > 0)
        return int(idx[0]) if len(idx) else -1

    def to(self, device) -> dict:
        t = lambda x: torch.tensor(np.asarray(x, np.float64), device=device,          # noqa: E731
                                   dtype=torch.float32)
        return dict(pos=t(self.pos), dir=t(self.dir), ambient=t(self.ambient),
                    diffuse=t(self.diffuse), specular=t(self.specular),
                    attenuation=t(self.attenuation), cutoff=[float(c) for c in self.cutoff],
                    exponent=[float(e) for e in self.exponent],
                    kind=[int(k) for k in self.kind],
                    castshadow=[bool(c) for c in self.castshadow])


class _Scene:
    """Everything a render needs that does not depend on the state."""

    __slots__ = ("plate", "plate_shadow", "plate_norefl", "reflectance", "static_t", "cam", "model",
                 "data", "movers", "statics", "size", "lights", "prims", "shadow_light", "floor",
                 "eps", "tendon", "shadow_candidates", "rays", "const")


# --------------------------------------------------------------------------- shading

def _norm(v):
    return v / torch.clamp(torch.sqrt((v * v).sum(-1, keepdim=True)), min=1e-12)


def _dot(a, b):
    """Unrolled for the reason ``raycast3d._dot`` is: a reduction whose kernel is chosen from the
    batch count sums its terms in a batch-dependent order, and one ulp there is a hit/miss flip."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _light_geometry(L: dict, i: int, p: torch.Tensor):
    """``(toward_light, attenuation)`` for light ``i`` at points ``p``.

    OpenGL fixed function, which is what ``mjr_render`` drives: a positional light attenuates by
    ``1/(k0 + k1 d + k2 d^2)`` and, inside its cutoff, by ``(spotdir . -l)^exponent``; a directional
    light does neither and has no position at all.
    """
    shape = (1,) * (p.dim() - 1) + (3,)
    if L["kind"][i] == LIGHT_DIRECTIONAL:
        l = -_norm(L["dir"][i].view(shape)).expand_as(p)
        return l, torch.ones(p.shape[:-1], device=p.device, dtype=p.dtype)
    v = L["pos"][i].view(shape) - p
    dist = torch.sqrt(torch.clamp(_dot(v, v), min=1e-24))
    l = v / dist.unsqueeze(-1)
    k = L["attenuation"][i]
    att = 1.0 / torch.clamp(k[0] + k[1] * dist + k[2] * dist * dist, min=1e-12)
    cut = L["cutoff"][i]
    if cut < 180.0:
        se = _dot(_norm(L["dir"][i].view(shape)).expand_as(l), -l)
        inside = se >= float(np.cos(np.deg2rad(cut)))
        att = torch.where(inside, att * torch.clamp(se, min=0) ** L["exponent"][i],
                          torch.zeros_like(att))
    return l, att


def _shade(p, n, v, L, mat, spec, shininess, unlit=None):
    """``(...,3)`` colour in [0,1] for surface points ``p`` with unit normal ``n``, viewer dir ``v``.

    The composition is the one measured in ``docs/research/ogbench.md``: the specular term is WHITE
    and ADDITIVE and the texture modulates the whole primary colour. What is new here is that the
    per-light weights cannot be summed in advance — the two lights point in different directions and
    one of them is a spotlight — and that ``unlit`` masks the diffuse and specular of the
    shadow-casting light only, because a shadow does not remove ambient.
    """
    out = torch.zeros_like(p)
    highlight = torch.zeros(p.shape[:-1], device=p.device, dtype=p.dtype)
    exponent = 128.0 * shininess
    for i in range(len(L["kind"])):
        l, att = _light_geometry(L, i, p)
        lit = att
        if unlit is not None and L["castshadow"][i]:
            lit = att * (~unlit).to(att.dtype)
        ndl = torch.clamp(_dot(n, l), min=0)
        h = _norm(l + v)
        ndh = torch.clamp(_dot(n, h), min=0)
        out = out + mat * (att * L["ambient"][i][0]).unsqueeze(-1)
        out = out + mat * (lit * ndl * L["diffuse"][i][0]).unsqueeze(-1)
        highlight = highlight + lit * L["specular"][i][0] * ndh ** exponent
    return torch.clamp(out + (highlight * spec * SPECULAR_SCALE).unsqueeze(-1), 0, 1)


def _upsample_rgb(img: torch.Tensor, ss: int) -> torch.Tensor:
    """Nearest-neighbour blow-up of a baked plate, chosen so the box average that follows is the
    IDENTITY on plate pixels: every sample inside an output pixel carries the same integer, so the
    mean is that integer and ``round`` returns it. The plate is exact by construction, not
    approximately preserved."""
    return img if ss == 1 else img.repeat_interleave(ss, 0).repeat_interleave(ss, 1)

