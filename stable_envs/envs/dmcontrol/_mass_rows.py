from __future__ import annotations

import numpy as np
import torch

from ..._core import physics as _physics
from .simulation import physics as _leaves


class MassRows:
    """Per-row body masses, centres of mass and subtree masses under a cell's mass leaves.

    The same arithmetic as ``PhysicsKnobs``: a leaf scales its geoms' contribution to their body,
    and a model compiled with settotalmass rescales every body. Tracking cameras read the
    resulting subtree centres of mass, so these leaves move the frame."""

    def __init__(self, key: str, model):
        knobs = [k for k in _leaves.knobs(key) if isinstance(k, _physics.GeomMass)]
        spec = _physics.PhysicsKnobs(knobs, model, **_leaves.compile_options(key))
        self.paths = tuple(dict.fromkeys(k.leaf for k in knobs))
        nbody, count = int(model.nbody), len(self.paths)
        self.nominal = np.zeros(count)
        self.part_mass = np.zeros((count, nbody))
        self.part_moment = np.zeros((count, nbody, 3))
        self.unscaled = np.zeros(count)
        for mass in spec._masses:
            i = self.paths.index(mass.knob.leaf)
            self.nominal[i] = float(mass.knob.nominal)
            self.part_mass[i, mass.body] += float(mass.part[0])
            self.part_moment[i, mass.body] += np.asarray(mass.part[1], np.float64)
            self.unscaled[i] += float(mass.unscaled_mass)
        self.mass = np.array(model.body_mass, np.float64)
        self.ipos = np.array(model.body_ipos, np.float64)
        self.subtree = np.array(model.body_subtreemass, np.float64)
        self.moment = self.mass[:, None] * self.ipos
        # ancestor[b, d] is 1 when body d lies in the subtree rooted at b.
        self.ancestor = np.zeros((nbody, nbody))
        for d in range(nbody):
            b = d
            while True:
                self.ancestor[b, d] = 1.0
                if b == 0:
                    break
                b = int(model.body_parentid[b])
        self.unscaled_total = None if spec.total_mass is None else spec._unscaled_total

    def __call__(self, values: torch.Tensor):
        """``values`` is ``(B, len(paths))``; return float64 ``(mass, ipos, subtreemass)`` rows."""
        t = lambda x: torch.as_tensor(x, device=values.device, dtype=torch.float64)  # noqa: E731
        s = values.to(torch.float64) / t(self.nominal) - 1.0
        mass = t(self.mass) + s @ t(self.part_mass)
        moment = t(self.moment) + torch.einsum("bp,pnk->bnk", s, t(self.part_moment))
        ipos = torch.where(mass[..., None] > 0, moment / mass.clamp_min(1e-300)[..., None], t(self.ipos))
        if self.unscaled_total is not None:
            total = self.unscaled_total
            mass = mass * (total / (total + s @ t(self.unscaled)))[:, None]
        return mass, ipos, mass @ t(self.ancestor).T

    def restore(self, model) -> None:
        model.body_mass[:], model.body_ipos[:], model.body_subtreemass[:] = self.mass, self.ipos, self.subtree
