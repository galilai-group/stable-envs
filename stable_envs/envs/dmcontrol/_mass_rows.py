from __future__ import annotations

import numpy as np
import torch

from ..._core import physics as _physics
from .simulation import physics as _leaves

#: Model arrays a mass leaf changes that tracking cameras and lights read.
ARRAYS = ("body_mass", "body_ipos", "body_subtreemass", "cam_poscom0", "light_poscom0")


class MassRows:
    """Per-row model arrays that place tracking cameras and lights under a cell's mass leaves.

    The same arithmetic as ``PhysicsKnobs``: a leaf scales its geoms' contribution to their body,
    and a model compiled with settotalmass rescales every body. A tracking camera sits at its
    body's subtree centre of mass plus an offset taken at qpos0, so both move with the masses."""

    def __init__(self, key: str, model):
        import mujoco

        knobs = [k for k in _leaves.knobs(key) if isinstance(k, _physics.GeomMass)]
        spec = _physics.PhysicsKnobs(knobs, model, **_leaves.compile_options(key))
        self.paths = tuple(dict.fromkeys(k.leaf for k in knobs))
        nbody, count = int(model.nbody), len(self.paths)
        self.leaf_nominal = np.zeros(count)
        self.part_mass = np.zeros((count, nbody))
        self.part_moment = np.zeros((count, nbody, 3))
        self.unscaled = np.zeros(count)
        for mass in spec._masses:
            i = self.paths.index(mass.knob.leaf)
            self.leaf_nominal[i] = float(mass.knob.nominal)
            self.part_mass[i, mass.body] += float(mass.part[0])
            self.part_moment[i, mass.body] += np.asarray(mass.part[1], np.float64)
            self.unscaled[i] += float(mass.unscaled_mass)
        self.unscaled_total = None if spec.total_mass is None else spec._unscaled_total
        self.nominal = {name: np.array(getattr(model, name), np.float64) for name in ARRAYS}
        # ancestor[b, d] is 1 when body d lies in the subtree rooted at b.
        self.ancestor = np.zeros((nbody, nbody))
        for d in range(nbody):
            b = d
            while True:
                self.ancestor[b, d] = 1.0
                if b == 0:
                    break
                b = int(model.body_parentid[b])
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        mujoco.mj_kinematics(model, data)
        self.xpos0 = np.array(data.xpos, np.float64)
        self.xmat0 = np.array(data.xmat, np.float64).reshape(nbody, 3, 3)
        self.cam_body = np.array(model.cam_bodyid, np.int64)
        self.light_body = np.array(model.light_bodyid, np.int64)

    def __call__(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        """``values`` is ``(B, len(paths))``; return float64 rows for every name in ``ARRAYS``."""
        t = lambda x: torch.as_tensor(x, device=values.device, dtype=torch.float64)  # noqa: E731
        s = values.to(torch.float64) / t(self.leaf_nominal) - 1.0
        mass0, ipos0 = t(self.nominal["body_mass"]), t(self.nominal["body_ipos"])
        mass = mass0 + s @ t(self.part_mass)
        moment = mass0[:, None] * ipos0 + torch.einsum("bp,pnk->bnk", s, t(self.part_moment))
        ipos = torch.where(mass[..., None] > 0, moment / mass.clamp_min(1e-300)[..., None], ipos0)
        if self.unscaled_total is not None:
            total = self.unscaled_total
            mass = mass * (total / (total + s @ t(self.unscaled)))[:, None]
        ancestor = t(self.ancestor)
        subtree = mass @ ancestor.T
        base = self._com0(mass0[None], ipos0[None], t(self.nominal["body_subtreemass"])[None], ancestor)
        shift = base - self._com0(mass, ipos, subtree, ancestor)
        return {
            "body_mass": mass,
            "body_ipos": ipos,
            "body_subtreemass": subtree,
            "cam_poscom0": t(self.nominal["cam_poscom0"]) + shift[:, self.cam_body],
            "light_poscom0": t(self.nominal["light_poscom0"]) + shift[:, self.light_body],
        }

    def _com0(self, mass, ipos, subtree, ancestor):
        """Subtree centres of mass at qpos0, ``(B, nbody, 3)``."""
        t = lambda x: torch.as_tensor(x, device=mass.device, dtype=torch.float64)  # noqa: E731
        xipos = t(self.xpos0) + torch.einsum("nij,bnj->bni", t(self.xmat0), ipos)
        weighted = torch.einsum("cn,bnk->bck", ancestor, mass[..., None] * xipos)
        return weighted / subtree.clamp_min(1e-300)[..., None]

    def restore(self, model) -> None:
        for name, value in self.nominal.items():
            getattr(model, name)[:] = value
