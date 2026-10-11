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
        self._device: dict = {}

    def _constants(self, device) -> dict[str, torch.Tensor]:
        hit = self._device.get(str(device))
        if hit is None:
            t = lambda x: torch.as_tensor(x, device=device, dtype=torch.float64)  # noqa: E731
            hit = {name: t(value) for name, value in self.nominal.items()}
            hit.update(
                leaf_nominal=t(self.leaf_nominal),
                leaf_nominal32=t(self.leaf_nominal).to(torch.float32),
                part_mass=t(self.part_mass),
                part_moment=t(self.part_moment),
                unscaled=t(self.unscaled),
                ancestor=t(self.ancestor),
                xpos0=t(self.xpos0),
                xmat0=t(self.xmat0),
            )
            hit["com0"] = self._com0(
                hit, hit["body_mass"][None], hit["body_ipos"][None], hit["body_subtreemass"][None]
            )
            self._device[str(device)] = hit
        return hit

    def registered(self, values: torch.Tensor) -> bool:
        """Whether every row holds the registered value of every leaf."""
        return bool((values == self._constants(values.device)["leaf_nominal32"]).all())

    def __call__(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        """``values`` is ``(B, len(paths))``; return float64 rows for every name in ``ARRAYS``."""
        c = self._constants(values.device)
        s = values.to(torch.float64) / c["leaf_nominal"] - 1.0
        mass0, ipos0 = c["body_mass"], c["body_ipos"]
        mass = mass0 + s @ c["part_mass"]
        moment = mass0[:, None] * ipos0 + torch.einsum("bp,pnk->bnk", s, c["part_moment"])
        ipos = torch.where(mass[..., None] > 0, moment / mass.clamp_min(1e-300)[..., None], ipos0)
        if self.unscaled_total is not None:
            total = self.unscaled_total
            mass = mass * (total / (total + s @ c["unscaled"]))[:, None]
        subtree = mass @ c["ancestor"].T
        shift = c["com0"] - self._com0(c, mass, ipos, subtree)
        return {
            "body_mass": mass,
            "body_ipos": ipos,
            "body_subtreemass": subtree,
            "cam_poscom0": c["cam_poscom0"] + shift[:, self.cam_body],
            "light_poscom0": c["light_poscom0"] + shift[:, self.light_body],
        }

    @staticmethod
    def _com0(c, mass, ipos, subtree):
        """Subtree centres of mass at qpos0, ``(B, nbody, 3)``."""
        xipos = c["xpos0"] + torch.einsum("nij,bnj->bni", c["xmat0"], ipos)
        weighted = torch.einsum("cn,bnk->bck", c["ancestor"], mass[..., None] * xipos)
        return weighted / subtree.clamp_min(1e-300)[..., None]

    def restore(self, model) -> None:
        for name, value in self.nominal.items():
            getattr(model, name)[:] = value
