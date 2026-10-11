from __future__ import annotations

from ...spaces import Discrete
from ._delegated import DelegatedSemantics


class DMCAlignedSemantics(DelegatedSemantics):
    """BallInCup, Finger, Manipulator, Quadruped, PointMass and Swimmer on CPU dm_control lanes."""

    def make_variation_space(self):
        space = super().make_variation_space()
        if self._simulator_key() == "finger":
            space["visibility"]["target"] = Discrete(2, init_value=0)
        return space

    @classmethod
    def _hideable(cls):
        entries = super()._hideable()
        if cls._simulator_key() != "finger":
            return entries
        # The target material is shared by the marker and tip sites; hide every site using it.
        model = cls._model()
        material = model.material("target").id
        sites = tuple(i for i in range(model.nsite) if model.site_matid[i] == material)
        return tuple(
            (kind, name, sites if kind == "site" and name == "target" else ids) for kind, name, ids in entries
        )
