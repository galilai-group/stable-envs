from __future__ import annotations

from ._delegated import DelegatedSemantics


class DMCAlignedSemantics(DelegatedSemantics):
    """BallInCup, Finger, Manipulator, Quadruped, PointMass and Swimmer on CPU dm_control lanes."""

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
