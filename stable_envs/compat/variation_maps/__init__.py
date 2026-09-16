"""Per-family ``stable_worldmodel`` VariationMaps, one module per family.

``stable_envs.compat.worldmodel`` owns the ``VariationMap`` dataclass and the four maps that
predate Step 8. The families added in Step 8 are large enough — twelve Fetch ids, thirteen DM
Control ids — that keeping them inline would bury the dataclass they use under tables. Each module
here exports a ``MAPS`` dict (``our class name -> VariationMap``) and nothing else that a caller
needs; ``worldmodel`` merges them into ``VARIATION_MAPS`` at import, which is the only name any
caller should read.

A module here must import cleanly on ANY interpreter: the maps are data, and a caller that wants to
know how ``swm/FetchReach-v3``'s leaves are classified should not need ``mujoco_warp``, a GPU, or
even the env's optional dependencies installed. So nothing here imports ``stable_envs.envs.*``.
"""
