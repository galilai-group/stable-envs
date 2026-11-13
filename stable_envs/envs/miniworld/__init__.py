"""Miniworld ``MovingBlockWorld``: a first-person 3-D room of moving blocks (``miniworld`` extra)."""

from ...registry import _import_env_module

# The class needs the miniworld extra (the hlillemark/Miniworld fork, pyglet); a missing extra is reported under "miniworld".
_import_env_module("miniworld.envs", "miniworld")
