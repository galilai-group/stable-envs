from ..._core.mujoco_gl import configure_mujoco_gl
from ...registry import _import_env_module

configure_mujoco_gl()  # before the simulator imports MuJoCo
# The classes need the roadster extra (MuJoCo, Pillow); a missing extra is reported under "roadster".
_import_env_module("roadster.envs", "roadster")
