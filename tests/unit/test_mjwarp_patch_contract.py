from types import SimpleNamespace

import pytest

from stable_envs.render import mjwarp as M


@pytest.mark.parametrize("target", [None, 42])
def test_missing_or_renamed_patch_target_fails_clearly(target):
    module = SimpleNamespace(__name__="mujoco_warp._src.render", pack_rgba_to_uint32=target)
    with pytest.raises(RuntimeError, match=r"incompatible MJWarp.*pack_rgba_to_uint32"):
        M._patch_target(module, "pack_rgba_to_uint32")
