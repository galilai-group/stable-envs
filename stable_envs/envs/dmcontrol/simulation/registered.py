"""Build dm_control suite models with the registered defaults applied before compiling."""

from __future__ import annotations

import os
import tempfile

import numpy as np


def source_model(module, xml, assets):
    """Parse a suite module's MJCF with its ``common/`` asset paths resolved."""
    from dm_control import mjcf

    return mjcf.from_xml_string(
        xml.replace(b'file="./common/', b'file="common/'),
        model_dir=os.path.dirname(module.__file__),
        assets=assets or {},
    )


def apply_appearance(model, color=(0.7, 0.5, 0.3), light="light"):
    """Set the registered grid texture, ``self`` material color and light diffuse."""
    grid = model.find("texture", "grid")
    grid.rgb1, grid.rgb2 = [0.1, 0.2, 0.3], [0.2, 0.3, 0.4]
    model.find("material", "self").rgba = np.array([*color, 1.0], np.float32)
    model.find("light", light).diffuse = np.full(3, 0.7, np.float32)


def apply_floor_friction(floor):
    """Give ``floor`` the registered unit sliding friction, keeping its other coefficients."""
    friction = (
        np.array([1.0, 0.005, 0.0001], np.float32)
        if floor.friction is None
        else np.asarray(floor.friction, np.float32).copy()
    )
    friction[0] = 1.0
    floor.friction = friction


def compile_physics(physics_type, model, filename):
    """Export ``model`` with its assets and compile it as ``physics_type``."""
    from dm_control import mjcf

    with tempfile.TemporaryDirectory() as directory:
        mjcf.export_with_assets(model, directory, out_file_name=filename)
        return physics_type.from_xml_path(os.path.join(directory, filename))


def normalized(env):
    """Wrap ``env`` so actions are given in [-1, 1]."""
    from dm_control.suite.wrappers import action_scale

    return action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)
