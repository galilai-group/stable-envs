"""The source-renderer table in tests/helpers/reference/sources.py."""

from __future__ import annotations

import pytest

import stable_envs as se
from tests.helpers.reference import sources

pytestmark = pytest.mark.unit


def test_only_mujoco_and_pybullet_sources_draw_on_a_gpu():
    for name in se.list_envs():
        cls = se.registry.REGISTRY[name]
        expected = sources.GPU_RENDER.get(cls.family)
        assert sources.reference_gpu_render(cls) == expected
        assert expected in (None, "mujoco-egl", "pybullet-egl")
    assert sources.reference_gpu_render(type("Other", (), {"family": "classic_control"})) is None


def test_the_comparison_metadata_is_not_on_the_env_classes():
    for name in se.list_envs():
        cls = se.registry.REGISTRY[name]
        assert not hasattr(cls, "reference_gpu_render") and not hasattr(cls, "reference_inputs"), name
