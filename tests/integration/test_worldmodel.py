"""Certification/provenance for the frozen SWM correspondence. Not a public interop contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import stable_envs as snv
from stable_envs import registry
from tests.helpers.reference import (
    SWM_IDS,
    SYMBOLIC_IDS,
    VariationMap,
    canonical_names,
    class_for_frozen_cell,
    pusht_state,
    style_from_variation,
    swap_renderer,
    variation_from_style,
    variation_map,
)

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
FROZEN_MATRIX = ROOT / "tests" / "fixtures" / "regression" / "gate_f_matrix.json"


def _frozen_rows() -> list[dict]:
    rows = json.loads(FROZEN_MATRIX.read_text())["rows"]
    assert len(rows) == 42
    return rows


def test_shipped_package_has_no_interop_module():
    with pytest.raises(ModuleNotFoundError):
        __import__("stable_envs.interop")


def test_all_42_frozen_swm_ids_join_the_same_canonical_classes():
    expected = {row["swm_id"]: tuple(row["stable_envs_cells"]) for row in _frozen_rows()}
    assert SWM_IDS == expected
    assert len(expected) == 42

    for swm_id, class_names in expected.items():
        assert canonical_names(swm_id) == class_names
        for class_name in class_names:
            cls = class_for_frozen_cell(class_name)
            if cls is not None:
                assert getattr(snv, cls.__name__) is cls
                if class_name != cls.__name__:
                    with pytest.raises(KeyError):
                        registry._resolve(class_name)
        if len(class_names) == 1:
            live = class_for_frozen_cell(class_names[0])
            if live is not None:
                with pytest.raises(KeyError, match="unknown env"):
                    snv.info(swm_id)

    assert SYMBOLIC_IDS == (
        "swm/CraftaxClassicSymbolic-v1",
        "swm/CraftaxSymbolic-v1",
    )


def test_swm_ids_are_not_registry_aliases():
    with pytest.raises(KeyError, match="unknown env"):
        snv.make("swm/PushT-v1")
    info = snv.info("PushT")
    assert "swm_ids" not in info
    assert info["class_name"] == "PushT"


def test_variation_map_bridge_is_test_only():
    env = snv.PushT(num_envs=1, device="cpu", size=32)
    try:
        mapping = variation_map(env)
        assert isinstance(mapping, VariationMap)
        assert mapping.swm_id == "swm/PushT-v1"
        style = env.default_variation()
        variation = variation_from_style(env, style)
        restored = style_from_variation(env, variation)
        np.testing.assert_array_equal(restored, style)
    finally:
        env.close()


def test_swap_renderer_keeps_the_live_swm_pusht_and_replaces_only_pixels():
    gymnasium = pytest.importorskip("gymnasium")
    pytest.importorskip("stable_worldmodel")

    official = gymnasium.make("swm/PushT-v1", render_mode="rgb_array")
    renderer = None
    try:
        official.reset(seed=0)
        official_type = type(official.unwrapped)
        original_render = official.unwrapped.render
        returned = swap_renderer(
            official,
            "PushT",
            pusht_state,
            size=32,
            device="cpu",
        )
        renderer = official.unwrapped._stable_envs_renderer

        assert returned is official
        assert type(official.unwrapped) is official_type
        assert callable(official.unwrapped._stable_envs_original_render)
        assert official.unwrapped._stable_envs_original_render.__func__ is original_render.__func__
        frame = official.render()
        assert frame.shape == (32, 32, 3)
        assert frame.dtype == np.uint8

        action = official.action_space.sample()
        _, reward, terminated, truncated, _ = official.step(action)
        assert np.isscalar(reward)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
    finally:
        if renderer is not None:
            renderer.close()
        official.close()
