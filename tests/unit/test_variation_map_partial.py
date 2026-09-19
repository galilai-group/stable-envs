"""``VariationMap.partial``: a mapped leaf whose upstream effect is reproduced only in part."""

import pytest

from stable_envs.interop import VariationMap


def test_partial_is_a_classification_on_a_mapped_leaf():
    m = VariationMap(
        swm_id="swm/X-v0",
        renamed={"floor.color": "background.color"},
        partial={"background.color": "skybox half not reproduced (61.7-73.0 % of the frame)"},
    )
    assert m.pairs() == {"floor.color": ("background.color", 1.0)}
    assert m.classification()["background.color"] == "partial"
    assert "background.color" in m.mapped_swm_paths()
    assert "background.color" not in m.unmapped()


def test_partial_on_an_unmapped_leaf_is_refused():
    with pytest.raises(ValueError, match="partial names"):
        VariationMap(swm_id="swm/X-v0", partial={"background.color": "nothing reaches it"})


def test_maps_without_partial_are_unchanged():
    m = VariationMap(swm_id="swm/X-v0", identical=("a.b",), unsupported={"c.d": "why"})
    assert m.partial == {}
    assert m.classification() == {"a.b": "mapped", "c.d": "unsupported"}
