"""Replacement-readiness overlay on VariationMap buckets (buckets themselves stay unchanged)."""

import pytest

from tests.helpers.reference import VARIATION_MAPS
from tests.helpers.reference.readiness import (
    STATUSES,
    inventory,
    status_counts,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("cell", "leaf", "bucket", "replacement"),
    [
        ("PushT", "agent.color", None, "NATIVE"),
        ("TwoRoom", "agent.speed", None, "NATIVE"),
        ("Piecewise", "zones.bias_3", None, "NATIVE"),
        ("FetchPush", "block.mass", "mapped", "NATIVE"),
        ("FetchPush", "background.color", "mapped", "NATIVE"),
        ("FetchPush", "camera.angle_delta", "dead_upstream", "DEAD_UPSTREAM"),
    ],
)
def test_replacement_class_overlays_the_variation_bucket(cell, leaf, bucket, replacement):
    m = VARIATION_MAPS[cell]
    assert m.replacement_class()[leaf] == replacement
    if bucket is not None:
        assert m.classification()[leaf] == bucket


def test_ready_forbids_meaningful_gaps():
    for row in inventory():
        if row["status"] != "READY":
            continue
        assert row["counts"]["UNSUPPORTED_MEANINGFUL"] == 0
        assert row["counts"]["PARTIAL"] == 0
        assert row["execution"] in {"NATIVE", "DELEGATED_CPU"}


def test_every_cell_has_a_known_status():
    rows = inventory()
    assert len(rows) == 42
    assert all(r["status"] in STATUSES for r in rows)
    assert sum(status_counts().values()) == 42
