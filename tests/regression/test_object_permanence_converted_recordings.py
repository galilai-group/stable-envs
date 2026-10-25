"""Converted curve sections and text identities remain independent scene objects."""

from copy import deepcopy

import pytest

from stable_envs.envs.object_permanence import RigidTrajectory
from tests.helpers.reference.object_permanence_converted import CONVERTED_TASKS, load_converted_geometry
from tests.helpers.reference.object_permanence_recordings import RECORDINGS


@pytest.mark.parametrize("task", CONVERTED_TASKS)
def test_source_geometry_identity_counts_and_nonempty_triangle_assets(task):
    from scripts.export.object_permanence import CONVERTED_GEOMETRY_TASKS
    from scripts.testing.freeze_object_permanence import REFERENCE_CONVERTED_TASKS

    assert set(t.upper() for t in CONVERTED_TASKS) == CONVERTED_GEOMETRY_TASKS == REFERENCE_CONVERTED_TASKS
    oracle, arrays = load_converted_geometry(task)
    with RigidTrajectory(RECORDINGS, task + "-nominal") as clip:
        manifest = clip.manifest
        assert manifest["geometry_conversion"] == oracle["geometry_conversion"]
        source = {obj["name"]: obj["source_geometry"] for obj in manifest["objects"] if "source_geometry" in obj}
        assert source == oracle["source_geometry"] and source
        assert set(source) <= set(oracle["object_ids"])
        for obj in manifest["objects"]:
            mesh = clip.mesh(obj["asset"])
            assert len(mesh["triangles"]) and len(mesh["vertices"])
        labels = {name: data["text"] for name, data in source.items() if data["type"] == "FONT"}
        if task == "g13":
            assert labels == {f"drawer_{i}_front_label": str(i) for i in (1, 2, 3)}
        elif task == "g46":
            assert labels == {f"box_{text}_label": text for text in ("A", "B")}
        else:
            assert not labels and all(data["type"] == "CURVE" for data in source.values())
        for name in labels:
            assert (arrays["object_ids"] == oracle["object_ids"][name]).any()
        assert manifest["source"]["preparation_recipe"] == oracle["source_preparation_recipe"]


@pytest.mark.parametrize("change", ["policy", "text", "identity"])
def test_conversion_metadata_mismatch_fails_before_gpu_construction(monkeypatch, change):
    from scripts.testing import check_object_permanence_rendering as checker

    oracle, arrays = load_converted_geometry("g46")
    wrong = deepcopy(oracle)
    if change == "policy":
        wrong["geometry_conversion"]["id"] = "different-policy"
    elif change == "text":
        wrong["source_geometry"]["box_A_label"]["text"] = "B"
    else:
        wrong["source_geometry"].pop("box_B_label")
    monkeypatch.setattr(checker, "load_geometry_directory", lambda _: (wrong, arrays))
    with pytest.raises(ValueError, match="conversion policies differ|converted geometry identities differ"):
        checker.check(RECORDINGS, "unused", "g46", visibility_contract=oracle["visibility_contract"])
