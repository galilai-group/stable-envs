"""V2 orthographic source references, separate from frozen V1 registrations."""

from functools import lru_cache

from .object_permanence_recordings import FIXTURES, load_geometry_directory

ORTHOGRAPHIC_TASKS = (
    "g08",
    "g09",
    "g10",
    "g101",
    "g102",
    "g109",
    "g11",
    "g110",
    "g112",
    "g12",
    "g122",
    "g128",
    "g15",
    "g16",
    "g17",
    "g18",
    "g19",
    "g20",
    "g21",
    "g22",
    "g23",
    "g24",
    "g25",
    "g26",
    "g43",
    "g45",
    "g47",
    "g48",
    "g49",
    "g50",
    "g51",
    "g52",
    "g53",
    "g57",
    "g67",
    "g75",
    "g76",
    "g77",
    "g78",
    "g91",
    "g92",
    "g13",
    "g14",
    "g46",
)


@lru_cache(maxsize=2)
def load_orthographic_geometry(task):
    if task not in ORTHOGRAPHIC_TASKS:
        raise ValueError("unknown certified orthographic scene")
    manifest, arrays = load_geometry_directory(
        FIXTURES / f"semantics/object_permanence/{task}/opaque_pixel_rays_v2/nominal"
    )
    assert manifest["schema"] == "stable-envs/object-permanence-rigid-oracle-v2"
    assert manifest["visibility_contract"] == "stable-envs/opaque-pixel-rays-v2"
    assert manifest["camera"]["projection_type"] == "orthographic"
    return manifest, arrays
