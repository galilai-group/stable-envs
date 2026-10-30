"""Converted geometry keeps source identity and owns no persistent Blender mesh."""

from types import SimpleNamespace

import pytest

from scripts.export.object_permanence_geometry import evaluated_mesh


def fake(kind="FONT", *, mesh=True):
    data = SimpleNamespace(resolution_u=12, render_resolution_u=0, body="AB", font=SimpleNamespace(name="Bfont"))
    evaluated = SimpleNamespace(data=data, modifiers=[], calls=0, clears=0)
    value = SimpleNamespace(polygons=[SimpleNamespace(material_index=0)]) if mesh else None

    def to_mesh():
        evaluated.calls += 1
        return value

    def clear():
        evaluated.clears += 1

    evaluated.to_mesh, evaluated.to_mesh_clear = to_mesh, clear
    return SimpleNamespace(type=kind, name="source", evaluated_get=lambda _: evaluated), evaluated, value


@pytest.mark.parametrize("kind", ["MESH", "CURVE", "FONT"])
@pytest.mark.parametrize("failure", [False, True])
def test_temporary_evaluated_mesh_is_released_even_when_extraction_fails(kind, failure):
    obj, evaluated, mesh = fake(kind)
    try:
        with evaluated_mesh(obj, SimpleNamespace(mode="VIEWPORT")) as value:
            assert value is mesh and evaluated.clears == 0
            if failure:
                raise RuntimeError("extraction failed")
    except RuntimeError:
        assert failure
    assert evaluated.calls == evaluated.clears == 1
    assert obj.type == kind and evaluated.data.body == "AB"


def proxy_graph():
    class Source:
        type = "CURVE"
        instance_type = "NONE"

    original = Source()
    matrix = ((1, 0), (0, 1))
    parent = SimpleNamespace(original=original, matrix_world=matrix)
    proxy = SimpleNamespace(
        is_instance=True,
        object=SimpleNamespace(type="MESH", original=original),
        parent=parent,
        instance_object=SimpleNamespace(original=original),
        matrix_world=matrix,
        show_self=True,
    )
    return original, proxy, SimpleNamespace(object_instances=[proxy])


def test_only_a_unique_source_identity_mesh_proxy_is_accepted():
    from scripts.export.object_permanence_geometry import validate_instances

    original, proxy, graph = proxy_graph()
    validate_instances(graph, [original], allow_evaluated_geometry=True)
    with pytest.raises(ValueError, match="generated instances"):
        validate_instances(graph, [original])
    graph.object_instances.append(proxy)
    with pytest.raises(ValueError, match="generated instances"):
        validate_instances(graph, [original], allow_evaluated_geometry=True)
