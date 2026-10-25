"""Converted geometry keeps source identity and owns no persistent Blender mesh."""

from types import SimpleNamespace

import pytest

from scripts.export.object_permanence_geometry import evaluated_mesh, source_geometry


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


def test_failed_conversion_is_released_and_not_silently_omitted():
    obj, evaluated, _ = fake(mesh=False)
    with pytest.raises(ValueError, match="cannot produce"):
        with evaluated_mesh(obj, SimpleNamespace(mode="VIEWPORT")):
            raise AssertionError("missing geometry was yielded")
    assert evaluated.calls == evaluated.clears == 1


@pytest.mark.parametrize("kind", ["CURVE", "FONT"])
def test_render_only_resolution_is_rejected_before_allocating_geometry(kind):
    obj, evaluated, _ = fake(kind)
    evaluated.data.render_resolution_u = 24
    with pytest.raises(ValueError, match="tessellation differs"):
        with evaluated_mesh(obj, SimpleNamespace(mode="VIEWPORT")):
            pass
    assert evaluated.calls == evaluated.clears == 0


@pytest.mark.parametrize("levels", [False, True])
def test_render_only_modifiers_are_not_silently_sampled_with_preview_settings(levels):
    obj, evaluated, _ = fake("CURVE")
    evaluated.modifiers = [
        SimpleNamespace(type="SUBSURF", show_render=True, show_viewport=levels, render_levels=2, levels=1)
    ]
    with pytest.raises(ValueError, match="modifier differs"):
        with evaluated_mesh(obj, SimpleNamespace(mode="VIEWPORT")):
            pass
    assert evaluated.calls == evaluated.clears == 0


def test_text_is_bound_to_its_source_object_and_native_meshes_keep_the_old_path():
    obj, evaluated, _ = fake()
    assert source_geometry(obj, None) == {"type": "FONT", "text": "AB", "font": "Bfont"}
    obj.type = "CURVE"
    assert source_geometry(obj, None) == {"type": "CURVE"}
    obj.type = "MESH"
    evaluated.data = None
    assert source_geometry(obj, None) is None
    with evaluated_mesh(obj, None):
        pass
    assert evaluated.calls == evaluated.clears == 1


def test_unrepresented_geometry_stays_rejected():
    obj, evaluated, _ = fake("SURFACE")
    with pytest.raises(ValueError, match="unsupported evaluated geometry"):
        with evaluated_mesh(obj, None):
            pass
    assert evaluated.calls == 0


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


@pytest.mark.parametrize("change", ["parent", "identity", "transform", "instancer", "missing", "hidden"])
def test_real_instances_or_missing_proxy_cannot_be_folded_into_a_single_rigid_object(change):
    from scripts.export.object_permanence_geometry import validate_instances

    original, proxy, graph = proxy_graph()
    if change == "parent":
        proxy.parent = None
    elif change == "identity":
        proxy.object.original = object()
    elif change == "transform":
        proxy.matrix_world = ((1, 1), (0, 1))
    elif change == "instancer":
        original.instance_type = "FACES"
    elif change == "missing":
        graph.object_instances = []
    else:
        proxy.show_self = False
    with pytest.raises(ValueError, match="instances|unique evaluated mesh proxy"):
        validate_instances(graph, [original], allow_evaluated_geometry=True)
