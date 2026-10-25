"""Offline conversion of source Curve/FONT objects to owned temporary meshes.

Blender evaluates the source bevel, caps, extrusion and glyphs. No source object,
curve resolution or text is rewritten. Runtime replay consumes ordinary mesh
assets and does not import this module or Blender.
"""

from contextlib import contextmanager

GEOMETRY_CONVERSION = {
    "id": "stable-envs/evaluated-rigid-mesh-v1",
    "types": ["MESH", "CURVE", "FONT"],
    "depsgraph_mode": "VIEWPORT",
    "tessellation": "source-settings",
    "instances": "single-source-identity-mesh-proxy",
}


def validate_instances(depsgraph, objects, *, allow_evaluated_geometry=False):
    """Distinguish a Curve/FONT's own evaluated mesh from actual instancing."""
    converted = {obj for obj in objects if allow_evaluated_geometry and obj.type in ("CURVE", "FONT")}
    seen = set()
    matrix = lambda value: tuple(tuple(row) for row in value)
    for instance in depsgraph.object_instances:
        if not instance.is_instance:
            continue
        parent = instance.parent
        original = None if parent is None else parent.original
        if (
            original is None
            or original not in converted
            or original in seen
            or original.instance_type != "NONE"
            or instance.object.type != "MESH"
            or instance.object.original != original
            or instance.instance_object is None
            or instance.instance_object.original != original
            or not instance.show_self
            or matrix(instance.matrix_world) != matrix(parent.matrix_world)
        ):
            raise ValueError("v1 does not support generated instances")
        seen.add(original)
    if seen != converted:
        raise ValueError("converted source has no unique evaluated mesh proxy")


def source_geometry(obj, depsgraph):
    """Describe converted source identity; text remains geometry, not an overlay."""
    if obj.type == "MESH":
        return None
    if obj.type not in ("CURVE", "FONT"):
        raise ValueError(f"unsupported evaluated geometry: {obj.type}")
    evaluated = obj.evaluated_get(depsgraph)
    result = {"type": obj.type}
    if obj.type == "FONT":
        result.update(text=evaluated.data.body, font=evaluated.data.font.name)
    return result


@contextmanager
def evaluated_mesh(obj, depsgraph):
    """Borrow evaluated triangles and always release their temporary storage.

    Native MESH extraction retains its original API path. Curve/FONT conversion
    requires the source's viewport and render tessellation to agree; render-only
    settings must be audited explicitly rather than silently approximated.
    """
    if obj.type not in ("MESH", "CURVE", "FONT"):
        raise ValueError(f"unsupported evaluated geometry: {obj.type}")
    evaluated = obj.evaluated_get(depsgraph)
    if obj.type in ("CURVE", "FONT"):
        data = evaluated.data
        if depsgraph.mode != "VIEWPORT" or data.render_resolution_u not in (0, data.resolution_u):
            raise ValueError(f"render/viewport curve tessellation differs: {obj.name}")
        for modifier in evaluated.modifiers:
            if modifier.show_render != modifier.show_viewport or (
                modifier.type in ("SUBSURF", "MULTIRES") and modifier.render_levels != modifier.levels
            ):
                raise ValueError(f"render/viewport modifier differs: {obj.name}")
    try:
        mesh = evaluated.to_mesh()
        if mesh is None:
            raise ValueError(f"source cannot produce evaluated triangles: {obj.name}")
        if obj.type != "MESH" and any(p.material_index != 0 for p in mesh.polygons):
            raise ValueError(f"converted geometry requires one material: {obj.name}")
        yield mesh
    finally:
        evaluated.to_mesh_clear()
