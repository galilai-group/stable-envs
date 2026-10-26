"""Freeze audited rigid scenes directly from pinned Blender, without candidate imports.

The depth encoding XORs IEEE float32 bits against frame one, losslessly removing
repeated background data. It does not change depth precision or acceptance gates.
Existing output directories are never overwritten, including the G34 fixture.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import io
import json
import platform
import runpy
import subprocess
from pathlib import Path

import numpy as np

SOURCE_COMMIT = "824baddcdad087810d35638bf8f273b884f74975"
# Independently audited positive diagonal scale animation, sampled in world space.
REFERENCE_SCALE_TASKS = {"G146"}
# Separate from candidate extraction/registration; only audited converted cases.
REFERENCE_CONVERTED_TASKS = {
    "G04",
    "G13",
    "G14",
    "G29",
    "G31",
    "G46",
}
# Deliberately independent of the candidate exporter's entry registry/extraction.
# These calls and targets were checked against each source main()/selected builder.
# Independently audited entries/targets; never imported from the exporter.
REFERENCE_MEMBERS = {
    ("G02", 1): ("$prefix:render_png", "", ("target_red_ball",)),
    ("G02", 2): ("$prefix:render_png", "", ("front_yellow_ball", "back_blue_ball")),
    ("G04", 1): (
        "build_scene",
        "animate_scene",
        ("identity_A_red_ball_on_upper_straight_track", "identity_B_blue_ball_on_lower_straight_track"),
    ),
    ("G05", 2): ("build_scene", "animate_scene", ("red_wheeled_car_body",)),
    ("G05", 4): ("build_scene", "animate_scene", ("red_wheeled_car_body",)),
    ("G05", 5): ("build_scene", "animate_scene", ("red_wheeled_car_body",)),
    ("G05", 6): ("build_scene", "animate_scene", ("front_red_car_body", "rear_blue_car_body")),
}

REFERENCE_CASES = {
    "G98": ("build_scene_by_kind", "animate_scene_by_kind", ("colored_ball_on_base",)),
    "G132": ("build_scene_by_kind", "animate_scene_by_kind", ("hidden_ball_in_capsule",)),
    "G05": ("build_scene", "animate_scene", ("red_wheeled_car_body",)),
    "G06": ("build_scene", "animate_scene", ("hole_bridge_red_car_body",)),
    "G07": ("build_scene_by_kind", "animate_scene_by_kind", ("red_car_body",)),
    "G02": ("$prefix:render_png", "", ("target_red_ball",)),
    "G03": ("$prefix:render_png", "", ("target_red_ball_blocked_by_solid_wall",)),
    "G27": ("build_scene", "animate_scene", ("target_orange_ball",)),
    "G37": ("build_scene_by_kind", "animate_scene", ("orange_ball",)),
    "G38": ("build_scene_by_kind", "animate_scene", ("red_pendulum_bob",)),
    "G64": ("build_scene_by_kind", "animate_scene_by_kind", ("static_object_a_ball", "static_object_b_cylinder")),
    "G55": ("build_scene_by_kind", "animate_scene", ("red_block", "green_block")),
    "G63": ("build_scene_by_kind", "animate_scene", ("red_block", "green_block", "blue_block")),
    "G66": ("build_scene_by_kind", "animate_scene", ("hidden_ball",)),
    "G80": ("build_scene_by_kind", "animate_scene", ("red_cube_target", "blue_sphere_target")),
    "G81": ("build_scene_by_kind", "animate_scene", ("red_box", "blue_cylinder")),
    "G65": ("build_scene_by_kind", "animate_scene", ("ball_red", "ball_green")),
    "G83": ("build_scene_by_kind", "animate_scene", ("hidden_ball",)),
    "G82": ("build_scene_by_kind", "animate_scene", ("target_red_cube", "target_blue_cylinder")),
    "G84": ("build_scene_by_kind", "animate_scene", ("red_hidden_ball", "green_hidden_ball")),
    "G86": ("build_scene_by_kind", "animate_scene", ("hidden_ball",)),
    "G85": ("build_scene_by_kind", "animate_scene", ("hidden_ball",)),
    "G87": ("build_scene_by_kind", "animate_scene", ("hidden_ball",)),
    "G88": ("build_scene_by_kind", "animate_scene", ("hidden_ball",)),
    "G93": ("build_scene_by_kind", "animate_scene", ("red_ball_on_shelf",)),
    "G94": ("build_scene_by_kind", "animate_scene", ("trapdoor_ball",)),
    "G95": ("build_scene_by_kind", "animate_scene", ("orange_ball_on_conveyor",)),
    "G96": ("build_scene_by_kind", "animate_scene_by_kind", ("colored_ball_inside_clamshell",)),
    "G97": ("build_scene_by_kind", "animate_scene_by_kind", ("colored_ball_inside_box",)),
    "G99": ("build_scene_by_kind", "animate_scene", ("incoming_ball", "target_ball")),
    "G100": ("build_scene_by_kind", "animate_scene", ("heavy_ball", "light_ball")),
    "G103": ("build_scene_by_kind", "animate_scene_by_kind", ("corner_turn_red_ball",)),
    "G89": (
        "build_scene_by_kind",
        "animate_scene",
        ("short_orange_puck_passes_under", "tall_red_cylinder_blocked_by_bar"),
    ),
    "G28": ("build_scene", "animate_scene", ("identity_A_red_ball", "identity_B_blue_ball")),
    "G30": ("build_scene", "animate_scene", ("target_orange_ball_inside_moving_cart",)),
    "G32": (
        "build_scene",
        "animate_scene",
        ("identity_A_red_ball_inside_cart_A", "identity_B_blue_ball_inside_cart_B"),
    ),
    "G33": ("build_scene", "animate_scene", ("far_blue_tall_car_body", "near_red_low_car_body")),
    "G35": ("build_scene", "animate_scene", ("near_red_low_car_body",)),
    "G39": ("build_scene_by_kind", "animate_scene", ("orange_ball",)),
    "G40": ("build_scene_by_kind", "animate_scene", ("orange_ball_inside_guided_elevator",)),
    "G41": ("build_scene_by_kind", "animate_scene", ("orange_ball_on_trapdoor",)),
    "G42": ("build_scene_by_kind", "animate_scene", ("orange_ball_on_ring_track",)),
    "G44": ("build_scene_by_kind", "animate_scene", ("blue_ball_right", "red_ball_left")),
    "G56": ("build_scene_by_kind", "animate_scene", ("hidden_ball",)),
    "G58": (
        "build_scene_by_kind",
        "animate_scene",
        ("large_red_ball_wider_than_hole", "small_orange_ball_narrower_than_hole"),
    ),
    "G59": ("build_scene_by_kind", "animate_scene_by_kind", ("colored_ball_inside_box",)),
    "G60": (
        "build_scene_by_kind",
        "animate_scene",
        ("metal_ball_0", "metal_ball_1", "metal_ball_2", "metal_ball_3", "metal_ball_4"),
    ),
    "G68": (
        "build_scene_by_kind",
        "animate_scene",
        ("narrow_orange_ball_passes_slot", "wide_red_ball_blocked_by_wall"),
    ),
    "G69": ("build_scene_by_kind", "animate_scene", ("cascade_ball",)),
    "G70": ("build_scene_by_kind", "animate_scene", ("red_ball_on_slab",)),
    "G71": ("build_scene_by_kind", "animate_scene_by_kind", ("colored_ball_inside_box",)),
    "G72": ("build_scene_by_kind", "animate_scene_by_kind", ("colored_ball_inside_box",)),
    "G73": (
        "build_scene_by_kind",
        "animate_scene",
        ("cue_ball", "rack_ball_0", "rack_ball_1", "rack_ball_2", "rack_ball_3", "rack_ball_4", "rack_ball_5"),
    ),
    "G74": ("build_scene_by_kind", "animate_scene", ("target_ball",)),
    "G79": ("build_scene_by_kind", "animate_scene_by_kind", ("static_object_a_ball", "static_object_b_cylinder")),
    "G104": ("build_scene_by_kind", "animate_scene", ("blue_sphere_target", "red_cube_target")),
    "G105": ("build_scene_by_kind", "animate_scene", ("blue_sphere_target", "red_cube_target")),
    "G106": ("build_scene_by_kind", "animate_scene", ("blue_sphere_target", "red_cube_target")),
    "G107": ("build_scene_by_kind", "animate_scene", ("hidden_ball",)),
    "G108": ("build_scene_by_kind", "animate_scene", ("hidden_ball",)),
    "G111": ("build_scene", "animate_scene", ("first_orange_ball", "second_blue_ball")),
    "G113": ("build_scene_by_kind", "animate_scene", ("orange_ball_on_card",)),
    "G114": ("build_scene_by_kind", "animate_scene", ("orange_ball_on_pins",)),
    "G115": ("build_scene_by_kind", "animate_scene", ("orange_ball_sliding_down_slope",)),
    "G116": ("build_scene_by_kind", "animate_scene_by_kind", ("colored_ball_inside_safe",)),
    "G117": ("build_scene_by_kind", "animate_scene_by_kind", ("colored_ball_inside_petals",)),
    "G118": ("build_scene_by_kind", "animate_scene_by_kind", ("colored_ball_in_tray",)),
    "G119": ("build_scene_by_kind", "animate_scene", ("incoming_ball", "resting_ball_left", "resting_ball_right")),
    "G120": ("build_scene_by_kind", "animate_scene", ("pendulum_ball", "target_ball_on_ledge")),
    "G121": ("build_scene_by_kind", "animate_scene_by_kind", ("shown_object_ball",)),
    "G123": ("build_scene_by_kind", "animate_scene", ("hidden_ball",)),
    "G124": ("build_scene_by_kind", "animate_scene", ("shown_cube_target",)),
    "G125": ("build_scene_by_kind", "animate_scene", ("shown_ball",)),
    "G126": ("build_scene_by_kind", "animate_scene", ("shown_ball",)),
    "G127": ("build_scene", "animate_scene", ("rolling_ball",)),
    "G129": ("build_scene_by_kind", "animate_scene", ("falling_ball",)),
    "G130": ("build_scene_by_kind", "animate_scene", ("dropping_ball",)),
    "G131": ("build_scene_by_kind", "animate_scene_by_kind", ("ball_in_lower_drawer",)),
    "G133": ("build_scene_by_kind", "animate_scene", ("striker_ball",)),
    "G134": ("build_scene_by_kind", "animate_scene", ("resting_ball", "striker_ball")),
    "G135": ("build_scene_by_kind", "animate_scene", ("dropping_ball", "launched_ball")),
    "G136": ("build_scene_by_kind", "animate_scene", ("moving_ball", "swinging_ball")),
    "G137": (
        "build_scene_by_kind",
        "animate_scene",
        ("cue_ball", "rack_ball_0", "rack_ball_1", "rack_ball_2", "rack_ball_3", "rack_ball_4", "rack_ball_5"),
    ),
    "G138": ("build_scene_by_kind", "animate_scene", ("striker_ball", "target_ball")),
    "G139": ("build_scene_by_kind", "animate_scene", ("moving_ball",)),
    "G140": ("build_scene_by_kind", "animate_scene", ("heavy_ball",)),
    "G141": ("build_scene_by_kind", "animate_scene", ("target_ball",)),
    "G142": ("build_scene_by_kind", "animate_scene", ("resting_ball", "striker_ball")),
    "G143": ("build_scene_by_kind", "animate_scene", ("moving_ball",)),
    "G144": ("build_scene_by_kind", "animate_scene_by_kind", ("colored_ball_inside_box",)),
    "G145": ("build_scene_by_kind", "animate_scene", ("rolling_ball",)),
    "G146": ("build_scene_by_kind", "animate_scene", ("hanging_ball",)),
    "G147": ("build_scene_by_kind", "animate_scene", ("falling_ball",)),
    "G148": ("build_scene_by_kind", "animate_scene", ("falling_ball",)),
    "G149": ("build_scene_by_kind", "animate_scene", ("falling_ball",)),
    "G150": ("build_scene_by_kind", "animate_scene", ("falling_ball",)),
    # Fixed orthographic nominal entries, certified under opaque-pixel-rays-v2.
    "G08": ("build_scene", "animate_scene", ("orange_ball_inside_drawer",)),
    "G09": ("build_scene_by_kind", "animate_scene_by_kind", ("red_tall_body", "red_tall_roof")),
    "G10": ("build_scene", "animate_scene", ("target_orange_ball",)),
    "G101": ("build_scene_by_kind", "animate_scene_by_kind", ("rolling_ball",)),
    "G102": ("build_scene_by_kind", "animate_scene_by_kind", ("rolling_ball",)),
    "G109": ("build_scene", "animate_scene", ("relocating_orange_ball",)),
    "G11": ("build_scene", "animate_scene", ("inner_orange_ball",)),
    "G110": ("build_scene_by_kind", "animate_scene_by_kind", ("pass_pink_ball", "blocked_teal_ball")),
    "G112": ("build_scene_by_kind", "animate_scene_by_kind", ("redirect_orange_ball",)),
    "G12": ("build_scene", "animate_scene", ("inner_orange_ball", "inner_blue_cube", "inner_yellow_pyramid")),
    "G122": ("build_scene_by_kind", "animate_scene_by_kind", ("rolling_ball",)),
    "G128": ("build_scene_by_kind", "animate_scene_by_kind", ("rolling_ball",)),
    "G15": ("build_scene_by_kind", "animate_scene_by_kind", ("red_ball", "blue_ball", "yellow_ball")),
    "G16": ("build_scene_by_kind", "animate_scene_by_kind", ("orange_ball", "blue_cube", "yellow_pyramid")),
    "G17": ("build_scene", "animate_scene", ("tray_orange_ball", "tray_blue_cube")),
    "G18": ("build_scene", "animate_scene", ("blue_cube_on_turntable",)),
    "G19": ("build_scene", "animate_scene", ("orange_ball", "blue_cube", "yellow_pyramid")),
    "G20": ("build_scene", "animate_scene", ("left_orange_ball", "right_blue_cube")),
    "G21": ("build_scene", "animate_scene", ("center_orange_ball",)),
    "G22": ("build_scene", "animate_scene", ("orange_ball", "blue_cube", "yellow_pyramid")),
    "G23": ("build_scene", "animate_scene", ("orange_ball_inside_cabinet", "blue_cube_inside_cabinet")),
    "G24": ("build_scene", "animate_scene", ("orange_ball", "blue_cube", "yellow_pyramid")),
    "G25": ("build_scene", "animate_scene", ("orange_ball", "blue_cube")),
    "G26": ("build_scene", "animate_scene", ("orange_ball", "blue_cube")),
    "G43": (
        "build_scene",
        "animate_scene",
        ("upper_lane_red_ball", "middle_lane_blue_ball", "lower_lane_yellow_ball"),
    ),
    "G45": ("build_scene_by_kind", "animate_scene_by_kind", ("orange_ball_inside_box",)),
    "G47": (
        "build_scene",
        "animate_scene",
        ("stage_left_orange_ball", "stage_middle_blue_cube", "stage_right_yellow_pyramid"),
    ),
    "G48": ("build_scene", "animate_scene", ("row_01_orange_ball", "row_02_blue_cube", "row_03_yellow_pyramid")),
    "G49": ("build_scene_by_kind", "animate_scene_by_kind", ("left_orange_ball", "right_blue_cube")),
    "G50": ("build_scene_by_kind", "animate_scene_by_kind", ("ramp_orange_ball",)),
    "G51": ("build_scene", "animate_scene", ("orange_ball", "blue_cube")),
    "G52": ("build_scene", "", ("orange_ball_inside_open_right_gated_box_on_ramp",)),
    "G53": ("build_scene", "animate_scene", ("red_ball", "blue_ball", "yellow_ball")),
    "G57": ("build_scene_by_kind", "animate_scene_by_kind", ("ramp_orange_ball",)),
    "G67": ("build_scene_by_kind", "animate_scene_by_kind", ("orange_ball",)),
    "G75": ("build_scene_by_kind", "animate_scene_by_kind", ("billboard_ball",)),
    "G76": ("build_scene_by_kind", "animate_scene_by_kind", ("pendulum_bob",)),
    "G77": ("build_scene_by_kind", "animate_scene_by_kind", ("chain_ball_red", "chain_ball_green", "chain_ball_blue")),
    "G78": ("build_scene_by_kind", "animate_scene_by_kind", ("rolling_disc",)),
    "G91": ("build_scene_by_kind", "animate_scene_by_kind", ("pass_orange_ball", "blocked_blue_ball")),
    "G92": ("build_scene_by_kind", "animate_scene_by_kind", ("carom_orange_ball",)),
    # Evaluated Curve/FONT entries; source objects retain their semantic identity.
    "G04": (
        "build_scene",
        "animate_scene",
        ("identity_A_red_ball_on_high_track", "identity_B_blue_ball_on_low_track"),
    ),
    "G13": (
        "build_scene",
        "animate_scene",
        (
            "drawer_1_front_label",
            "drawer_1_orange_ball",
            "drawer_2_blue_cube",
            "drawer_2_front_label",
            "drawer_3_front_label",
            "drawer_3_yellow_pyramid",
        ),
    ),
    "G14": ("build_scene", "animate_scene", ("far_yellow_ball", "middle_blue_ball", "near_red_ball")),
    "G29": ("build_scene", "animate_scene", ("target_orange_ball_on_occupied_left_u_then_straight_track",)),
    "G31": ("build_scene", "animate_scene", ("identity_A_red_ball_front_track", "identity_B_blue_ball_back_track")),
    "G46": ("build_scene", "animate_scene", ("box_A_label", "box_A_orange_ball", "box_B_blue_cube", "box_B_label")),
}


def reference_execution_context(script, task):
    """Independently read the pinned driver's empty surface-profile globals.

    This does not import the candidate context builder or apply driver hooks.
    It records only the execution context supplied before source construction.
    """
    if task not in {"G08", "G44", "G45", "G46", "G48"}:
        return None
    root = script.parents[4]
    driver = root / "object_permanence/generator/core/render.py"
    module = root / "object_permanence/generator/core/diversity.py"
    functions = runpy.run_path(str(module))
    settings = functions["load_script_diversity"](str(script), seed=0, profile="surface")
    text, applied = functions["apply_source_bindings"](script.read_text(), settings)
    if (
        settings["values"] != {}
        or settings["source_bindings"] != {}
        or settings["factorial"] is not None
        or settings["universal_viewpoint"]
        or applied != {}
        or text != script.read_text()
    ):
        raise ValueError("source reference requires unmodified surface-profile construction")
    return {
        "profile": "surface",
        "globals": {"DIVERSITY": dict(settings["values"])},
        "source_bindings": applied,
        "scope": "source construction only; no driver appearance/render hooks",
        "driver": {"file": str(driver.relative_to(root)), "sha256": sha256(driver)},
        "diversity_module": {"file": str(module.relative_to(root)), "sha256": sha256(module)},
    }


def reference_source_realization(script, task):
    """Independently validate the native scene used as source input, not candidate arrays."""
    if task not in {"G98", "G132"}:
        return None
    directory = Path(__file__).resolve().parents[2] / f"tests/fixtures/source/object_permanence/{task.lower()}/nominal"
    manifest = directory / "manifest.json"
    m = json.loads(manifest.read_text())
    if (
        m["schema"] != "stable-envs/native-source-realization-v1"
        or m["task"] != task
        or m["member"] != 0
        or m["source_commit"] != SOURCE_COMMIT
        or m["file"] != "source.blend"
        or m["source_preparation_recipe"] != dict(zip(("build", "animate"), REFERENCE_CASES[task][:2], strict=True))
        or m["source_script_sha256"] != sha256(script)
        or m["source_task_manifest_sha256"] != sha256(script.parent / "task.json")
        or m["sha256"] != sha256(directory / "source.blend")
    ):
        raise ValueError("independent native source realization mismatch")
    return {
        "schema": m["schema"],
        "file": f"tests/fixtures/source/object_permanence/{task.lower()}/nominal/source.blend",
        "sha256": m["sha256"],
        "manifest_sha256": sha256(manifest),
    }


def prepare_reference(script, task, *, member=0):
    """Independently audited source calls; no candidate preparation imports."""
    import bpy

    realization = reference_source_realization(script, task)
    if realization is not None:
        path = Path(__file__).resolve().parents[2] / realization["file"]
        m = json.loads(path.with_name("manifest.json").read_text())
        if m["versions"]["blender"] != bpy.app.version_string:
            raise ValueError("native reference Blender version mismatch")
        bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False, use_scripts=False)
        return
    build, animate, _ = REFERENCE_CASES[task] if member == 0 else REFERENCE_MEMBERS[task, member]
    if build.startswith("$prefix:"):
        # Independent text boundary plus AST removal of the sole directory setup.
        # All scene creation and animation statements execute verbatim.
        boundary = "\ndef " + build.split(":", 1)[1] + "("
        text = script.read_text()
        if text.count(boundary) != 1:
            raise ValueError("independent source construction boundary differs")
        tree = ast.parse(text.split(boundary)[0])
        tree.body = [n for n in tree.body if ast.unparse(n) != "os.makedirs(OUT_DIR, exist_ok=True)"]
        prefix_namespace = {"__file__": str(script), "__name__": "independent_source_construction"}
        exec(compile(tree, str(script), "exec"), prefix_namespace)
        return
    context = reference_execution_context(script, task)
    source = runpy.run_path(
        str(script),
        init_globals=None if context is None else context["globals"],
        run_name="independent_source_preparation",
    )
    source["clear_scene"]()
    objects = source[build]()
    if not animate:
        return
    if task == "G27":
        source[animate](bpy.context.scene, objects["support_plate"], objects["ball"])
    elif isinstance(objects, dict) and len(inspect.signature(source[animate]).parameters) > 1:
        names = inspect.signature(source[animate]).parameters
        source[animate](**{name: objects[name] for name in names})
    else:
        source[animate](*(objects if isinstance(objects, tuple) else (objects,)))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_geometry(scene, task):
    """Include source Curve/FONT identities only for explicitly audited entries."""
    types = ("MESH", "CURVE", "FONT") if task in REFERENCE_CONVERTED_TASKS else ("MESH",)
    if task in REFERENCE_CONVERTED_TASKS and any(
        obj.type not in (*types, "EMPTY", "CAMERA", "LIGHT") for obj in scene.objects
    ):
        raise ValueError("unsupported source renderable type")
    return {obj.name: obj for obj in scene.objects if obj.type in types}


def reference_geometry_description(objects, graph):
    """Capture converted source identity without candidate conversion helpers."""
    descriptions = {}
    for name, obj in objects.items():
        if obj.type not in ("CURVE", "FONT"):
            continue
        data = obj.evaluated_get(graph).data
        descriptions[name] = {"type": obj.type}
        if obj.type == "FONT":
            descriptions[name].update(text=data.body, font=data.font.name)
    return descriptions


def reference_render_meshes(objects, graph):
    """Yield actual draw meshes/world transforms, including Curve/FONT proxies.

    This reads the source dependency graph's MESH entries, not candidate
    Curve.to_mesh() results. Each yielded mesh is an independent owned copy.
    Iterator instance records are consumed before advancing the graph iterator.
    """
    import bpy

    seen = set()
    for item in graph.object_instances:
        if item.object.type != "MESH":
            continue
        source = item.object.original
        if item.is_instance:
            parent = item.parent
            if (
                parent is None
                or parent.original.type not in ("CURVE", "FONT")
                or source != parent.original
                or parent.original.instance_type != "NONE"
                or item.instance_object is None
                or item.instance_object.original != source
                or tuple(map(tuple, item.matrix_world)) != tuple(map(tuple, parent.matrix_world))
            ):
                raise ValueError("source has unsupported generated geometry instances")
        elif source.type != "MESH":
            raise ValueError("source converted geometry is not an evaluated mesh proxy")
        name = source.name
        if name not in objects or source != objects[name] or name in seen or not item.show_self:
            raise ValueError("source draw objects do not map one-to-one to semantic identities")
        seen.add(name)
        matrix = np.asarray(item.matrix_world, np.float64)
        mesh = bpy.data.meshes.new_from_object(item.object, preserve_all_data_layers=True, depsgraph=graph)
        try:
            yield name, mesh, matrix
        finally:
            bpy.data.meshes.remove(mesh)
    if seen != set(objects):
        raise ValueError("source draw geometry missing")


def write_depth_parts(output, bits):
    """Store all lossless depth bits in <=30-frame, <=1000-KiB NPZ parts.

    Bound reference file sizes without reducing frame count or precision. A part
    with high temporal change splits in half; this is independent of task ID.
    """
    parts: list[dict] = []

    def publish(start, stop):
        stream = io.BytesIO()
        np.savez_compressed(stream, depth_xor=bits[start:stop])
        content = stream.getvalue()
        if len(content) > 1000 * 1024:
            if stop - start == 1:
                raise ValueError("a single reference depth frame exceeds repository file limit")
            middle = (start + stop) // 2
            publish(start, middle)
            publish(middle, stop)
            return
        path = output / f"depth_{start:04d}_{stop:04d}.npz"
        path.write_bytes(content)
        parts.append(
            {
                "file": path.name,
                "sha256": sha256(path),
                "start": start,
                "stop": stop,
                "shape": list(bits[start:stop].shape),
                "dtype": str(bits.dtype),
            }
        )

    for start in range(0, len(bits), 30):
        publish(start, min(start + 30, len(bits)))
    return parts


def freeze(root, output, task, width=320, height=180):
    import bpy
    from mathutils import Vector

    if bpy.app.version[:2] != (4, 4):
        raise ValueError("rigid scene oracle requires Blender 4.4.x")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT or subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ):
        raise ValueError("reference source must be the clean pinned commit")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite oracle: {output}")
    if task not in REFERENCE_CASES:
        raise ValueError("unaudited independent reference entry")
    matches = list((root / "object_permanence/generator/tasks").glob(task + "_*/task.json"))
    if len(matches) != 1:
        raise ValueError("source task must be unique")
    task_manifest = matches[0]
    metadata = json.loads(task_manifest.read_text())
    if metadata["id"] != task:
        raise ValueError("source task identity differs")
    script = task_manifest.parent / metadata["member_scripts"][0]
    source_script = str(script.relative_to(root))
    build, animate, targets = REFERENCE_CASES[task]
    prepare_reference(script, task)
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.frame_set(scene.frame_start)
    bpy.context.view_layer.update()
    camera = scene.camera
    world = camera.matrix_world.copy()
    corners = camera.data.view_frame(scene=scene)
    xmax, ymax = max(p.x / -p.z for p in corners), max(p.y / -p.z for p in corners)
    origin, rotation = world.translation.copy(), world.to_3x3()
    forward = rotation @ Vector((0, 0, -1))
    directions = [
        (rotation @ Vector(((2 * (x + 0.5) / width - 1) * xmax, (1 - 2 * (y + 0.5) / height) * ymax, -1))).normalized()
        for y in range(height)
        for x in range(width)
    ]
    names = sorted(o.name for o in scene.objects if o.type == "MESH")
    labels = {name: index + 1 for index, name in enumerate(names)}
    if not set(targets) <= set(names):
        raise ValueError("audited reference targets missing")
    frames = np.arange(scene.frame_start, scene.frame_end + 1, dtype=np.int64)
    depths = np.zeros((len(frames), height, width), np.float32)
    ids = np.zeros((len(frames), height, width), np.int16)
    poses = np.empty((len(frames), len(names), 7), np.float64)
    for row, frame in enumerate(frames):
        scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        if not np.array_equal(world, scene.camera.matrix_world):
            raise ValueError("reference camera changed")
        for index, name in enumerate(names):
            matrix = bpy.data.objects[name].evaluated_get(depsgraph).matrix_world
            poses[row, index] = [*matrix.translation, *matrix.to_quaternion()]
        for pixel, direction in enumerate(directions):
            hit, location, _, _, obj, _ = scene.ray_cast(depsgraph, origin, direction, distance=camera.data.clip_end)
            if hit:
                y, x = divmod(pixel, width)
                depths[row, y, x] = (location - origin).dot(forward)
                ids[row, y, x] = labels[obj.name]
        if row % 10 == 0 or row == len(frames) - 1:
            print(
                f"{task} source frame {frame}: target pixels "
                + str([int((ids[row] == labels[n]).sum()) for n in targets]),
                flush=True,
            )
    arrays = {
        "source_frames": frames,
        "poses": poses,
        "depth_base": depths[0].copy(),
        "depth_xor": depths.view(np.uint32) ^ depths[0].view(np.uint32),
        "object_ids": ids,
        "camera_matrix": np.asarray(world, np.float64),
    }
    reconstructed = (arrays["depth_xor"] ^ arrays["depth_base"].view(np.uint32)).view(np.float32)
    np.testing.assert_array_equal(reconstructed, depths)
    output.mkdir(parents=True)
    archive = output / f"blender_{task.lower()}_nominal.npz"
    # Chunk only lossless depth bits, never downsample frames or reduce precision.
    # The moving board changes many pixels; one monolithic NPZ exceeds repo limits.
    depth_bits = arrays.pop("depth_xor")
    np.savez_compressed(archive, **arrays)
    depth_parts = write_depth_parts(output, depth_bits)
    manifest = {
        "schema": "stable-envs/object-permanence-rigid-oracle-v1",
        "task": task,
        "source_url": "https://github.com/hokindeng/object-permanence",
        "source_commit": commit,
        "source_script": source_script,
        "source_task_manifest_sha256": sha256(task_manifest),
        "source_script_sha256": sha256(script),
        "source_profile": "nominal first member; no diversity wrapper",
        "source_license": "CC-BY-NC-4.0",
        "generator": "scripts/testing/freeze_object_permanence.py",
        "generator_sha256": sha256(Path(__file__)),
        "versions": {"blender": bpy.app.version_string, "numpy": np.__version__, "python": platform.python_version()},
        "file": archive.name,
        "sha256": sha256(archive),
        "width": width,
        "height": height,
        "fps": float(scene.render.fps / scene.render.fps_base),
        "object_ids": labels,
        "object_order": names,
        "targets": list(targets),
        "target_visible_pixels": {name: (ids == labels[name]).sum(axis=(1, 2)).tolist() for name in targets},
        "depth_encoding": "float32 bitwise XOR against depth_base; lossless chunks of at most 30 frames",
        "depth_parts": depth_parts,
        "depth_convention": "positive camera-axis distance in source scene units; zero for no hit",
        "mask_convention": "visible source mesh at the pixel center; zero for no hit",
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
    }
    context = reference_execution_context(script, task)
    if context is not None:
        manifest["source_profile"] = "nominal first member; surface diversity globals; no driver hooks"
        manifest["source_execution_context"] = context
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "SOURCE_LICENSE.txt").write_text((root / "LICENSE").read_text())
    (output / "README.md").write_text(
        f"# Blender {task} nominal geometry oracle\n\n"
        "All 120 source frames at 320x180, generated independently from pinned Blender 4.4.\n"
        "No candidate exporter, recording or renderer is imported. Source, camera dimensions,\n"
        "IDs, units, array shapes, hashes and versions are in manifest.json.\n"
        "Depth XOR chunks preserve every float32 bit; shader RGB is not an oracle.\n\n"
        "Regenerate in a new directory with `python -m scripts.testing.freeze_object_permanence\n"
        f"--source-root <pinned-checkout> --output <new-directory> --task {task}` (one command).\n"
        "Existing fixtures are never overwritten. CC-BY-NC-4.0 source-derived data;\n"
        "see SOURCE_LICENSE.txt for full attribution and license terms.\n"
    )
    print(f"frozen {archive}: {archive.stat().st_size} bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(REFERENCE_CASES), required=True)
    args = parser.parse_args()
    freeze(args.source_root.resolve(), args.output.resolve(), args.task)


if __name__ == "__main__":
    main()
