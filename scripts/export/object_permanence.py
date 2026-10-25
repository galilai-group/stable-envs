"""Export evaluated rigid Blender scenes without rendering videos.

Run the inventory with ordinary Python, or export with the existing isolated
Python 3.11 + bpy 4.4 reference interpreter. Only explicitly audited preparation
entry points run; source main(), render(), save() and diversity wrappers do not.
Audited nominal members share the same extraction and recording code. Explicitly
registered source contexts supply the official surface profile's DIVERSITY
globals without executing the driver's rendering or appearance hooks.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import io
import json
import platform
import re
import runpy
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SOURCE_COMMIT = "824baddcdad087810d35638bf8f273b884f74975"
SCHEMA = "stable-envs/object-permanence-rigid-v1"
SCHEMA_V2 = "stable-envs/object-permanence-rigid-v2"
CONVENTIONS = {"world": "right-handed-z-up", "quaternion": "wxyz", "units": "source-scene-units"}
# Context audit supplies nominal globals independently of camera/type certification.
SOURCE_CONTEXT_TASKS = frozenset({"G08", "G44", "G45", "G46", "G48"})
# Certified orthographic entries are bound to the separate v2 visibility gate.
ORTHOGRAPHIC_TASKS = {
    "G08",
    "G09",
    "G10",
    "G101",
    "G102",
    "G109",
    "G11",
    "G110",
    "G112",
    "G12",
    "G122",
    "G128",
    "G15",
    "G16",
    "G17",
    "G18",
    "G19",
    "G20",
    "G21",
    "G22",
    "G23",
    "G24",
    "G25",
    "G26",
    "G43",
    "G45",
    "G47",
    "G48",
    "G49",
    "G50",
    "G51",
    "G52",
    "G53",
    "G57",
    "G67",
    "G75",
    "G76",
    "G77",
    "G78",
    "G91",
    "G92",
}
AUDITED_TASKS = {
    "G02": ("$prefix:render_png", ""),
    "G03": ("$prefix:render_png", ""),
    "G05": ("build_scene", "animate_scene"),
    "G06": ("build_scene", "animate_scene"),
    "G07": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G27": ("build_scene", "animate_scene"),
    "G34": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G62": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G37": ("build_scene_by_kind", "animate_scene"),
    "G38": ("build_scene_by_kind", "animate_scene"),
    "G64": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G55": ("build_scene_by_kind", "animate_scene"),
    "G63": ("build_scene_by_kind", "animate_scene"),
    "G66": ("build_scene_by_kind", "animate_scene"),
    "G80": ("build_scene_by_kind", "animate_scene"),
    "G81": ("build_scene_by_kind", "animate_scene"),
    "G65": ("build_scene_by_kind", "animate_scene"),
    "G83": ("build_scene_by_kind", "animate_scene"),
    "G82": ("build_scene_by_kind", "animate_scene"),
    "G84": ("build_scene_by_kind", "animate_scene"),
    "G86": ("build_scene_by_kind", "animate_scene"),
    "G85": ("build_scene_by_kind", "animate_scene"),
    "G87": ("build_scene_by_kind", "animate_scene"),
    "G88": ("build_scene_by_kind", "animate_scene"),
    "G89": ("build_scene_by_kind", "animate_scene"),
    "G93": ("build_scene_by_kind", "animate_scene"),
    "G94": ("build_scene_by_kind", "animate_scene"),
    "G95": ("build_scene_by_kind", "animate_scene"),
    "G96": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G97": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G99": ("build_scene_by_kind", "animate_scene"),
    "G100": ("build_scene_by_kind", "animate_scene"),
    "G103": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G28": ("build_scene", "animate_scene"),
    "G30": ("build_scene", "animate_scene"),
    "G32": ("build_scene", "animate_scene"),
    "G33": ("build_scene", "animate_scene"),
    "G35": ("build_scene", "animate_scene"),
    "G39": ("build_scene_by_kind", "animate_scene"),
    "G40": ("build_scene_by_kind", "animate_scene"),
    "G41": ("build_scene_by_kind", "animate_scene"),
    "G42": ("build_scene_by_kind", "animate_scene"),
    "G44": ("build_scene_by_kind", "animate_scene"),
    "G56": ("build_scene_by_kind", "animate_scene"),
    "G58": ("build_scene_by_kind", "animate_scene"),
    "G59": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G60": ("build_scene_by_kind", "animate_scene"),
    "G68": ("build_scene_by_kind", "animate_scene"),
    "G69": ("build_scene_by_kind", "animate_scene"),
    "G70": ("build_scene_by_kind", "animate_scene"),
    "G71": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G72": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G73": ("build_scene_by_kind", "animate_scene"),
    "G74": ("build_scene_by_kind", "animate_scene"),
    "G79": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G104": ("build_scene_by_kind", "animate_scene"),
    "G105": ("build_scene_by_kind", "animate_scene"),
    "G106": ("build_scene_by_kind", "animate_scene"),
    "G107": ("build_scene_by_kind", "animate_scene"),
    "G108": ("build_scene_by_kind", "animate_scene"),
    "G111": ("build_scene", "animate_scene"),
    "G113": ("build_scene_by_kind", "animate_scene"),
    "G114": ("build_scene_by_kind", "animate_scene"),
    "G115": ("build_scene_by_kind", "animate_scene"),
    "G116": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G117": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G118": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G119": ("build_scene_by_kind", "animate_scene"),
    "G120": ("build_scene_by_kind", "animate_scene"),
    "G121": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G123": ("build_scene_by_kind", "animate_scene"),
    "G124": ("build_scene_by_kind", "animate_scene"),
    "G125": ("build_scene_by_kind", "animate_scene"),
    "G126": ("build_scene_by_kind", "animate_scene"),
    "G127": ("build_scene", "animate_scene"),
    "G129": ("build_scene_by_kind", "animate_scene"),
    "G130": ("build_scene_by_kind", "animate_scene"),
    "G131": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G133": ("build_scene_by_kind", "animate_scene"),
    "G134": ("build_scene_by_kind", "animate_scene"),
    "G135": ("build_scene_by_kind", "animate_scene"),
    "G136": ("build_scene_by_kind", "animate_scene"),
    "G137": ("build_scene_by_kind", "animate_scene"),
    "G138": ("build_scene_by_kind", "animate_scene"),
    "G139": ("build_scene_by_kind", "animate_scene"),
    "G140": ("build_scene_by_kind", "animate_scene"),
    "G141": ("build_scene_by_kind", "animate_scene"),
    "G142": ("build_scene_by_kind", "animate_scene"),
    "G143": ("build_scene_by_kind", "animate_scene"),
    "G144": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G145": ("build_scene_by_kind", "animate_scene"),
    "G147": ("build_scene_by_kind", "animate_scene"),
    "G148": ("build_scene_by_kind", "animate_scene"),
    "G149": ("build_scene_by_kind", "animate_scene"),
    "G150": ("build_scene_by_kind", "animate_scene"),
    # Fixed orthographic nominal entries, certified under opaque-pixel-rays-v2.
    "G08": ("build_scene", "animate_scene"),
    "G09": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G10": ("build_scene", "animate_scene"),
    "G101": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G102": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G109": ("build_scene", "animate_scene"),
    "G11": ("build_scene", "animate_scene"),
    "G110": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G112": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G12": ("build_scene", "animate_scene"),
    "G122": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G128": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G15": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G16": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G17": ("build_scene", "animate_scene"),
    "G18": ("build_scene", "animate_scene"),
    "G19": ("build_scene", "animate_scene"),
    "G20": ("build_scene", "animate_scene"),
    "G21": ("build_scene", "animate_scene"),
    "G22": ("build_scene", "animate_scene"),
    "G23": ("build_scene", "animate_scene"),
    "G24": ("build_scene", "animate_scene"),
    "G25": ("build_scene", "animate_scene"),
    "G26": ("build_scene", "animate_scene"),
    "G43": ("build_scene", "animate_scene"),
    "G45": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G47": ("build_scene", "animate_scene"),
    "G48": ("build_scene", "animate_scene"),
    "G49": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G50": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G51": ("build_scene", "animate_scene"),
    "G52": ("build_scene", ""),
    "G53": ("build_scene", "animate_scene"),
    "G57": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G67": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G75": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G76": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G77": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G78": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G91": ("build_scene_by_kind", "animate_scene_by_kind"),
    "G92": ("build_scene_by_kind", "animate_scene_by_kind"),
}
# These preparation recipes are certified only with the versioned ray contract.
# The legacy BVH deferrals above/below remain meaningful and are never erased.
RAY_CONTRACT_TASKS = {
    "G02",
    "G03",
    "G05",
    "G06",
    "G07",
    "G28",
    "G30",
    "G32",
    "G33",
    "G35",
    "G39",
    "G40",
    "G41",
    "G42",
    "G44",
    "G56",
    "G58",
    "G59",
    "G60",
    "G63",
    "G65",
    "G66",
    "G68",
    "G69",
    "G70",
    "G71",
    "G72",
    "G73",
    "G74",
    "G79",
    "G80",
    "G81",
    "G82",
    "G83",
    "G84",
    "G85",
    "G86",
    "G87",
    "G88",
    "G89",
    "G93",
    "G94",
    "G95",
    "G96",
    "G97",
    "G99",
    "G100",
    "G103",
    "G104",
    "G105",
    "G106",
    "G107",
    "G108",
    "G111",
    "G113",
    "G114",
    "G115",
    "G116",
    "G117",
    "G118",
    "G119",
    "G120",
    "G121",
    "G123",
    "G124",
    "G125",
    "G126",
    "G127",
    "G129",
    "G130",
    "G131",
    "G133",
    "G134",
    "G135",
    "G136",
    "G137",
    "G138",
    "G139",
    "G140",
    "G141",
    "G142",
    "G143",
    "G144",
    "G145",
    "G147",
    "G148",
    "G149",
    "G150",
}
CONTRACT_ONLY_TASKS = {
    "G02",
    "G03",
    "G05",
    "G06",
    "G07",
    "G28",
    "G30",
    "G32",
    "G33",
    "G35",
    "G39",
    "G40",
    "G41",
    "G42",
    "G44",
    "G56",
    "G58",
    "G59",
    "G60",
    "G65",
    "G68",
    "G69",
    "G70",
    "G71",
    "G72",
    "G73",
    "G74",
    "G79",
    "G82",
    "G83",
    "G84",
    "G85",
    "G86",
    "G87",
    "G88",
    "G89",
    "G93",
    "G94",
    "G95",
    "G96",
    "G97",
    "G99",
    "G100",
    "G103",
    "G104",
    "G105",
    "G106",
    "G107",
    "G108",
    "G111",
    "G113",
    "G114",
    "G115",
    "G116",
    "G117",
    "G118",
    "G119",
    "G120",
    "G121",
    "G123",
    "G124",
    "G125",
    "G126",
    "G127",
    "G129",
    "G130",
    "G131",
    "G133",
    "G134",
    "G135",
    "G136",
    "G137",
    "G138",
    "G139",
    "G140",
    "G141",
    "G142",
    "G143",
    "G144",
    "G145",
    "G147",
    "G148",
    "G149",
    "G150",
}
CONTRACT_ONLY_TASKS |= ORTHOGRAPHIC_TASKS
# Preparation/rigid serialization passed, but source pixel fidelity did not.
# Explicit export remains useful for diagnosis; default batch execution is blocked.
DEFERRED_TASKS = {
    "G63": "source fidelity deferred: boundary depth MSE exceeds 1e-8 at frames 16/54/66/104",
    "G66": "source fidelity deferred: boundary depth MSE exceeds 1e-8 at frame 34",
    "G80": "source fidelity deferred: coplanar door mask IoU below .99 at frame 54",
    "G81": "source fidelity deferred: coplanar housing/rail mask IoU below .99 at frame 1",
}
G34_ORDER = (
    "large_floor_base",
    "rear_backdrop_panel",
    "straight_gray_track",
    "wide_opaque_screen",
    "small_orange_ball",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(arrays):
    # Same wire-format rule as trajectory.py; no stable_envs/Torch import is
    # needed in Blender's isolated interpreter. Tests check interoperability.
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        stream = io.BytesIO()
        np.save(stream, array, allow_pickle=False)
        digest.update(name.encode() + b"\0" + stream.getvalue())
    return digest.hexdigest()


def check_source(root):
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"expected source {SOURCE_COMMIT}, got {commit}")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True
    )
    if dirty:
        raise ValueError("source checkout must be unmodified, including untracked files")


def task_manifests(root):
    tasks = root / "object_permanence/generator/tasks"
    return sorted(tasks.glob("*/task.json"), key=lambda p: int(json.loads(p.read_text())["id"][1:]))


def capability_group(tree, *, audited=False):
    """Conservative source-only review queue; never permission to execute a script.

    Whole-file hints include unused builders. Only an audited entry plus all-frame
    runtime checks can certify the selected scene. No source code is executed here.
    """
    if audited:
        return "audited-nominal-runtime-gate"
    constants = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    transparent = any(
        (
            k.arg == "alpha"
            and isinstance(k.value, ast.Constant)
            and isinstance(k.value.value, (int, float))
            and k.value.value < 1
        )
        or (k.arg == "transparent" and isinstance(k.value, ast.Constant) and k.value.value is True)
        for call in calls
        for k in call.keywords
    )
    if transparent or attributes & {"hide_render", "hide_viewport"}:
        return "appearance-or-visibility-review"
    if "ORTHO" in constants:
        return "orthographic-camera-review"
    animated = {
        k.value.value
        for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "keyframe_insert"
        for k in call.keywords
        if k.arg == "data_path" and isinstance(k.value, ast.Constant)
    }
    if animated & {"scale", "dimensions", "value"} or "shape_key_add" in attributes:
        return "geometry-or-scale-review"
    if "rigidbody" in attributes or any(str(value).startswith("rigid_body") for value in animated):
        return "physics-bake-review"
    return "rigid-candidate-needs-entry-audit"


def inventory(root, output):
    """List all source members and script-wide signals, not runtime certification.

    Several source files contain builders for unused scene kinds. These flags
    deliberately overapproximate their requirements; they cannot certify that
    the selected nominal scene, let alone diversity variants, is rigid.
    """
    check_source(root)
    rows, members = [], 0
    audit_path = Path(__file__).resolve().parents[2] / "docs/adding/object-permanence-audit.json"
    measured = {}
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit["source_commit"] != SOURCE_COMMIT:
            raise ValueError("measured source audit has a different source pin")
        measured = {entry["task"]: entry for entry in audit["entries"]}
    groups: dict[str, int] = {}
    for path in task_manifests(root):
        task = json.loads(path.read_text())
        signals, review_groups = set(), []
        for name in task["member_scripts"]:
            members += 1
            tree = ast.parse((path.parent / name).read_text())
            group = capability_group(tree, audited=task["id"] in AUDITED_TASKS and name == task["member_scripts"][0])
            if task["id"] in DEFERRED_TASKS and name == task["member_scripts"][0]:
                group = "source-fidelity-review"
            elif task["id"] in CONTRACT_ONLY_TASKS and name == task["member_scripts"][0]:
                group = "versioned-visibility-runtime-gate"
            groups[group] = groups.get(group, 0) + 1
            review_groups.append(group)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr == "keyframe_insert":
                        for keyword in node.keywords:
                            if keyword.arg == "data_path" and isinstance(keyword.value, ast.Constant):
                                signals.add(str(keyword.value.value))
                    if (
                        node.func.attr == "new"
                        and isinstance(node.func.value, ast.Attribute)
                        and node.func.value.attr == "modifiers"
                    ):
                        signals.add("modifiers")
                    if node.func.attr in ("rigidbody", "object_add") and "rigidbody" in ast.unparse(node.func):
                        signals.add("rigidbody")
                    if node.func.attr == "shape_key_add":
                        signals.add("shape_keys")
                if isinstance(node, ast.Attribute) and node.attr in ("hide_render", "hide_viewport"):
                    signals.add(node.attr)
        pilot = (
            "versioned ray contract only"
            if task["id"] in CONTRACT_ONLY_TASKS
            else "source fidelity deferred"
            if task["id"] in DEFERRED_TASKS
            else "nominal export available"
            if task["id"] in AUDITED_TASKS
            else "not evaluated"
        )
        result = measured.get(task["id"])
        outcome = "not recorded" if result is None else result["status"]
        if result is not None and result["status"] == "deferred":
            outcome += ": " + result["category"]
        rows.append(
            f"| {task['id']} {task['name']} | {len(task['member_scripts'])} | {', '.join(sorted(signals)) or 'none found'} | {review_groups[0]} | {pilot} | {outcome} |"
        )
    text = (
        "# Object-permanence source inventory\n\n"
        f"Pinned source: `{SOURCE_COMMIT}`; {len(rows)} task groups, {members} member scripts.\n\n"
        "Regenerate with `python -m scripts.export.object_permanence inventory --source-root <checkout> --output <file>`.\n"
        "This is a static, script-wide inventory. Unused builders can contribute signals;\n"
        "absence of a signal does not prove support. Counts include all members.\n"
        "Only explicitly listed first nominal members have audited preparation entry points;\n"
        "run export to check their evaluated geometry, transforms, materials and camera at every frame.\n"
        "Diversity variants, sampled scene semantics and downstream accuracy are not certified here.\n\n"
        "The measured nominal column comes from the separately recorded "
        "[source audit](object-permanence-audit.json), not from static hints. "
        "It covers first nominal members; other member scripts remain unregistered.\n\n"
        "## Capability triage\n\n"
        "Groups cover every member. Static groups are conservative AST hints, not proof of support.\n"
        "Known source-fidelity failures form a separate measured deferral queue.\n"
        "Unused helpers may trigger a flag. Remaining priority is audited nominal, visibility/appearance,\n"
        "orthographic, geometry/scale, physics bake, then rigid candidate requiring entry audit.\n"
        "An applied, time-invariant modifier is allowed; an animated/deforming mesh is not.\n\n"
        "| Review group | Members |\n|---|---:|\n"
        + "".join(f"| {group} | {count} |\n" for group, count in sorted(groups.items()))
        + "\n## Source tasks\n\n"
        "| Task | Members | Script-wide signals | First-member review group | Nominal preparation coverage | Measured nominal audit |\n"
        "|---|---:|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )
    output.write_text(text)


def source_prefix(script, boundary):
    """Execute an audited legacy construction prefix, excluding output setup.

    These pinned scripts finish construction/animation before render_png. Do not
    import the complete module: its remaining top-level statements render, delete
    output directories and save .blend/JSON files. No scene statement is removed.
    """
    statements = []
    for statement in ast.parse(script.read_text()).body:
        if isinstance(statement, ast.FunctionDef) and statement.name == boundary:
            break
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if ast.unparse(call.func) == "os.makedirs":
                if ast.unparse(call) != "os.makedirs(OUT_DIR, exist_ok=True)":
                    raise ValueError("unaudited source output setup")
                continue
        statements.append(statement)
    else:
        raise ValueError("audited construction boundary missing")
    namespace = {"__file__": str(script), "__name__": "audited_source_construction"}
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(script), "exec"), namespace)
    return namespace


def source_execution_context(root, task, script):
    """Describe audited driver globals, with no source rewriting or postprocessing.

    The pinned driver's surface profile supplies an empty DIVERSITY mapping;
    task factors therefore take their script-defined defaults. This is only its
    source-execution context, not a claim to reproduce its RGB surface hooks.
    Older entries keep their original namespace and provenance unchanged.
    """
    if task not in SOURCE_CONTEXT_TASKS:
        return None
    module = root / "object_permanence/generator/core/diversity.py"
    driver = root / "object_permanence/generator/core/render.py"
    api = runpy.run_path(str(module))
    config = api["load_script_diversity"](str(script), 0, "surface")
    transformed, bindings = api["apply_source_bindings"](script.read_text(), config)
    if (
        config["values"]
        or config["source_bindings"]
        or config["factorial"] is not None
        or config["universal_viewpoint"]
        or bindings
        or transformed != script.read_text()
    ):
        raise ValueError("audited surface execution context must not vary or rewrite the source")
    return {
        "profile": "surface",
        "globals": {"DIVERSITY": dict(config["values"])},
        "source_bindings": bindings,
        "scope": "source construction only; no driver appearance/render hooks",
        "driver": {"file": str(driver.relative_to(root)), "sha256": sha256(driver)},
        "diversity_module": {"file": str(module.relative_to(root)), "sha256": sha256(module)},
    }


def prepare(root, task):
    """Execute an audited source build/animate pair, without source I/O hooks."""
    import bpy

    if bpy.app.version[:2] != (4, 4):
        raise ValueError("export requires Blender 4.4.x")
    if task not in AUDITED_TASKS:
        raise ValueError(f"unaudited preparation entry point: {task}")
    check_source(root)
    paths = [p for p in task_manifests(root) if json.loads(p.read_text())["id"] == task]
    if len(paths) != 1:
        raise ValueError(f"source task not found: {task}")
    path = paths[0]
    metadata = json.loads(path.read_text())
    source = path.parent / metadata["member_scripts"][0]
    build, animate = AUDITED_TASKS[task]
    if build.startswith("$prefix:"):
        source_prefix(source, build.removeprefix("$prefix:"))
        return bpy.context.scene, source, metadata
    context = source_execution_context(root, task, source)
    namespace = runpy.run_path(
        str(source), init_globals=None if context is None else context["globals"], run_name="stable_envs_rigid_export"
    )
    namespace["clear_scene"]()
    result = namespace[build]()
    if not animate:
        # An audited builder may already create the complete animation, as in
        # source main() calling only build_scene(). Do not run it a second time.
        return bpy.context.scene, source, metadata
    if task == "G27":
        namespace[animate](bpy.context.scene, result["support_plate"], result["ball"])
    elif isinstance(result, dict) and len(inspect.signature(namespace[animate]).parameters) > 1:
        parameters = inspect.signature(namespace[animate]).parameters
        namespace[animate](**{name: result[name] for name in parameters})
    else:
        namespace[animate](*(result if isinstance(result, tuple) else (result,)))
    return bpy.context.scene, source, metadata


def _mesh(obj, depsgraph, scale):
    """Copy evaluated local triangles/corner normals, baking constant scale once."""
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        vertices = np.asarray([v.co[:] for v in mesh.vertices], np.float64)
        triangles = np.asarray([t.vertices[:] for t in mesh.loop_triangles], np.int32)
        loops = np.asarray([t.loops[:] for t in mesh.loop_triangles], np.int32)
        normals = np.asarray([n.vector[:] for n in mesh.corner_normals], np.float64)
        if not len(triangles):
            raise ValueError(f"empty mesh: {obj.name}")
        # Split only at position/normal discontinuities; flat boxes keep sharp
        # normals and smooth spheres can still share vertices. No tessellation
        # or primitive-shape approximation is introduced by the exporter.
        corners = np.concatenate(
            [vertices[triangles.reshape(-1)] * scale, normals[loops.reshape(-1)] / scale], axis=-1
        )
        corners[:, 3:] /= np.linalg.norm(corners[:, 3:], axis=-1, keepdims=True)
        unique, inverse = np.unique(corners.astype(np.float32), axis=0, return_inverse=True)
        faces = inverse.reshape(-1, 3).astype(np.int32)
        # Blender can enumerate identical sphere triangles in a different order
        # between builds. Canonicalize cyclic starts and face order, preserving
        # winding and every value. Identical meshes then share a content key.
        starts = faces.argmin(axis=1)
        faces = np.take_along_axis(faces, (starts[:, None] + np.arange(3)) % 3, axis=1)
        faces = faces[np.lexsort(faces.T[::-1])]
        return {
            "vertices": np.ascontiguousarray(unique[:, :3]),
            "normals": np.ascontiguousarray(unique[:, 3:]),
            "triangles": np.ascontiguousarray(faces),
        }
    finally:
        evaluated.to_mesh_clear()


def _pose(obj, depsgraph):
    world = obj.evaluated_get(depsgraph).matrix_world
    if not np.isfinite(np.asarray(world)).all():
        raise ValueError(f"nonfinite world transform: {obj.name}")
    position, rotation, scale = world.decompose()
    rotation.normalize()
    scale = np.asarray(scale, np.float64)
    if np.any(scale <= 0) or not np.allclose(
        np.asarray(world)[:3, :3], np.asarray(rotation.to_matrix()) * scale, atol=2e-6, rtol=0
    ):
        raise ValueError(f"non-rigid shear/reflection: {obj.name}")
    return np.asarray([*position, *rotation], np.float32), scale


def _rgba(obj):
    if len(obj.material_slots) != 1 or obj.active_material is None:
        raise ValueError(f"v1 requires one opaque material: {obj.name}")
    material = obj.active_material
    color = np.asarray(material.diffuse_color, np.float32)
    if material.use_nodes:
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf is None or any(n.type == "TEX_IMAGE" for n in material.node_tree.nodes):
            raise ValueError(f"unsupported material nodes: {obj.name}")
        sockets = [bsdf.inputs[k] for k in ("Base Color", "Alpha", "Transmission Weight")]
        if any(s.is_linked for s in sockets) or sockets[1].default_value != 1 or sockets[2].default_value != 0:
            raise ValueError(f"unsupported transparency/material links: {obj.name}")
        color = np.asarray(sockets[0].default_value, np.float32)
    if not np.isfinite(color).all() or color[3] != 1:
        raise ValueError(f"unsupported transparency: {obj.name}")
    return color


def _camera(scene, depsgraph, *, allow_orthographic=False):
    camera = scene.camera.evaluated_get(depsgraph)
    if camera.data.type not in (("PERSP", "ORTHO") if allow_orthographic else ("PERSP",)) or camera.data.dof.use_dof:
        message = (
            "v2 requires a standard perspective or orthographic camera without depth of field"
            if allow_orthographic
            else "v1 requires a perspective camera without depth of field"
        )
        raise ValueError(message)
    basis = np.asarray(camera.matrix_world, np.float64)[:3, :3]
    if not np.allclose(basis.T @ basis, np.eye(3), atol=2e-6, rtol=0) or np.linalg.det(basis) <= 0:
        raise ValueError("v1 requires a rigid camera transform")
    width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
    projection = camera.calc_matrix_camera(
        depsgraph,
        x=width,
        y=height,
        scale_x=scene.render.pixel_aspect_x,
        scale_y=scene.render.pixel_aspect_y,
    )
    result = {
        "world": np.asarray(camera.matrix_world, np.float64).tolist(),
        "projection": np.asarray(projection, np.float64).tolist(),
        "width": width,
        "height": height,
        "near": float(camera.data.clip_start),
        "far": float(camera.data.clip_end),
    }
    if camera.data.type == "ORTHO":
        result["projection_type"] = "orthographic"
    return result


def sample_scene(scene, *, object_order=None, allow_orthographic=False):
    """Check every frame and return a rigid recording, or fail explicitly.

    Motion comes exclusively from evaluated source data. Constant positive scale
    is baked into local geometry; deformation, moving camera, visibility changes,
    instances and changing base colors require a future schema/implementation.
    Metadata pb_is_dynamic does not substitute for observing all frames.
    """
    import bpy

    if scene.camera is None:
        raise ValueError("scene has no camera")
    fps = float(scene.render.fps / scene.render.fps_base)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("scene fps must be finite and positive")
    objects = sorted((o for o in scene.objects if o.type == "MESH"), key=lambda o: o.name)
    if object_order is not None:
        by_name = {o.name: o for o in objects}
        if set(by_name) != set(object_order):
            raise ValueError("audited object set differs")
        objects = [by_name[n] for n in object_order]
    if not objects or any(o.type not in ("MESH", "EMPTY", "CAMERA", "LIGHT") for o in scene.objects):
        raise ValueError("unsupported renderable object type")
    frames = np.arange(scene.frame_start, scene.frame_end + 1, dtype=np.int64)
    if not len(frames):
        raise ValueError("empty frame range")

    def hidden_collection(collection):
        return collection.hide_render or any(hidden_collection(child) for child in collection.children)

    original = scene.frame_current
    poses = np.empty((len(frames), len(objects), 7), np.float32)
    initial = None
    try:
        for row, frame in enumerate(frames):
            scene.frame_set(int(frame))
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            if any(i.is_instance for i in depsgraph.object_instances):
                raise ValueError("v1 does not support generated instances")
            if {o.name for o in scene.objects if o.type == "MESH"} != {o.name for o in objects}:
                raise ValueError("object set changes")
            if hidden_collection(scene.collection) or any(o.hide_render or not o.visible_get() for o in objects):
                raise ValueError("v1 does not support hidden objects/visibility changes")
            camera = _camera(scene, depsgraph, allow_orthographic=allow_orthographic)
            meshes, scales, colors = [], [], []
            for index, obj in enumerate(objects):
                poses[row, index], scale = _pose(obj, depsgraph)
                scales.append(scale)
                # Decomposition of float32 source rotations can perturb a
                # constant scale by a few ULPs. Bake the first scale consistently
                # and check subsequent scales against the explicit error budget.
                meshes.append(_mesh(obj, depsgraph, scale if initial is None else initial[1][index]))
                colors.append(_rgba(obj))
            keys = [array_digest(m) for m in meshes]
            if initial is None:
                initial = (meshes, np.asarray(scales), np.asarray(colors), camera, keys)
            else:
                if not np.allclose(scales, initial[1], atol=2e-6, rtol=0) or keys != initial[4]:
                    raise ValueError(f"geometry/scale changes at source frame {frame}")
                if not np.array_equal(colors, initial[2]):
                    raise ValueError(f"base colors change at source frame {frame}")
                if camera != initial[3]:
                    raise ValueError(f"camera changes at source frame {frame}")
        assert initial is not None  # Nonempty object and frame ranges were checked above.
        # q and -q denote the same rotation. Keep signs continuous, then detect
        # all actual world motion, including motion inherited from a parent.
        for row in range(1, len(poses)):
            flip = (poses[row, :, 3:] * poses[row - 1, :, 3:]).sum(-1) < 0
            poses[row, flip, 3:] *= -1
        moving = np.any(poses != poses[:1], axis=(0, 2))
        moving |= np.asarray([bool(o.get("pb_is_dynamic", False)) for o in objects], dtype=bool)
        nodes = [o.name for o, dynamic in zip(objects, moving, strict=True) if dynamic]
        node_indices = np.cumsum(moving) - 1
        records = [
            {
                "name": o.name,
                "id": index + 1,
                "asset": initial[4][index],
                "node": int(node_indices[index]) if moving[index] else -1,
                "pose": poses[0, index].tolist(),
                "rgba": initial[2][index].tolist(),
                "role": str(o.get("pb_role", "")),
                "source_id": str(o.get("pb_object_id", o.name)),
                "shape": str(o.get("pb_shape", "")),
            }
            for index, o in enumerate(objects)
        ]
        return {
            "poses": np.ascontiguousarray(poses[:, moving]),
            "frames": frames,
            "objects": records,
            "dynamic_nodes": nodes,
            "camera": initial[3],
            "meshes": dict(zip(initial[4], initial[0], strict=True)),
            "fps": fps,
        }
    finally:
        scene.frame_set(original)


def _license(root, text):
    content = text.encode("utf-8")
    key = hashlib.sha256(content).hexdigest()
    directory = root / "licenses"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.txt"
    with tempfile.NamedTemporaryFile(dir=directory) as stream:
        stream.write(content)
        stream.flush()
        try:
            path.hardlink_to(stream.name)
        except FileExistsError:
            pass
    if path.read_bytes() != content:
        raise ValueError("existing license content differs")
    return {"sha256": key}


def _category(objects, key):
    """Choose a readable source shape/role, without inferring it from names."""
    labels = {o.get("shape") or o.get("role") or "mesh" for o in objects if o["asset"] == key}
    label = sorted(labels)[0]
    return re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-") or "mesh"


def organize(root):
    """Move assets into semantic directories and consolidate legacy licenses.

    Run after a batch of exports, without concurrent organizers. File bytes are
    preserved, but changed manifests have new identities. No index is needed.
    """
    obsolete = set()
    for path in sorted((root / "clips").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        original = json.dumps(manifest, sort_keys=True)
        old_license = path.parent / "SOURCE_LICENSE.txt"
        if old_license.exists():
            manifest["license_ref"] = _license(root, old_license.read_text())
        for key, description in manifest["assets"].items():
            if not re.fullmatch(r"[0-9a-f]{64}", key):
                raise ValueError("invalid asset key")
            relative = description.get("file", f"assets/{key}.npz")
            if not re.fullmatch(r"assets/(?:[A-Za-z0-9_-]+/)?" + key + r"\.npz", relative):
                raise ValueError("invalid asset path")
            source = root / relative
            if sha256(source) != description["sha256"]:
                raise ValueError("asset file hash differs")
            target = root / "assets" / _category(manifest["objects"], key) / f"{key}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            if source != target:
                if not target.exists():
                    target.hardlink_to(source)
                if sha256(target) != description["sha256"]:
                    raise ValueError("existing semantic asset differs")
                obsolete.add(source)
            description["file"] = str(target.relative_to(root))
        if "license_ref" in manifest:
            key = manifest["license_ref"]["sha256"]
            if sha256(root / "licenses" / f"{key}.txt") != key:
                raise ValueError("license hash differs")
        if json.dumps(manifest, sort_keys=True) != original:
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
                json.dump(manifest, stream, indent=2, sort_keys=True)
                stream.write("\n")
            Path(stream.name).replace(path)
        if old_license.exists():
            old_license.unlink()
    # Do not remove a file still referenced by another clip, even if its semantic
    # category differs. Hard links share physical storage where supported.
    referenced: set[Path] = set()
    for path in (root / "clips").glob("*/manifest.json"):
        m = json.loads(path.read_text())
        referenced.update(root / d.get("file", f"assets/{k}.npz") for k, d in m["assets"].items())
    for path in obsolete - referenced:
        path.unlink()
    for directory in (root / "assets").iterdir():
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    (root / "index.json").unlink(missing_ok=True)
    return root


def write_recording(root, clip, recording, source):
    """Publish a new clip and reuse verified content-addressed mesh files."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", clip):
        raise ValueError("clip must be a plain identifier")
    assets, clips = root / "assets", root / "clips"
    assets.mkdir(parents=True, exist_ok=True)
    clips.mkdir(parents=True, exist_ok=True)
    destination = clips / clip
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite clip: {destination}")
    descriptions = {}
    for key, mesh in recording["meshes"].items():
        if array_digest(mesh) != key:
            raise ValueError("mesh content hash differs")
        directory = assets / _category(recording["objects"], key)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{key}.npz"
        if not path.exists():
            # Publish complete files atomically. Concurrent scene exporters may
            # share an asset, but must not expose a partially written ZIP or
            # overwrite a file whose byte hash another manifest already uses.
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".mesh-") as stream:
                np.savez_compressed(stream, **mesh)
                stream.flush()
                try:
                    path.hardlink_to(stream.name)
                except FileExistsError:
                    pass
        with np.load(path, allow_pickle=False) as archive:
            if array_digest(dict(archive)) != key:
                raise ValueError("existing asset content hash differs")
        descriptions[key] = {"sha256": sha256(path), "file": str(path.relative_to(root))}
    manifest = {
        "schema": SCHEMA_V2 if recording["camera"].get("projection_type") == "orthographic" else SCHEMA,
        "clip": clip,
        "conventions": CONVENTIONS,
        "source": {key: value for key, value in source.items() if key != "license_text"},
        "fps": recording["fps"],
        "camera": recording["camera"],
        "dynamic_nodes": recording["dynamic_nodes"],
        "objects": recording["objects"],
        "assets": descriptions,
        "arrays": {},
        "license_ref": _license(root, source["license_text"]),
    }
    if manifest["schema"] == SCHEMA_V2:
        manifest["visibility_contract"] = "stable-envs/opaque-pixel-rays-v2"
    with tempfile.TemporaryDirectory(dir=clips, prefix=".export-") as temporary:
        directory = Path(temporary)
        for name in ("poses", "frames"):
            path = directory / f"{name}.npy"
            array = recording[name]
            np.save(path, array, allow_pickle=False)
            manifest["arrays"][name] = {"dtype": str(array.dtype), "shape": list(array.shape), "sha256": sha256(path)}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        directory.rename(destination)
        # TemporaryDirectory tolerates the renamed (now absent) staging path.
    return destination / "manifest.json"


def export(root, output, task):
    import bpy

    if (output / "clips" / f"{task.lower()}-nominal").exists():
        raise FileExistsError("refusing to overwrite an existing nominal clip")
    scene, script, task_metadata = prepare(root, task)
    recording = sample_scene(
        scene, object_order=G34_ORDER if task == "G34" else None, allow_orthographic=task in ORTHOGRAPHIC_TASKS
    )
    if task in ORTHOGRAPHIC_TASKS and recording["camera"].get("projection_type") != "orthographic":
        raise ValueError("audited orthographic entry changed projection type")
    source = {
        "url": "https://github.com/hokindeng/object-permanence",
        "commit": SOURCE_COMMIT,
        "script": str(script.relative_to(root)),
        "script_sha256": sha256(script),
        "task_manifest_sha256": sha256(script.parent / "task.json"),
        "profile": "nominal-first-member-no-diversity",
        "parameters": {},
        "seed": None,
        "task": task,
        "video_split": task_metadata.get("video_split"),
        "license": "CC-BY-NC-4.0",
        "license_text": (root / "LICENSE").read_text(),
        "versions": {"blender": bpy.app.version_string, "numpy": np.__version__, "python": platform.python_version()},
        "exporter": "scripts/export/object_permanence.py",
        "exporter_sha256": sha256(Path(__file__)),
    }
    context = source_execution_context(root, task, script)
    if context is not None:
        source["profile"] = "nominal-first-member-surface-diversity-context"
        source["execution_context"] = context
    if task in ORTHOGRAPHIC_TASKS:
        source["preparation_recipe"] = {"build": AUDITED_TASKS[task][0], "animate": AUDITED_TASKS[task][1]}
    return write_recording(output, f"{task.lower()}-nominal", recording, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("inventory", "export"):
        item = sub.add_parser(command)
        item.add_argument("--source-root", type=Path, required=True)
        item.add_argument("--output", type=Path, required=True)
        if command == "export":
            item.add_argument("--task", choices=tuple(AUDITED_TASKS), required=True)
    item = sub.add_parser("organize", help="organize semantic asset folders and deduplicate licenses")
    item.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        inventory(args.source_root.resolve(), args.output)
    elif args.command == "export":
        print(export(args.source_root.resolve(), args.output, args.task))
    else:
        print(organize(args.output.resolve()))


if __name__ == "__main__":
    main()
