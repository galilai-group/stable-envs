"""Optional live Blender checks; normal profiles need no Blender installation."""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.export.object_permanence import AUDITED_TASKS, DEFERRED_TASKS
from stable_envs.envs.object_permanence import G34Replay, RigidTrajectory
from tests.helpers.reference.object_permanence_recordings import RECORDED_TASKS


@pytest.fixture(scope="module")
def export_case(tmp_path_factory):
    """Share a dataset, but charge source work to each scene's test timeout."""
    source = os.environ.get("SE_OBJECT_PERMANENCE_SOURCE")
    python = os.environ.get("SE_OBJECT_PERMANENCE_BLENDER_PYTHON")
    if not source or not python:
        pytest.skip("live export needs SE_OBJECT_PERMANENCE_SOURCE and SE_OBJECT_PERMANENCE_BLENDER_PYTHON")
    output = tmp_path_factory.mktemp("rigid-export")
    checked = set()

    def ensure(task):
        if task in checked:
            return output
        if task not in AUDITED_TASKS or task in DEFERRED_TASKS:
            raise ValueError("source case is not eligible for the live geometry gate")
        subprocess.run(
            [
                python,
                "-m",
                "scripts.export.object_permanence",
                "export",
                "--source-root",
                source,
                "--output",
                str(output),
                "--task",
                task,
            ],
            check=True,
            timeout=180,
        )
        subprocess.run(
            [
                python,
                "-m",
                "scripts.testing.check_object_permanence_export",
                "--source-root",
                source,
                "--dataset",
                str(output),
                "--task",
                task,
            ],
            check=True,
            timeout=180,
        )
        checked.add(task)
        return output

    return ensure


@pytest.fixture(scope="module")
def exported(export_case):
    """Legacy pose/ownership gates need only their declared recorded scenes."""
    for task in RECORDED_TASKS:
        output = export_case(task.upper())
    return output


@pytest.mark.parametrize("task", sorted(set(AUDITED_TASKS) - set(DEFERRED_TASKS)))
def test_audited_source_export_and_independent_world_geometry_per_scene(export_case, task):
    output = export_case(task)
    with RigidTrajectory(output, task.lower() + "-nominal") as clip:
        assert len(clip) == 120


@pytest.mark.parametrize(
    "task,member,dynamic_count,version",
    [
        ("G02", 1, 1, "v1"),
        ("G02", 2, 2, "v1"),
        ("G04", 1, 2, "v1"),
        ("G05", 2, 5, "v1"),
        ("G05", 4, 5, "v1"),
        ("G05", 5, 5, "v1"),
        ("G05", 6, 10, "v1"),
        ("G06", 1, 5, "v1"),
        ("G06", 2, 10, "v1"),
        ("G07", 1, 3, "v1"),
        ("G08", 1, 13, "v2"),
        ("G09", 1, 6, "v2"),
        ("G09", 2, 12, "v2"),
        ("G10", 1, 1, "v2"),
        ("G11", 1, 8, "v2"),
        ("G11", 2, 9, "v2"),
        ("G12", 1, 10, "v2"),
        ("G12", 2, 10, "v2"),
        ("G13", 1, 40, "v2"),
        ("G13", 2, 40, "v2"),
        ("G13", 3, 40, "v2"),
        ("G13", 4, 40, "v2"),
        ("G14", 1, 3, "v2"),
        ("G14", 2, 3, "v2"),
        ("G14", 3, 3, "v2"),
        ("G14", 4, 3, "v2"),
        ("G14", 5, 3, "v2"),
    ],
)
def test_additional_member_unified_export_and_world_check(tmp_path, task, member, dynamic_count, version):
    """Fresh source processes must reproduce each additional member's frozen poses."""
    source = os.environ.get("SE_OBJECT_PERMANENCE_SOURCE")
    python = os.environ.get("SE_OBJECT_PERMANENCE_BLENDER_PYTHON")
    if not source or not python:
        pytest.skip("live member export needs the pinned source and Blender interpreter")
    from scripts.object_permanence_member import Member
    from tests.helpers.reference.object_permanence_recordings import RECORDINGS

    subprocess.run(
        [
            python,
            "-m",
            "scripts.export.batch_object_permanence",
            "--source-root",
            source,
            "--output",
            str(tmp_path / "recordings"),
            "--report",
            str(tmp_path / "report.json"),
            "--tasks",
            task,
            "--members",
            str(member),
            "--visibility-contract",
            f"stable-envs/opaque-pixel-rays-{version}",
        ],
        check=True,
        timeout=180,
    )
    entry = Member(task, member)
    with (
        RigidTrajectory(tmp_path / "recordings", entry.clip) as fresh,
        RigidTrajectory(RECORDINGS, entry.clip) as frozen,
    ):
        assert len(fresh) == 120 and len(fresh.manifest["dynamic_nodes"]) == dynamic_count
        assert fresh.manifest["source"]["member"] == member
        assert fresh.manifest["source"]["preparation_recipe"] == frozen.manifest["source"]["preparation_recipe"]
        assert fresh.manifest["source"].get("execution_context") == frozen.manifest["source"].get("execution_context")
        assert fresh.manifest["camera"] == frozen.manifest["camera"]
        np.testing.assert_array_equal(fresh.poses(np.arange(120)), frozen.poses(np.arange(120)))
        if task in ("G11", "G12"):
            objects = {o["name"]: o for o in fresh.manifest["objects"]}
            names = ["box_top_lid", "inner_orange_ball", "inner_blue_cube"]
            if task == "G12":
                # The source labels the partition static, but its parent's
                # rotation still changes its world pose. Do not cache it static.
                names.append("inner_partition_wall")
            for name in names:
                node = objects[name]["node"]
                assert node >= 0
                rows = fresh.poses([0, 85])[:, node]
                assert not np.array_equal(rows[0], rows[1]), name

        if task == "G13":
            # Labels are tagged static upstream but inherit drawer motion.
            # Check glyph identity and the drawer/content binding, not just shape.
            objects = {o["name"]: o for o in fresh.manifest["objects"]}
            orders = {
                1: ("orange_ball", "yellow_pyramid", "blue_cube"),
                2: ("blue_cube", "orange_ball", "yellow_pyramid"),
                3: ("blue_cube", "yellow_pyramid", "orange_ball"),
                4: ("yellow_pyramid", "orange_ball", "blue_cube"),
            }
            poses = fresh.poses(np.arange(120))
            for drawer, shape in enumerate(orders[member], 1):
                label = objects[f"drawer_{drawer}_front_label"]
                content = objects[f"drawer_{drawer}_{shape}"]
                assert label["source_geometry"] == {"type": "FONT", "text": str(drawer), "font": "Bfont Regular"}
                assert label["node"] >= 0 and content["node"] >= 0
                label_xyz = poses[:, label["node"], :3]
                content_xyz = poses[:, content["node"], :3]
                assert not np.array_equal(label_xyz[0], label_xyz[60])
                offset = label_xyz - content_xyz
                np.testing.assert_allclose(offset, np.broadcast_to(offset[0], offset.shape), atol=2e-7, rtol=0)

        if task == "G14":
            # A color permutation must preserve lane identity even while hidden.
            orders = {
                1: ("red", "yellow", "blue"),
                2: ("blue", "red", "yellow"),
                3: ("blue", "yellow", "red"),
                4: ("yellow", "red", "blue"),
                5: ("yellow", "blue", "red"),
            }
            colors = {"red": (0.92, 0.18, 0.18, 1), "blue": (0.16, 0.36, 0.95, 1), "yellow": (0.98, 0.78, 0.15, 1)}
            objects = {o["name"]: o for o in fresh.manifest["objects"]}
            poses = fresh.poses(np.arange(120))
            ball_assets = set()
            for lane, y, color in zip(("near", "middle", "far"), (-1.10, 0.0, 1.10), orders[member], strict=True):
                ball = objects[f"{lane}_{color}_ball"]
                assert ball["role"] == "target_ball" and ball["node"] >= 0
                np.testing.assert_array_equal(
                    np.asarray(ball["rgba"], np.float32), np.asarray(colors[color], np.float32)
                )
                xyz = poses[:, ball["node"], :3]
                np.testing.assert_array_equal(xyz[:, 1], np.full(120, y, np.float32))
                assert not np.array_equal(xyz[0], xyz[-1])
                tube = objects[f"{lane}_single_opaque_left_u_tube"]
                assert tube["source_geometry"]["type"] == "CURVE" and tube["node"] == -1
                ball_assets.add(ball["asset"])
            assert len(ball_assets) == 1  # Instance colors do not duplicate meshes.


@pytest.mark.parametrize("member", [1, 3])
def test_g05_member_deferrals_preserve_hidden_marker_and_scale_evidence(member):
    """Audit the exact source; never delete hidden markers or drop animated scale."""
    source = os.environ.get("SE_OBJECT_PERMANENCE_SOURCE")
    python = os.environ.get("SE_OBJECT_PERMANENCE_BLENDER_PYTHON")
    if not source or not python:
        pytest.skip("live member capability audit needs pinned source and Blender interpreter")
    code = r"""
import json, runpy, sys
from pathlib import Path
import bpy
from scripts.export.object_permanence import AUDITED_MEMBERS, check_source, sample_scene
root, member = Path(sys.argv[1]), int(sys.argv[2])
check_source(root)
assert ('G05', member) not in AUDITED_MEMBERS
task, = (root/'object_permanence/generator/tasks').glob('G05_*/task.json')
script = task.parent/json.loads(task.read_text())['member_scripts'][member]
source = runpy.run_path(str(script), run_name='independent_g05_member_capability_audit')
source['clear_scene']()
if member == 1:
    objs = source['build_scene']()
    source['animate_scene'](**{k: objs[k] for k in ('scene','track','wall_parts','car_body','car_parts')})
    marker = bpy.context.scene.objects['non_solid_car_passage_hole_volume_marker']
    assert marker.hide_render and marker.hide_viewport and not marker['pb_solid']
    reason = 'hidden objects/visibility changes'
else:
    # Audited sequence from this member's main(), stopping before all source I/O.
    scene = source['build_base_scene']()
    static = source['build_track']()
    body, parts = source['build_car']()
    near, far, seam, doors = source['build_double_door']()
    static.extend(doors)
    source['setup_camera'](scene)
    source['animate_scene'](scene, body, parts, near, far, seam, static)
    reason = 'geometry/scale changes at source frame 21'
try:
    sample_scene(bpy.context.scene)
except ValueError as error:
    assert reason in str(error), str(error)
else:
    raise AssertionError('deferred member bypassed the existing rigid capture contract')
if member == 3:
    # The existing positive-scale format is sufficient; the unaudited multi-step
    # preparation and per-member opt-in still need a separate import batch.
    recording = sample_scene(bpy.context.scene, allow_animated_scale=True)
    assert len(recording['frames']) == 120 and 'scales' in recording
    assert recording['scales'].min() > 0 and recording['scales'].min() < 0.1
"""
    subprocess.run([python, "-c", code, source, str(member)], check=True, timeout=180)


def test_g34_export_matches_unchanged_independent_source_fixture_and_legacy_replay(exported):
    with RigidTrajectory(exported, "g34-nominal") as trajectory:
        rows = trajectory.poses(np.arange(120))[:, 0]
        assert trajectory.manifest["dynamic_nodes"] == ["small_orange_ball"]
        path = Path(__file__).parents[1] / "fixtures/semantics/occlusion_ball/blender_g34_nominal.npz"
        with np.load(path, allow_pickle=False) as oracle:
            np.testing.assert_allclose(rows[:, :3], oracle["positions"], atol=3e-7, rtol=0)
            q = oracle["quaternions"]
            signs = np.where((q * rows[:, 3:]).sum(-1, keepdims=True) < 0, -1, 1)
            np.testing.assert_allclose(rows[:, 3:], q * signs, atol=1e-6, rtol=0)
        with G34Replay(rows) as replay:
            np.testing.assert_array_equal(replay.poses(replay.advance(44)), rows[44:45])


def test_g62_exports_both_movers_and_shared_mesh_with_distinct_colors(exported):
    with RigidTrajectory(exported, "g62-nominal") as trajectory:
        assert len(trajectory) == 120
        manifest = trajectory.manifest
        assert set(manifest["dynamic_nodes"]) == {"cross_ball_blue", "cross_ball_red"}
        objects = {obj["name"]: obj for obj in manifest["objects"]}
        red, blue = objects["cross_ball_red"], objects["cross_ball_blue"]
        assert red["id"] != blue["id"] and red["node"] != blue["node"]
        assert red["asset"] == blue["asset"] and red["rgba"] != blue["rgba"]
        poses = trajectory.poses([0, 59, 119])
        assert poses[0, red["node"], 0] < poses[-1, red["node"], 0]
        assert poses[0, blue["node"], 0] > poses[-1, blue["node"], 0]
        assert np.isfinite(poses[1]).all()  # hidden objects retain poses
        assert len(manifest["assets"]) < len(manifest["objects"])


def test_independent_checker_detects_shifted_hidden_pose(exported, tmp_path):
    destination = tmp_path / "shifted"
    shutil.copytree(exported, destination)
    path = destination / "clips/g62-nominal/poses.npy"
    poses = np.load(path)
    poses[59, 0, 0] += 0.1
    np.save(path, poses)
    result = subprocess.run(
        [
            os.environ["SE_OBJECT_PERMANENCE_BLENDER_PYTHON"],
            "-m",
            "scripts.testing.check_object_permanence_export",
            "--source-root",
            os.environ["SE_OBJECT_PERMANENCE_SOURCE"],
            "--dataset",
            str(destination),
            "--task",
            "G62",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode != 0 and "AssertionError" in result.stderr


def test_live_export_handles_parent_motion_and_rejects_unsupported_animation():
    python = os.environ.get("SE_OBJECT_PERMANENCE_BLENDER_PYTHON")
    if not python:
        pytest.skip("live extraction checks need SE_OBJECT_PERMANENCE_BLENDER_PYTHON")
    code = """
import bpy
import numpy as np
from scripts.export.object_permanence import sample_scene

def setup():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 2
    scene.frame_set(1)
    bpy.ops.object.camera_add(location=(0, -8, 3))
    scene.camera = bpy.context.object
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.object
    material = bpy.data.materials.new("opaque")
    material.diffuse_color = (0.5, 0.5, 0.5, 1)
    cube.data.materials.append(material)
    return scene, cube, material

def animate(obj, path, first, second):
    setattr(obj, path, first)
    obj.keyframe_insert(data_path=path, frame=1)
    setattr(obj, path, second)
    obj.keyframe_insert(data_path=path, frame=2)

scene, cube, _ = setup()
bpy.ops.object.empty_add()
parent = bpy.context.object
cube.parent = parent
animate(parent, "location", (0,0,0), (1,0,0))
scene.frame_set(2)
recording = sample_scene(scene)
assert scene.frame_current == 2
assert recording["poses"].shape == (2,1,7)
np.testing.assert_array_equal(recording["poses"][:,0,0], [0,1])

scene, cube, _ = setup()
cube.scale = (2,3,4)
recording = sample_scene(scene)
mesh = next(iter(recording["meshes"].values()))
np.testing.assert_array_equal(abs(mesh["vertices"]).max(axis=0), [2,3,4])

for kind in ("scale", "camera", "color", "visibility", "collection", "deformation", "reflection", "transparent"):
    scene, cube, material = setup()
    if kind == "scale":
        animate(cube, "scale", (1,1,1), (2,1,1))
    elif kind == "camera":
        animate(scene.camera, "location", (0,-8,3), (1,-8,3))
    elif kind == "color":
        animate(material, "diffuse_color", (1,0,0,1), (0,1,0,1))
    elif kind == "visibility":
        animate(cube, "hide_render", False, True)
    elif kind == "collection":
        cube.users_collection[0].hide_render = True
    elif kind == "deformation":
        cube.shape_key_add(name="Basis")
        key = cube.shape_key_add(name="deform")
        key.data[0].co.x += 0.5
        animate(key, "value", 0, 1)
    elif kind == "reflection":
        cube.scale = (-1,1,1)
    else:
        material.diffuse_color = (1,1,1,0.5)
    try:
        sample_scene(scene)
    except ValueError:
        pass
    else:
        raise AssertionError("silently accepted unsupported " + kind)
    if kind == "collection":
        cube.users_collection[0].hide_render = False
"""
    subprocess.run([python, "-c", code], check=True, timeout=60)


@pytest.mark.parametrize("task", ["g27", "g37", "g38", "g55", "g64"])
def test_expansion_live_export_matches_independent_frozen_world_poses(exported, task):
    from tests.helpers.reference.object_permanence_recordings import load_geometry

    reference, oracle = load_geometry(task)
    with RigidTrajectory(exported, f"{task}-nominal") as clip:
        np.testing.assert_array_equal(clip.frames, oracle["source_frames"])
        actual = clip.object_poses(np.arange(len(clip)))
        assert [o["name"] for o in clip.manifest["objects"]] == reference["object_order"]
        np.testing.assert_allclose(actual[..., :3], oracle["poses"][..., :3], atol=3e-7, rtol=0)
        q = oracle["poses"][..., 3:]
        signs = np.where((actual[..., 3:] * q).sum(-1, keepdims=True) < 0, -1, 1)
        np.testing.assert_allclose(actual[..., 3:], q * signs, atol=1e-6, rtol=0)


def test_next_source_task_with_active_glass_is_rejected_by_unchanged_opaque_contract():
    """G90 is an audited stopping boundary, not a silently accepted opaque port."""
    source = os.environ.get("SE_OBJECT_PERMANENCE_SOURCE")
    python = os.environ.get("SE_OBJECT_PERMANENCE_BLENDER_PYTHON")
    if not source or not python:
        pytest.skip("live capability audit needs pinned source and Blender interpreter")
    code = r"""
import json, runpy, sys
from pathlib import Path
import bpy
from scripts.export.object_permanence import ALPHA_TASKS, RAY_CONTRACT_TASKS, check_source, sample_scene
root = Path(sys.argv[1])
check_source(root)
assert 'G90' in ALPHA_TASKS and 'G90' not in RAY_CONTRACT_TASKS
matches = list((root/'object_permanence/generator/tasks').glob('G90_*/task.json'))
assert len(matches) == 1
metadata = json.loads(matches[0].read_text())
script = matches[0].parent / metadata['member_scripts'][0]
source = runpy.run_path(str(script), run_name='independent_g90_capability_audit')
source['clear_scene']()
objects = source['build_scene_by_kind']()
source['animate_scene'](objects)
scene = bpy.context.scene
scene.frame_set(scene.frame_start)
bpy.context.view_layer.update()
transparent = {}
for obj in scene.objects:
    if obj.type != 'MESH':
        continue
    bsdf = obj.active_material.node_tree.nodes.get('Principled BSDF')
    alpha = float(bsdf.inputs['Alpha'].default_value)
    if alpha != 1:
        transparent[obj.name] = alpha
assert set(transparent) == {'small_funnel_endcap_F','small_funnel_endcap_B','large_funnel_endcap_F','large_funnel_endcap_B'}
assert all(0 < alpha < 1 for alpha in transparent.values())
try:
    sample_scene(scene)
except ValueError as error:
    assert 'unsupported transparency/material links' in str(error), str(error)
else:
    raise AssertionError('active glass was accepted by the opaque-only contract')
"""
    subprocess.run([python, "-c", code, source], check=True, timeout=180)


@pytest.mark.parametrize(
    "task,build,animate",
    [
        ("G09", "build_scene_by_kind", "animate_scene_by_kind"),
        ("G10", "build_scene", "animate_scene"),
        ("G91", "build_scene_by_kind", "animate_scene_by_kind"),
        ("G92", "build_scene_by_kind", "animate_scene_by_kind"),
    ],
)
def test_audited_orthographic_source_is_rejected_by_unchanged_perspective_contract(task, build, animate):
    """V1 keeps rejecting ORTHO; explicit V2 extraction preserves its projection."""
    source = os.environ.get("SE_OBJECT_PERMANENCE_SOURCE")
    python = os.environ.get("SE_OBJECT_PERMANENCE_BLENDER_PYTHON")
    if not source or not python:
        pytest.skip("live capability audit needs pinned source and Blender interpreter")
    code = r"""
import json, runpy, sys
from pathlib import Path
import bpy
from scripts.export.object_permanence import ORTHOGRAPHIC_TASKS, RAY_CONTRACT_TASKS, check_source, sample_scene
root, task, build, animate = Path(sys.argv[1]), *sys.argv[2:]
check_source(root)
assert task in ORTHOGRAPHIC_TASKS and task not in RAY_CONTRACT_TASKS
matches = list((root/'object_permanence/generator/tasks').glob(task+'_*/task.json'))
assert len(matches) == 1
metadata = json.loads(matches[0].read_text())
script = matches[0].parent / metadata['member_scripts'][0]
source = runpy.run_path(str(script), run_name='independent_orthographic_capability_audit')
source['clear_scene']()
objects = source[build]()
source[animate](*(objects if isinstance(objects,tuple) else (objects,)))
scene = bpy.context.scene
scene.frame_set(scene.frame_start)
bpy.context.view_layer.update()
assert scene.camera.data.type == 'ORTHO'
try:
    sample_scene(scene)
except ValueError as error:
    assert 'requires a perspective camera without depth of field' in str(error), str(error)
else:
    raise AssertionError('orthographic camera was accepted by the perspective-only contract')
recording = sample_scene(scene, allow_orthographic=True)
assert recording['camera']['projection_type'] == 'orthographic'
assert recording['camera']['projection'][3] == [0,0,0,1]
assert len(recording['frames']) == 120
"""
    subprocess.run([python, "-c", code, source, task, build, animate], check=True, timeout=180)


@pytest.mark.parametrize("task", ["G01", "G04"])
def test_default_export_keeps_transparency_and_evaluated_geometry_opt_in(task):
    source = os.environ.get("SE_OBJECT_PERMANENCE_SOURCE")
    python = os.environ.get("SE_OBJECT_PERMANENCE_BLENDER_PYTHON")
    if not source or not python:
        pytest.skip("live capability audit needs pinned source and Blender interpreter")
    code = r"""
import inspect, json, runpy, sys
from pathlib import Path
import bpy
from scripts.export.object_permanence import AUDITED_TASKS, CONVERTED_GEOMETRY_TASKS, check_source, sample_scene, source_prefix
root, task = Path(sys.argv[1]), sys.argv[2]
check_source(root)
assert (task not in AUDITED_TASKS) if task == 'G01' else (task in CONVERTED_GEOMETRY_TASKS)
path, = (root/'object_permanence/generator/tasks').glob(task+'_*/task.json')
script = path.parent/json.loads(path.read_text())['member_scripts'][0]
if task == 'G01':
    source_prefix(script, 'render_png')
else:
    source = runpy.run_path(str(script), run_name='independent_early_capability_audit')
    source['clear_scene']()
    objects = source['build_scene']()
    assert any(o.type == 'CURVE' for o in bpy.context.scene.objects)
    source['animate_scene'](**{n: objects[n] for n in inspect.signature(source['animate_scene']).parameters})
try:
    sample_scene(bpy.context.scene)
except ValueError as error:
    expected = 'unsupported transparency/material links' if task == 'G01' else 'unsupported renderable object type'
    assert expected in str(error), str(error)
else:
    raise AssertionError('unsupported source was accepted')
check_source(root)
"""
    subprocess.run([python, "-c", code, source, task], check=True, timeout=180)


@pytest.mark.parametrize("task", ["G08", "G44", "G45", "G46", "G48"])
def test_official_surface_diversity_context_runs_all_frames_and_retains_capability_rejections(task):
    source = os.environ.get("SE_OBJECT_PERMANENCE_SOURCE")
    python = os.environ.get("SE_OBJECT_PERMANENCE_BLENDER_PYTHON")
    if not source or not python:
        pytest.skip("source context audit needs pinned source and Blender interpreter")
    code = r"""
import json, runpy, sys
from pathlib import Path
import bpy
import numpy as np
from scripts.export.object_permanence import AUDITED_TASKS, check_source, sample_scene, source_execution_context
from scripts.testing.freeze_object_permanence import reference_execution_context
root, task = Path(sys.argv[1]), sys.argv[2]
check_source(root)
path, = (root/'object_permanence/generator/tasks').glob(task+'_*/task.json')
script = path.parent/json.loads(path.read_text())['member_scripts'][0]
context = source_execution_context(root, task, script)
assert context == reference_execution_context(script, task)
assert context['globals'] == {'DIVERSITY': {}} and context['source_bindings'] == {}
official = runpy.run_path(str(root/'object_permanence/generator/core/diversity.py'))
settings = official['load_script_diversity'](str(script), 0, 'surface')
assert context['globals']['DIVERSITY'] == settings['values']
assert official['apply_source_bindings'](script.read_text(), settings) == (script.read_text(), {})
recipes = {
    'G08': ('build_scene', 'animate_scene'),
    'G44': ('build_scene_by_kind', 'animate_scene'),
    'G45': ('build_scene_by_kind', 'animate_scene_by_kind'),
    'G46': ('build_scene', 'animate_scene'),
    'G48': ('build_scene', 'animate_scene'),
}
source = runpy.run_path(str(script), run_name='source_context_integration', init_globals=context['globals'])
source['clear_scene']()
build, animate = recipes[task]
objects = source[build]()
source[animate](objects)
scene = bpy.context.scene
assert (scene.frame_start, scene.frame_end) == (1, 120)
for frame in range(1, 121):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    assert all(np.isfinite(np.asarray(obj.matrix_world)).all() for obj in scene.objects)
    assert scene.camera.data.type == ('PERSP' if task == 'G44' else 'ORTHO')
if task == 'G44':
    assert task in AUDITED_TASKS
    recording = sample_scene(scene)
    assert recording['poses'].shape == (120, 2, 7)
    assert set(recording['dynamic_nodes']) == {'blue_ball_right', 'red_ball_left'}
else:
    if task == 'G46':
        assert task in AUDITED_TASKS
        assert any(obj.type == 'FONT' for obj in scene.objects)
    else:
        assert task in AUDITED_TASKS
    try:
        sample_scene(scene)
    except ValueError as error:
        expected = 'unsupported renderable object type' if task == 'G46' else 'requires a perspective camera'
        assert expected in str(error), str(error)
    else:
        raise AssertionError('source context must not waive unsupported scene capabilities')
check_source(root)
"""
    subprocess.run([python, "-c", code, source, task], check=True, timeout=180)


@pytest.mark.parametrize("task,member", [("G01", 0), ("G01", 1), ("G03", 1)])
@pytest.mark.timeout(900)
def test_live_optical_export_and_independent_world_geometry(tmp_path, task, member):
    source = os.environ.get("SE_OBJECT_PERMANENCE_SOURCE")
    python = os.environ.get("SE_OBJECT_PERMANENCE_BLENDER_PYTHON")
    if not source or not python:
        pytest.skip("live optical export needs pinned source and Blender interpreter")
    command = [
        python,
        "-B",
        "-m",
        "scripts.export.object_permanence_optical",
        "--source-root",
        source,
        "--output",
        str(tmp_path),
        "--task",
        task,
        "--member",
        str(member),
    ]
    subprocess.run(command, check=True, timeout=180)
    subprocess.run(
        [
            python,
            "-B",
            "-m",
            "scripts.testing.check_object_permanence_export",
            "--source-root",
            source,
            "--dataset",
            str(tmp_path),
            "--task",
            task,
            "--member",
            str(member),
        ],
        check=True,
        timeout=180,
    )
    clip = task.lower() + ("-nominal" if member == 0 else "-member1")
    with RigidTrajectory(tmp_path, clip) as recording:
        assert recording.transforms([0, 44, 119]).shape == (3, 1, 4, 4)
        assert recording.poses([0, 44, 119]).shape == (3, 1, 7)
        assert recording.manifest["schema"] == "stable-envs/object-permanence-rigid-v5"


@pytest.mark.parametrize("task", ["G98", "G132"])
def test_native_realization_reexports_identical_arrays_in_fresh_processes(tmp_path, task):
    from tests.helpers.reference.object_permanence_recordings import RECORDINGS

    source = os.environ.get("SE_OBJECT_PERMANENCE_SOURCE")
    python = os.environ.get("SE_OBJECT_PERMANENCE_BLENDER_PYTHON")
    if not source or not python:
        pytest.skip("native realization export needs pinned source and Blender interpreter")
    with RigidTrajectory(RECORDINGS, f"{task.lower()}-nominal") as frozen:
        for repeat in range(2):
            output = tmp_path / str(repeat)
            subprocess.run(
                [
                    python,
                    "-B",
                    "-m",
                    "scripts.export.object_permanence",
                    "export",
                    "--source-root",
                    source,
                    "--output",
                    str(output),
                    "--task",
                    task,
                ],
                check=True,
                timeout=180,
            )
            subprocess.run(
                [
                    python,
                    "-B",
                    "-m",
                    "scripts.testing.check_object_permanence_export",
                    "--source-root",
                    source,
                    "--dataset",
                    str(output),
                    "--task",
                    task,
                ],
                check=True,
                timeout=180,
            )
            with RigidTrajectory(output, f"{task.lower()}-nominal") as fresh:
                np.testing.assert_array_equal(fresh.poses(np.arange(120)), frozen.poses(np.arange(120)))
                assert fresh.manifest["camera"] == frozen.manifest["camera"]
                assert fresh.manifest["source"]["realization"] == frozen.manifest["source"]["realization"]
                for key, value in fresh.triangles().items():
                    np.testing.assert_array_equal(value, frozen.triangles()[key])
