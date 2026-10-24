"""Optional live Blender checks; normal profiles need no Blender installation."""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from stable_envs.envs.object_permanence import G34Replay, RigidTrajectory


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    source = os.environ.get("SE_OBJECT_PERMANENCE_SOURCE")
    python = os.environ.get("SE_OBJECT_PERMANENCE_BLENDER_PYTHON")
    if not source or not python:
        pytest.skip("live export needs SE_OBJECT_PERMANENCE_SOURCE and SE_OBJECT_PERMANENCE_BLENDER_PYTHON")
    output = tmp_path_factory.mktemp("rigid-export")
    from scripts.export.object_permanence import AUDITED_TASKS, DEFERRED_TASKS

    for task in AUDITED_TASKS:
        if task in DEFERRED_TASKS:
            continue
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
    return output


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
from scripts.export.object_permanence import AUDITED_TASKS, RAY_CONTRACT_TASKS, check_source, sample_scene
root = Path(sys.argv[1])
check_source(root)
assert 'G90' not in AUDITED_TASKS and 'G90' not in RAY_CONTRACT_TASKS
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


@pytest.mark.parametrize("task", ["G91", "G92"])
def test_audited_orthographic_source_is_rejected_by_unchanged_perspective_contract(task):
    """Skip unsupported cameras without changing projection or certifying a port."""
    source = os.environ.get("SE_OBJECT_PERMANENCE_SOURCE")
    python = os.environ.get("SE_OBJECT_PERMANENCE_BLENDER_PYTHON")
    if not source or not python:
        pytest.skip("live capability audit needs pinned source and Blender interpreter")
    code = r"""
import json, runpy, sys
from pathlib import Path
import bpy
from scripts.export.object_permanence import AUDITED_TASKS, RAY_CONTRACT_TASKS, check_source, sample_scene
root, task = Path(sys.argv[1]), sys.argv[2]
check_source(root)
assert task not in AUDITED_TASKS and task not in RAY_CONTRACT_TASKS
matches = list((root/'object_permanence/generator/tasks').glob(task+'_*/task.json'))
assert len(matches) == 1
metadata = json.loads(matches[0].read_text())
script = matches[0].parent / metadata['member_scripts'][0]
source = runpy.run_path(str(script), run_name='independent_orthographic_capability_audit')
source['clear_scene']()
objects = source['build_scene_by_kind']()
source['animate_scene_by_kind'](*(objects if isinstance(objects,tuple) else (objects,)))
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
"""
    subprocess.run([python, "-c", code, source, task], check=True, timeout=180)
