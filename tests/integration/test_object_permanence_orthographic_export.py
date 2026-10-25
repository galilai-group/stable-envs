"""The V2 camera extension cannot silently accept transparent source geometry."""

import os
import subprocess

import pytest


@pytest.mark.parametrize("task,mesh", [("G54", "guardwall_a_back"), ("G61", "spiral_guard_00")])
def test_orthographic_source_glass_remains_rejected_by_the_opaque_contract(task, mesh):
    source = os.environ.get("SE_OBJECT_PERMANENCE_SOURCE")
    python = os.environ.get("SE_OBJECT_PERMANENCE_BLENDER_PYTHON")
    if not source or not python:
        pytest.skip("live capability audit needs pinned source and Blender interpreter")
    code = r"""
import json, sys
from pathlib import Path
import bpy
from scripts.export.object_permanence import ALPHA_TASKS, check_source, sample_scene
from scripts.testing.freeze_object_permanence import REFERENCE_CASES, prepare_reference
root, task, mesh = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
check_source(root)
assert task in ALPHA_TASKS
path, = (root / 'object_permanence/generator/tasks').glob(task+'_*/task.json')
script = path.parent / json.loads(path.read_text())['member_scripts'][0]
REFERENCE_CASES[task] = ('build_scene_by_kind', 'animate_scene_by_kind', ())
prepare_reference(script, task)
scene = bpy.context.scene
assert scene.camera.data.type == 'ORTHO' and scene.objects[mesh].type == 'MESH'
try:
    sample_scene(scene, allow_orthographic=True)
except ValueError as error:
    assert 'unsupported transparency/material links: '+mesh in str(error), str(error)
else:
    raise AssertionError('orthographic support waived the opaque material contract')
check_source(root)
"""
    subprocess.run([python, "-B", "-c", code, source, task, mesh], check=True, timeout=180)
