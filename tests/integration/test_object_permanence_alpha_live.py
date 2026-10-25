"""Constant alpha is opt-in; optical graphs and changing materials stay rejected."""

import os
import subprocess

import pytest


def test_live_alpha_graph_animation_and_transmission_guards():
    python = os.environ.get("SE_OBJECT_PERMANENCE_BLENDER_PYTHON")
    if not python:
        pytest.skip("needs pinned Blender interpreter")
    code = r"""
import bpy
from scripts.export.object_permanence import sample_scene
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
scene=bpy.context.scene;scene.frame_start,scene.frame_end=1,2
bpy.ops.object.camera_add(location=(0,0,8));scene.camera=bpy.context.object
bpy.ops.mesh.primitive_cube_add();obj=bpy.context.object
material=bpy.data.materials.new('source_alpha');material.use_nodes=True
shader=material.node_tree.nodes['Principled BSDF'];shader.inputs['Alpha'].default_value=.25
obj.data.materials.append(material)
try:sample_scene(scene)
except ValueError as error:assert 'transparency' in str(error)
else:raise AssertionError('opaque export silently accepted alpha')
r=sample_scene(scene,allow_transparent=True)
assert r['material_convention']=='stable-envs/constant-alpha-surfaces-v1'
assert r['objects'][0]['rgba'][3]==.25
shader.inputs['Alpha'].keyframe_insert(data_path='default_value',frame=1)
shader.inputs['Alpha'].default_value=.5;shader.inputs['Alpha'].keyframe_insert(data_path='default_value',frame=2)
try:sample_scene(scene,allow_transparent=True)
except ValueError as error:assert 'base colors change' in str(error)
else:raise AssertionError('animated alpha accepted')
material.node_tree.animation_data_clear();shader.inputs['Alpha'].default_value=.25
shader.inputs['Transmission Weight'].default_value=.25
try:sample_scene(scene,allow_transparent=True)
except ValueError as error:assert 'transparency/material' in str(error)
else:raise AssertionError('refracting glass replaced with alpha')
shader.inputs['Transmission Weight'].default_value=0
value=material.node_tree.nodes.new('ShaderNodeValue')
material.node_tree.links.new(value.outputs[0],shader.inputs['Alpha'])
try:sample_scene(scene,allow_transparent=True)
except ValueError as error:assert 'alpha shader graph' in str(error)
else:raise AssertionError('linked alpha accepted')
"""
    subprocess.run([python, "-B", "-c", code], check=True, timeout=180)
