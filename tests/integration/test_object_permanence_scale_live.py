"""Scale-only source animation is recorded without weakening rigid mesh guards."""

import os
import subprocess

import pytest


def test_positive_scale_sampling_origin_and_rejection_of_unsupported_deformation():
    python = os.environ.get("SE_OBJECT_PERMANENCE_BLENDER_PYTHON")
    if not python:
        pytest.skip("live scale sampling needs the pinned Blender interpreter")
    code = r"""
import bpy, numpy as np
from scripts.export.object_permanence import sample_scene
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
scene=bpy.context.scene;scene.frame_start,scene.frame_end=1,3
bpy.ops.object.camera_add(location=(0,0,8));scene.camera=bpy.context.object
material=bpy.data.materials.new('opaque');material.diffuse_color=(.2,.4,.6,1)
bpy.ops.mesh.primitive_cube_add(location=(1,2,0))
obj=bpy.context.object;obj.name='scale_only';obj.data.materials.append(material)
# Geometry deliberately offset from its origin; no recentering is allowed.
for vertex in obj.data.vertices:vertex.co.z-=1
obj.scale=(2,3,4);obj.keyframe_insert(data_path='scale',frame=1)
obj.scale=(4,1.5,4e-4);obj.keyframe_insert(data_path='scale',frame=3)
scene.frame_set(2)
try:sample_scene(scene)
except ValueError as e:assert 'geometry/scale changes' in str(e),str(e)
else:raise AssertionError('default scale guard changed')
recording=sample_scene(scene,allow_animated_scale=True)
assert scene.frame_current==2 and recording['dynamic_nodes']==['scale_only']
assert recording['poses'].shape==(3,1,7) and recording['scales'].dtype==np.float32
np.testing.assert_array_equal(recording['poses'][:,0,:3],[[1,2,0]]*3)
np.testing.assert_array_equal(recording['scales'][0],[[1,1,1]])
np.testing.assert_allclose(recording['scales'][-1],[[2,.5,1e-4]],rtol=1e-6)
mesh=next(iter(recording['meshes'].values()))
assert mesh['vertices'][:,2].min()==-8 and mesh['vertices'][:,2].max()==0
for bad in [(2,3,0),(2,3,-1)]:
 obj.scale=bad;obj.keyframe_insert(data_path='scale',frame=3)
 try:sample_scene(scene,allow_animated_scale=True)
 except ValueError as e:assert 'shear/reflection' in str(e),str(e)
 else:raise AssertionError('zero/reflected scale accepted')
obj.animation_data_clear();obj.scale=(2,3,4)
obj.data.vertices[0].co.x=-2
obj.shape_key_add(name='Basis');deform=obj.shape_key_add(name='Deformed')
deform.data[0].co.x-=1
deform.value=0;deform.keyframe_insert(data_path='value',frame=1)
deform.value=1;deform.keyframe_insert(data_path='value',frame=3)
try:sample_scene(scene,allow_animated_scale=True)
except ValueError as e:assert 'geometry/scale changes' in str(e),str(e)
else:raise AssertionError('non-affine mesh deformation accepted')
obj.shape_key_clear()
constant=sample_scene(scene,allow_animated_scale=True)
assert 'scales' not in constant and constant['poses'].shape==(3,0,7)
bpy.ops.object.empty_add()
parent=bpy.context.object;parent.scale=(2,1,1)
obj.parent=parent;obj.rotation_euler.z=.37
try:sample_scene(scene,allow_animated_scale=True)
except ValueError as e:assert 'shear/reflection' in str(e),str(e)
else:raise AssertionError('inherited shear accepted')
"""
    subprocess.run([python, "-B", "-c", code], check=True, timeout=180)
