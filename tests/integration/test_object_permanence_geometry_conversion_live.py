"""Real Blender curves/text retain source tessellation, identity and parent motion."""

import os
import subprocess

import pytest


def test_live_evaluated_curve_and_font_conversion_and_rigid_contract_guards():
    python = os.environ.get("SE_OBJECT_PERMANENCE_BLENDER_PYTHON")
    if not python:
        pytest.skip("live conversion needs the pinned Blender interpreter")
    code = r"""
import bpy, numpy as np
from scripts.export.object_permanence import sample_scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene=bpy.context.scene
scene.frame_start,scene.frame_end=1,2
bpy.ops.object.camera_add(location=(0,0,8))
scene.camera=bpy.context.object
material=bpy.data.materials.new('opaque')
material.diffuse_color=(0.2,0.4,0.6,1)
data=bpy.data.curves.new('source_curve','CURVE')
data.dimensions='3D';data.bevel_depth=0.1;data.bevel_resolution=3;data.use_fill_caps=True
spline=data.splines.new('POLY');spline.points.add(2)
for p,co in zip(spline.points,[(0,0,0,1),(1,0,0,1),(1,1,0,1)]):p.co=co
curve=bpy.data.objects.new('rail',data);scene.collection.objects.link(curve);data.materials.append(material)
bpy.ops.object.text_add()
label=bpy.context.object;label.name='identity_label';label.data.body='OB8';label.data.extrude=0.02;label.data.materials.append(material)
bpy.ops.object.empty_add()
parent=bpy.context.object;label.parent=parent
parent.location=(0,0,0);parent.keyframe_insert(data_path='location',frame=1)
parent.location=(1,0,0);parent.keyframe_insert(data_path='location',frame=2)
scene.frame_set(2)
try:sample_scene(scene)
except ValueError as e:assert 'unsupported renderable object type' in str(e)
else:raise AssertionError('default mesh-only interface changed')
recording=sample_scene(scene,allow_evaluated_geometry=True)
assert scene.frame_current==2 and curve.type=='CURVE' and label.type=='FONT'
assert recording['poses'].shape==(2,1,7)
np.testing.assert_array_equal(recording['poses'][:,0,0],[0,1])
objects={o['name']:o for o in recording['objects']}
assert objects['identity_label']['source_geometry']['text']=='OB8'
assert objects['identity_label']['node']==0 and objects['rail']['node']==-1
assert recording['geometry_conversion']['id']=='stable-envs/evaluated-rigid-mesh-v1'
assert all(len(m['triangles'])>0 for m in recording['meshes'].values())
# Font holes/bevels/caps remain actual source triangles, not a bounding primitive.
for obj in (curve,label):
 graph=bpy.context.evaluated_depsgraph_get();original=obj.evaluated_get(graph)
 independent=bpy.data.meshes.new_from_object(original,preserve_all_data_layers=True,depsgraph=graph)
 try:
  independent.calc_loop_triangles()
  assert len(recording['meshes'][objects[obj.name]['asset']]['triangles'])==len(independent.loop_triangles)
 finally:bpy.data.meshes.remove(independent)
# Changing an invisible trailing space preserves glyph triangles but must not
# silently change the recorded source text identity.
def text_change(scene):label.data.body='OB8' if scene.frame_current==1 else 'OB8 '
bpy.app.handlers.frame_change_pre.append(text_change)
try:
 try:sample_scene(scene,allow_evaluated_geometry=True)
 except ValueError as e:assert 'source geometry identity changes' in str(e),str(e)
 else:raise AssertionError('animated text was accepted')
finally:bpy.app.handlers.frame_change_pre.remove(text_change)
label.data.body='OB8'
data.bevel_depth=0.1;data.keyframe_insert(data_path='bevel_depth',frame=1)
data.bevel_depth=0.2;data.keyframe_insert(data_path='bevel_depth',frame=2)
try:sample_scene(scene,allow_evaluated_geometry=True)
except ValueError as e:assert 'geometry/scale changes' in str(e),str(e)
else:raise AssertionError('deforming curve was accepted by rigid replay')
assert scene.frame_current==2
data.animation_data_clear();data.bevel_depth=0.1
collection=bpy.data.collections.new('actual_instance_collection')
scene.collection.children.link(collection);collection.objects.link(curve)
instance=bpy.data.objects.new('actual_collection_instance',None)
scene.collection.objects.link(instance);instance.instance_type='COLLECTION';instance.instance_collection=collection
try:sample_scene(scene,allow_evaluated_geometry=True)
except ValueError as e:assert 'generated instances' in str(e),str(e)
else:raise AssertionError('Curve/FONT conversion waived actual instancing')
"""
    subprocess.run([python, "-B", "-c", code], check=True, timeout=180)
