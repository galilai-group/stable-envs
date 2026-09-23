"""Isolate upstream joint-type tuple membership in the observed dependency profile."""
import json
import numpy as np
import mujoco
j=np.int32(2)
e=mujoco.mjtJoint.mjJNT_SLIDE
print(json.dumps({'joint_type':int(j),'enum_int':int(e),'numpy_left_equality':bool(j==e),'enum_left_equality':bool(e==j),'tuple_membership':bool(j in (mujoco.mjtJoint.mjJNT_HINGE,e)),'integer_tuple_membership':bool(int(j) in (int(mujoco.mjtJoint.mjJNT_HINGE),int(e)))},indent=2))
