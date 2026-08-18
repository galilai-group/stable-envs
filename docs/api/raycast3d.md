# `stable_envs.render.raycast3d`

The third rendering path: a batched analytic ray-cast for 3-D scenes built from MuJoCo's primitive geoms,
used by the OGBench mazes and the DM Control cells. Every primitive is a closed-form intersection, so the
caster is exact rather than approximate — there is no tessellation to match — and its geom-id map agrees
with MuJoCo's own segmentation buffer on 100.000% of interior pixels.

Two conventions in it were recovered by measurement and are invisible by eye: the camera is the *average*
of MuJoCo's two stereo eyes, and MuJoCo's depth buffer holds perpendicular distance rather than radial.

::: stable_envs.render.raycast3d
