# MS-Human-700

`MSHumanLocomotion` wraps the locomotion model of
[MS-Human-700](https://github.com/LNSGroup/MS-Human-700)
([Zuo et al.](https://lnsgroup.cc/research/MS-Human/)), a whole-body musculoskeletal human. The model
has 36 degrees of freedom driven by 100 muscle actuators. [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)
steps the physics on the env device. Frames come from host-side serial MuJoCo OpenGL because MuJoCo Warp's
renderer does not draw the muscle tendons, so a CUDA env copies its states to the host and its frames back.

The action has 100 values in `[0, 1]`, one activation per muscle. This is a control sandbox with no
task reward; `reset` starts every lane from the model's `init` keyframe.

The model is Apache-2.0 and is not on PyPI, so first use downloads a pinned commit into the cache. Set
`STABLE_ENVS_MSHUMAN_HOME` to an existing checkout to skip the download. Install the backend with the
`mshuman` extra; see [System setup](../system-setup.md).

```python
import stable_envs as snv

env = snv.MSHumanLocomotion(num_envs=8)
state = env.reset(seed=0)
frames = env.render()
```

## Locomotion

### MSHumanLocomotion

<!-- env: MSHumanLocomotion -->
