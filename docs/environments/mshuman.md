# MS-Human-700

`MSHumanLocomotion` wraps the locomotion model of
[MS-Human-700](https://github.com/LNSGroup/MS-Human-700)
([Zuo et al.](https://lnsgroup.cc/research/MS-Human/)), a whole-body musculoskeletal human whose 36 degrees
of freedom are driven by 100 muscle actuators. Upstream ships only this MJCF model and no task code, so the
reference dynamics are CPU MuJoCo (`float64`) stepping the pinned model. The env steps it with
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) in `float32` on the env device, which is not
bit-identical to the reference. Over the first 12 steps (24 ms) from the `init` keyframe under random
activations, with the env on CPU, muscle activations and time stay within `1e-6` of CPU MuJoCo and `qpos` and
`qvel` within `1e-2` and `2.5`. Across 150 such trajectories the worst gaps were `6.8e-3` in `qpos` and `1.74`
in `qvel`, mostly opening at foot strike. Longer rollouts are not checked against CPU MuJoCo. Frames come from
host-side serial MuJoCo OpenGL because MuJoCo Warp's renderer does not draw the muscle tendons, so a CUDA env
copies its states to the host and its frames back.

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
