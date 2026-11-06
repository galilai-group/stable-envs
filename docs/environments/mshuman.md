# MS-Human-700

`MSHumanLocomotion` wraps the locomotion model of
[MS-Human-700](https://github.com/LNSGroup/MS-Human-700)
([Zuo et al.](https://lnsgroup.cc/research/MS-Human/)), a whole-body musculoskeletal human whose 36 degrees
of freedom are driven by 100 muscle actuators. Upstream ships only this MJCF model and no task code, so the
reference dynamics are CPU MuJoCo (`float64`) stepping the pinned model. A CPU env steps every lane in one
native `mujoco.batch` call and a CUDA env steps on
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp). The CPU path runs MuJoCo in `float64` but
rounds the integration State to `float32` after every step, so it restores bit-for-bit and matches a
per-lane `mj_step` loop with the same rounding exactly. Over the first 12 steps (24 ms) from the `init`
keyframe under uniformly random activations, across 3,000 trajectories (150 for each of 20 seeds), its worst
gaps from direct CPU MuJoCo were under `1.5e-5` in `qpos`, `3e-3` in `qvel`, `1.2e-7` in activation and
`2e-9` in time. Longer rollouts are not checked against direct MuJoCo. CUDA dynamics use MuJoCo Warp in
`float32` and are not bit-identical to that reference. Frames come from host-side serial MuJoCo OpenGL because
MuJoCo Warp's renderer does not draw the muscle tendons, so a CUDA env copies its states to the host and its
frames back.

A CPU env steps on up to 8 native threads, at most one per CPU in the process affinity and per lane. As with
DM Control, build it in the process that steps it, for example in a DataLoader's `worker_init_fn`: an env
built before `fork()` hangs when the child steps it.

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
