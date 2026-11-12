# PyFlyt

`PyFlytRocketLanding` ports the rocket landing task of [PyFlyt](https://github.com/jjshoots/PyFlyt)
([Tai et al., 2023](https://arxiv.org/abs/2304.01305)). A rocket falls from a height with a random tilt
and offset. The agent uses the engine and the fins to land it upright on a pad. PyFlyt and PyBullet
step the physics on CPU.

The action has seven values:

| Index | Control | Bounds |
|---|---|---|
| 0, 1, 2 | finlet deflection in x and y, and finlet roll | [-1, 1] |
| 3 | engine ignition | [0, 1] |
| 4 | engine throttle | [0, 1] |
| 5, 6 | engine gimbal in x and y | [-1, 1] |

**Success:** the source sets `env_complete` when the rocket lands on the pad at a low speed. The step
output reports it as `success`.

The class needs the `rocket` extra, which pins `PyFlyt==0.29.0` and `pybullet==3.2.7`. On CPU,
TinyRenderer draws the frames. On CUDA, nvdiffrast draws them. nvdiffrast is not on PyPI. See
[Install](../install.md). TinyRenderer is the source's renderer. The nvdiffrast frames differ from the
TinyRenderer frames. `env.describe()["visual_fidelity"]` gives the measured agreement.

On CPU with worker processes, the workers draw the step frames of their running lanes with
TinyRenderer. A step then does not export a State only to draw it. On Linux, `render` draws a batch of
2 or more States in render worker processes, for every number of lanes. The main process checks the
State while the workers draw. `render_workers` sets the number of workers and `render_worker_min_rows`
sets the smallest such batch.
These frames are byte-identical to frames drawn in the main process.

A step reads from PyBullet only the State leaves that a step can change: the rocket pose, velocity and fuel
tank mass, the cached link poses, and the task, RNG and history fields. The other leaves come from the last
reset, configuration or restore. A zero leaf that a step does not change, such as `current/wrapper/current_goal`
without a render mode, is a new zero tensor and is not copied. The State is the same as a full read of the
engine. On CUDA, the outputs of the worker processes go to the device when they arrive.

A reset makes new PyBullet clients before it closes the old ones, so that a failed reset changes no
environment. Each client holds about 27 MB. The worker processes keep freed memory for the next clients,
so a reset does not map and write new pages.

```python
import stable_envs as snv

env = snv.PyFlytRocketLanding(num_envs=8)
state = env.reset(seed=0)
frames = env.render()
```

## PyFlytRocketLanding

<!-- env: PyFlytRocketLanding -->
