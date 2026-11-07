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
TinyRenderer. A step then does not export a State only to draw it. The workers also draw `render`
batches of 2 or more States. These frames are byte-identical to frames drawn in the main process.

```python
import stable_envs as snv

env = snv.PyFlytRocketLanding(num_envs=8)
state = env.reset(seed=0)
frames = env.render()
```

## PyFlytRocketLanding

<!-- env: PyFlytRocketLanding -->
