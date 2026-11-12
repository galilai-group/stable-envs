# PyFlyt

`PyFlytRocketLanding` ports the rocket landing task of [PyFlyt](https://github.com/jjshoots/PyFlyt)
([Tai et al., 2023](https://arxiv.org/abs/2304.01305)). A rocket falls from a height with a random tilt
and offset. The agent uses the engine and the fins to land it upright on a pad. The physics runs on CPU.

The action has seven values:

| Index | Control | Bounds |
|---|---|---|
| 0, 1, 2 | finlet deflection in x and y, and finlet roll | [-1, 1] |
| 3 | engine ignition | [0, 1] |
| 4 | engine throttle | [0, 1] |
| 5, 6 | engine gimbal in x and y | [-1, 1] |

**Success:** the source sets `env_complete` when the rocket lands on the pad at a low speed. The step
output reports it as `success`.

## Simulation backends

`simulation_backend` selects the physics:

| `simulation_backend` | Physics | Device of the physics | `execution_mode` | State |
|---|---|---|---|---|
| `"numba"` (default on CPU) | Numba port of the PyFlyt forces and the Bullet rigid-body step, all environments in one call | CPU, parallel over environments | `NATIVE_NUMBA` on CPU, `HOST_NUMBA` on CUDA | physical values |
| `"numba_cuda"` (default on CUDA) | CUDA build of the same kernel, one GPU thread per environment, in one CUDA graph per step | CUDA | `NATIVE_NUMBA_CUDA` | physical values |
| `"auto"` | `"numba_cuda"` on CUDA from `numba_cuda_min_envs` environments (default 1, measured on an L40S), else `"numba"` | CPU or CUDA | as the selected backend | physical values |
| `"pybullet"` | PyFlyt 0.29.0 and PyBullet 3.2.7, one engine per environment, in worker processes for large batches | CPU | `DELEGATED_CPU` | reset recipe and action history |

`None` selects the default of the device. With `"numba"` on CUDA, the kernel steps on the host and the outputs
move to the device. With `"numba_cuda"`, a step reads the actions once on the host to check them; the physics,
the State and the outputs stay on the device. The leaves of a State are views of one buffer, which a few copies
fill, and `env.step` stacks the States of a step sequence with one copy. A reset samples the starts and seeds the
noise generators on the host, then settles every environment on the device. The CUDA build can round the last
bits differently from the CPU kernel.

The Numba backend models the rocket as one rigid body, as PyBullet does: all its links are fixed. It applies
the PyFlyt body drag, the four lifting surfaces, the gimballed booster with fuel burn, and the thrust noise.
In free flight, a step agrees with PyBullet to double-precision rounding. The contacts with the ground and the
pad are approximate. In landing rollouts of 300 or more steps, the episode outcome (crash or landing) is the
same as the source's, and the episode can end up to 3 steps earlier or later.

The thrust noise of each environment comes from its own PCG64 generator, which the reset seeds as the
source does. The same seed gives the same start state and the same noise as PyFlyt. The State holds the
position in each generator, so `set_state` restores the noise exactly.

The State of the Numba backend holds the physical values of the kernel and the PCG64 state of each noise
generator (`physics/*`), the Variation values and the drawn colours. The renderers read the same `current/*` and
`render/*` leaves as with the PyBullet backend, so a State with the same poses and colours draws the same frames.
`set_state` copies the values and does not replay the episode. The Numba backend accepts only `agent_hz=40`,
the registered value. Use `simulation_backend="pybullet"` for the replay-backed State that is exact to the
source, or for another `agent_hz`.

## Rendering

The class needs the `rocket` extra, which pins `PyFlyt==0.29.0` and `pybullet==3.2.7` and installs Numba. On CPU,
TinyRenderer draws the frames. On CUDA, nvdiffrast draws them. nvdiffrast is not on PyPI. See
[Install](../install.md). TinyRenderer is the source's renderer. The nvdiffrast frames differ from the
TinyRenderer frames. `env.describe()["visual_fidelity"]` gives the measured agreement.

With the PyBullet backend on CPU with worker processes, the workers draw the step frames of their running lanes with
TinyRenderer. A step then does not export a State only to draw it. On Linux, `render` draws a batch of
2 or more States in render worker processes, for every number of lanes. `render_workers` sets the number of
workers and `render_worker_min_rows` sets the smallest such batch.
These frames are byte-identical to frames drawn in the main process.

With each backend, `render(state)` checks only the State leaves that the frame reads: the body poses, the first
camera link pose in `current/diagnostic/cached_links` and the colours in `render/rgba`. `to_render_state(state)`
also checks the first 13 values of `current/wrapper/state`. They check the shape, the dtype, that the values are
finite and that the colours are in [0, 1]. `set_state` checks all the leaves.

With the PyBullet backend, a step reads from PyBullet only the State leaves that a step can change: the rocket pose, velocity and fuel
tank mass, the cached link poses, and the task, RNG and history fields. The other leaves come from the last
reset, configuration or restore. A zero leaf that a step does not change, such as `current/wrapper/current_goal`
without a render mode, is a new zero tensor and is not copied. The environments do not keep a stored copy of
such a leaf. The State is the same as a full read of the engine. On CUDA, the outputs of the worker
processes go to the device when they arrive.

Each PyBullet client holds about 27 MB. A reset or a restore replaces the client of one environment at a
time: it makes the new client and then closes the old one. Thus a worker process holds at most one client
more than its environments. A reset checks the arguments of every environment before it changes one. If
PyBullet fails during a reset, the environments before the failed one can have a new episode. A restore
keeps the State of each environment that it replaced. If a later environment is not valid, the restore
replays these States, so no environment changes. The worker processes keep freed memory for the next
clients, so a reset does not map and write new pages.

```python
import stable_envs as snv

env = snv.PyFlytRocketLanding(num_envs=8)
state = env.reset(seed=0)
frames = env.render()
```

## PyFlytRocketLanding

<!-- env: PyFlytRocketLanding -->
