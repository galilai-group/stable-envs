# PyFlyt

A rocket falls from a height with a random tilt and offset, and the agent uses the engine and the fins to land it upright on a pad. `PyFlytRocketLanding` wraps this rocket landing task of [PyFlyt](https://github.com/jjshoots/PyFlyt) ([Tai et al., 2023](https://arxiv.org/abs/2304.01305)). A Numba port of the PyFlyt and PyBullet physics steps the batch on CPU or CUDA, and a Numba port of TinyRenderer or nvdiffrast draws the [frames](../terms.md#frame).

```python
# needs the rocket extra
import stable_envs as snv

with snv.PyFlytRocketLanding(num_envs=8, device="cpu") as env:
    state = env.reset(seed=0)
    frames = env.render()
```

## Install

The class needs the `rocket` extra, which pins `PyFlyt==0.29.0` and `pybullet==3.2.7` and installs Numba. CUDA rendering needs Triton and nvdiffrast, which isn't on PyPI; [build it from source](../install.md#nvdiffrast).

`snv.import_errors()` reports a failed import of the family under `rocket_landing`, `rocket_render` and `rocket_semantics`.

## Task

The action has seven values:

| Index | Control | Bounds |
|---|---|---|
| 0, 1, 2 | finlet deflection in x and y, and finlet roll | [-1, 1] |
| 3 | engine ignition | [0, 1] |
| 4 | engine throttle | [0, 1] |
| 5, 6 | engine gimbal in x and y | [-1, 1] |

PyFlyt sets `env_complete` when the rocket lands on the pad at a low speed, and the step output reports it as `success`.

## Simulation

`simulation_backend` selects the physics:

| `simulation_backend` | Physics | Device of the physics | `execution_mode` | State |
|---|---|---|---|---|
| `"numba"` (default on CPU) | Numba port of the PyFlyt forces and the Bullet rigid-body step, all environments in one call | CPU, parallel over environments | `NATIVE_NUMBA` on CPU, `HOST_NUMBA` on CUDA | physical values |
| `"numba_cuda"` (default on CUDA) | CUDA build of the same kernel, one GPU thread per environment, in one CUDA graph per step | CUDA | `NATIVE_NUMBA_CUDA` | physical values |
| `"auto"` | `"numba_cuda"` on CUDA from `numba_cuda_min_envs` environments (default 1, measured on an L40S), else `"numba"` | CPU or CUDA | as the selected backend | physical values |
| `"pybullet"` | PyFlyt 0.29.0 and PyBullet 3.2.7, one engine per environment, in worker processes for large batches | CPU | `DELEGATED_CPU` | reset recipe and action history |

`None` selects the default of the device. With `"numba"` on CUDA, the kernel steps on the host and the outputs move to the device. With `"numba_cuda"`, a step reads the actions once on the host to check them; the physics, the [state](../terms.md#state) and the outputs stay on the device. The leaves of a state are views of one buffer, which a few copies fill, and `env.step` stacks the states of a step sequence with one copy. A reset samples the starts and seeds the noise generators on the host, then settles every environment on the device. The CUDA build can round the last bits differently from the CPU kernel.

The Numba backend models the rocket as one rigid body, as PyBullet does: all its links are fixed. It applies the PyFlyt body drag, the four lifting surfaces, the gimballed booster with fuel burn, and the thrust noise. In free flight, a step agrees with PyBullet to double-precision rounding. The contacts with the ground and the pad are approximate. In landing rollouts of 300 or more steps, the episode outcome (crash or landing) is the same as the source's, and the episode can end up to 3 steps earlier or later.

The thrust noise of each environment comes from its own PCG64 generator, which the reset seeds as the source does. The same seed gives the same start state and the same noise as PyFlyt. The state holds the position in each generator, so `set_state` restores the noise exactly.

The state of the Numba backend holds the physical values of the kernel and the PCG64 state of each noise generator (`physics/*`), the variation values and the drawn colors. The renderers read the same `current/*` and `render/*` leaves as with the PyBullet backend, so a state with the same poses and colors draws the same frames. `set_state` copies the values and does not replay the episode. The Numba backend accepts only `agent_hz=40`, the registered value. Use `simulation_backend="pybullet"` for the replay-backed state that is exact to the source, or for another `agent_hz`.

### PyBullet backend

With worker processes on CPU, the workers draw the step frames of the environments they run with TinyRenderer, without exporting a state to draw them. On CUDA, the workers send the frame fields and nvdiffrast draws them in the main process.

A step reads from PyBullet only the state leaves that a step can change: the rocket pose, velocity and fuel tank mass, the cached link poses, and the task, RNG and history fields. The other leaves come from the last reset, configuration or restore, and the state equals a full read of the engine. A large leaf that is all zero, such as `current/wrapper/current_goal` without a render mode, isn't stored or copied for each environment. On CUDA, the outputs of the worker processes move to the device as they arrive.

Each PyBullet client holds about 27 MB. A reset or a restore replaces the client of one environment at a time: it makes the new client and then closes the old one, so a worker process holds at most one client more than its environments. A reset checks the arguments of every environment before it changes one. If PyBullet fails during a reset, the environments before the failed one can already have a new episode. A restore keeps the state of each environment it replaced, and if a later environment fails, it replays these states, so no environment changes.

## Rendering

On CPU, a Numba port of TinyRenderer, PyFlyt's own renderer, draws the frames. On CUDA, nvdiffrast draws them. The `backend` argument selects the CPU renderer:

| `backend` | Renderer |
|---|---|
| `"numba"` | A Numba port of TinyRenderer that draws all the frames in one call, one frame for each thread |
| `"pybullet"` (also `"tiny"`) | PyBullet's TinyRenderer, one `getCameraImage` call for each frame |
| `"auto"` (default) | `"numba"` |

With `simulation_backend="pybullet"` and worker processes, the workers draw the step frames with PyBullet's TinyRenderer for each value of `backend`.

The Numba port uses the meshes that TinyRenderer makes from the URDFs: the Bullet convex hull of each cylinder, the textured cube of each box and the `plane100.obj` ground. It also uses the arithmetic of TinyRenderer: float32 vertex transforms, near-plane clipping, double-precision barycentric coordinates at integer pixel positions, a float32 depth buffer, the default light without shadows, and truncation to `uint8`. On CUDA, the renderer uses the same meshes, the same float32 matrices, the same rule for small triangles and the same color law. Its coverage and its interpolation of normals are float32.

`render(state)` checks only the state leaves that the frame reads: the body poses, the first camera link pose in `current/diagnostic/cached_links` and the colors in `render/rgba`. `to_render_state(state)` also checks the first 13 values of `current/wrapper/state`. They check the shape, the dtype, that the values are finite and that the colors are in [0, 1]. `set_state` checks all the leaves. A state to draw must have exactly `num_envs` batch rows.

`render` also accepts the render state from `to_render_state`. On CUDA, `render` also accepts `(B, 13)` pose rows from `sample_states` and draws them with the current variation, with the flame off. On CPU, `render` refuses rows: give a full state or a render state.

With `backend="pybullet"` on Linux, `render` draws a batch of 2 or more states in render worker processes. `render_workers` sets the number of workers, and `render_worker_min_rows` sets the smallest such batch (default 2). These frames are byte-identical to frames drawn in the main process.

## Variation

The `environment.start_*` [variation leaves](../tutorials/variation.md) set the start height, horizontal offset and tilt, and change the dynamics. The other leaves are colors: of the rocket parts, of the pad, and `environment.sky_color`, which PyFlyt declares but never draws.

## Classes

### PyFlytRocketLanding

<!-- env: PyFlytRocketLanding -->

## Limitations

- The nvdiffrast frames differ from the TinyRenderer frames by a [measured amount](../environments.md#comparison-with-the-upstream-library).
- The Numba physics backends approximate the contacts with the ground and the pad. Use `simulation_backend="pybullet"` for steps that are exact to the source.
