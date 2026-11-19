# Performance tuning

Change the stable-envs tuning settings only when rendering or stepping is slower than you expect, or a large batch runs out of GPU memory. The defaults need no setup. A setting trades speed against memory, and every [frame](terms.md#frame) and [state](terms.md#state) stays the same.

Change a setting for one block with `snv.tuning.override(...)`, or for the whole process with `snv.tuning.set(...)` or its [`STABLE_ENVS_<SETTING>` variable](environment-variables.md#tuning-settings). [Tuning](api/tuning.md) lists every setting, its default and the order in which the library looks for a value.

## What to sweep

- Out of GPU memory: lower `dmc_ray_budget`, `dmc_graph_rows`, `nvd_budget_bytes`, `dmc_launch_rays`, `fetch_mjw_worlds`, `ogb_mjw_worlds` or `ogb_context_cache_bytes`.
- Slow CUDA rendering of small batches of the fixed-camera DM Control classes: try `dmc_graph_rows`. Its default, 48, was chosen from timings on an RTX 3090 host and an RTX A5000 host on 15 November 2026.
- Slow CPU rendering of DM Control: try `dmc_cpu_ray_budget` (fixed-camera classes) in powers of two; smaller chunks stay in the CPU cache. With `backend="torch"`, also try `cpu3d_chunk` (body-tracking classes).
- Slow CPU stepping: try `workers` between 0 and the number of cores, and `worker_min_envs` for small batches. For DM Control, try `dmc_threads` between 1 and the number of cores, and for Fetch with `simulation_backend="source"` when it steps without worker processes, `fetch_threads`.
- Several DM Control environments in one process, or one beside data-loader workers: set `dmc_threads` so that all their threads fit in the CPU affinity.
- Slow classic control CPU rendering without Numba, or slow Fetch (`backend="opengl"`), Miniworld, Roadster or `PyFlytRocketLanding` CPU rendering with `backend="pybullet"`: try `render_workers` between 0 and the number of cores, and `render_worker_min_rows` for small batches.
- Numba render kernels (DM Control, classic control, navigation, Fetch and PushT on CPU) use one thread for each CPU in the process affinity. To use fewer, set `NUMBA_NUM_THREADS` before Numba is imported, that is, before you construct the environment.
- `import stable_envs` sets `OMP_WAIT_POLICY=PASSIVE` when the variable is not set. Idle OpenMP threads of Torch and Numba then sleep instead of spin, so they do not delay the Numba parallel step on the same cores. To keep spinning threads, set `OMP_WAIT_POLICY=ACTIVE` before you start Python. OpenMP reads the variable when it starts, so if you import Torch and run an OpenMP operation before `import stable_envs`, the setting has no effect: import `stable_envs` first, or set the variable in the environment.
- Triton kernels on a new GPU: try the `*_warps` settings (1, 2, 4, 8) and the `*_block` settings in powers of two.

## Worker processes

On Linux with at least four CPUs available, some families use worker processes: classic `Acrobot` with `simulation_backend="gym"`, Fetch with `simulation_backend="source"` and `PyFlytRocketLanding` with `simulation_backend="pybullet"` step in them, PushT, `OGBCube` and `OGBScene` reset in them, and PushT draws its pygame frames in them. `worker_min_envs` is the smallest batch that uses workers (64 by default, 8 for the PushT resets, and 2 for the resets of `OGBCube` and `OGBScene`, because a reset of one of their batch rows runs milliseconds of OGBench Python, many times the round trip to a worker), and `workers` caps the number of processes (8 by default). `worker_timeout` is the number of seconds that a worker can take to answer one request before the batch fails (120 by default). Set a larger value for environments whose reset or step takes minutes in one worker. Each worker runs one simulation thread and stays inside the process's CPU affinity. On a CUDA device the worker outputs move to the device before `step` returns. Most other families step every [batch row](terms.md#batch-row) in one call in the main process, such as DM Control, MS-Human, Fetch, Roadster and the OGBench classes in `mujoco.batch` or MuJoCo Warp, Miniworld and PushT in Numba or Triton, and `PyFlytRocketLanding` in Numba. DM Control's `mujoco.batch` call runs on at most `dmc_threads` threads. `OGBCube` and `OGBScene` draw OpenGL frames in their own pool of up to `ogb_gl_workers` processes.

PushT resets in workers from 8 environments (`RESET_WORKER_MIN_ENVS` in `stable_envs/envs/pusht/simulation/execution.py`). A source reset of one environment takes about 3 to 5 ms of Python. We measured 8-environment and 16-environment resets on two hosts with 8 pinned cores (an RTX 3090 node and an RTX A5000 node, both AMD EPYC): in workers they took 15 to 23 ms and 19 to 32 ms, and in one process 28 to 31 ms and 53 to 57 ms. Worker start adds about 4 s when you construct the environment.

While workers run the next known action, the main process renders the previous state. At most one transition is in flight, and every one finishes before `step` returns. The outputs match a run without workers. `set_state` checks the state in every worker before any worker restores it. If one batch row is invalid, it raises and every batch row stays as it was. If rendering fails while a transition is in flight, the workers stop and you need to construct a new environment.

PyBullet allows 1024 clients in one process. A Rocket reset or restore holds one client more than the batch rows, and the step frames use one more. One process holds at most 1022 Rocket batch rows, and a larger batch uses enough processes to fit, on every platform and whatever the settings say, unless the environment runs inside a daemon process.

## Physics backends

The `simulation_backend` constructor argument picks the physics implementation of [DM Control](environments/dmcontrol.md#physics-backends), [classic control](environments/classic-control.md#simulation), [Fetch](environments/fetch.md#simulation), [Miniworld](environments/miniworld.md#physics-backends), [MS-Human](environments/mshuman.md#simulation), [OGBench](environments/ogbench.md#maze-physics), [PushT](environments/pusht.md#simulation), [PyFlyt](environments/pyflyt.md#simulation), [Roadster](environments/roadster.md#simulation), [TwoRoom](environments/navigation.md#tworoom-backends), `TwoRoomTarget`, `Piecewise` and `PiecewiseZones`. It's a constructor argument rather than a setting, because some choices change results: most GPU backends compute in float32, and their trajectories can differ from the CPU ones.

With the default `"auto"`, each class uses one backend for each device type, at every batch size. A seed then gives the same trajectory at every batch size. `env.simulation_backend` and `env.describe()["simulation_backend"]` name the backend an environment uses.

| Classes | `"auto"` on CPU | `"auto"` on CUDA | GPU backends by name |
|---|---|---|---|
| DM Control, MS-Human, Fetch, Roadster, the OGBench mazes, `OGBCube`, `OGBScene` | `mujoco.batch` (`"native"`) | `"native"` | `"mjwarp"`, and `"mjx"` for `DMCReacher` |
| PushT, `PushTDiscrete` | `"numba"` | `"numba"` | `"triton"`, `"jax"` |
| TwoRoom | `"numba"` | `"numba"` | `"torch"`, `"jax"` |
| Classic control | `"numba"` (CartPole, MountainCar), else `"gym"` or `"numpy"` (see below) | `"torch"` | |
| `TwoRoomTarget`, `Piecewise`, `PiecewiseZones`, `SimplePointMaze` | `"numba"` | `"torch"` | |
| Miniworld | `"numba"` | `"triton"` | |
| `PyFlytRocketLanding` | `"numba"` | `"numba_cuda"` | |

On CUDA, a CPU backend steps on the CPU, and one copy moves the state, the outputs and the `states` of a single step to the device. When no tensor uses the result of an earlier step any more, the copy goes into the device memory of that result. The CPU backends of the MuJoCo classes and PushT compute in float64, and MuJoCo Warp and Triton compute in float32. A default that changes with the batch size would thus give different trajectories for the same seed at different batch sizes. For this reason, the GPU backends of these classes are used only when you name them.

A GPU backend has a cost for each call that is almost the same for any batch up to a few hundred environments. So a GPU backend is faster than the CPU backend only for large batches. Use `simulation_backend="mjwarp"` (or `"triton"` for PushT) for large CUDA batches when the float32 physics is acceptable.

Classic control and the navigation classes other than TwoRoom step in Torch on CUDA, because a step of one action replays one captured CUDA graph. On 19 November 2026, on an NVIDIA L40S host (AMD EPYC 9554, 8 cores), a classic-control step of one environment took about 60 µs in Torch on CUDA and 270 to 420 µs on a host backend with the copy to CUDA. TwoRoom steps in Numba on CUDA, because its Numba step of one environment (110 µs) was faster than its Torch step (130 µs) on the same host. The Numba and Torch steps of TwoRoom give the same values.

On CPU, classic control steps Acrobot and MountainCarContinuous with one Gymnasium environment per batch row below 4 and 8 environments (`CPU_MIN_ENVS` in `stable_envs/envs/classic_control/simulation/batched.py`), and in NumPy from these batch sizes. NumPy steps equal Gymnasium steps bit for bit, so this choice changes only the speed. At one environment, Gymnasium stepped these two classes about twice as fast as NumPy on the L40S host.

## DM Control threads

On the CPU, a DM Control batch steps in one `mujoco.batch` call on at most `dmc_threads` threads, one per CPU in the process affinity and one per batch row. A domain with little MuJoCo work per step uses fewer threads: one per `ENVS_PER_THREAD` batch rows (in `stable_envs/envs/dmcontrol/simulation/factory.py`), so that each thread gets about 0.1 ms of MuJoCo work per call, and each thread gets at least two batch rows. Nothing is timed when you build an environment. The rows per thread were measured on 15 November 2026 on two hosts with 8 CPU cores each, an AMD EPYC 7662 and an Intel Xeon Platinum 8268: the median time of one environment's `mj_step` call with the task's substeps, as the mean of the two hosts, sets the rows that hold 0.1 ms of work. `dmc_threads` lowers this count further: a value larger than the count has no effect. `stable-envs tune` can measure it for a batch range.

The differential IK of `OGBCube` and `OGBScene` uses one thread for each `IK_ENVS_PER_THREAD` batch rows (128, in `stable_envs/envs/ogbench/simulation/manipulation_control.py`), so batches of fewer than 256 rows solve the IK on one thread. This value was measured on one host, an NVIDIA L40S node with 8 CPU cores. On an AMD EPYC 7532 and an Intel Xeon Platinum 8268, the IK alone was faster on 2 to 4 threads at 16 rows, but inside a full step the gain at 8 to 16 rows was not consistent, so the value did not change. The thread count does not change the results.

## Renderers

`PushT`, `PushTDiscrete`, `TwoRoom` and `TwoRoomTarget` have a CPU renderer and a CUDA renderer that draw the same bytes. A CUDA renderer has a fixed cost for each call, so a small batch draws faster on the CPU. With the default `backend="auto"` on CUDA, a call that draws fewer than `large_renderer_min_rows` frames uses the CPU renderer. The frames then move to CUDA in one copy from pinned memory. A larger call uses the CUDA renderer. This applies to `render` and to the frames that `step` returns. An explicit `backend` draws every batch with that renderer.

The switch point comes from, in this order: `snv.tuning.override`, `snv.tuning.set` or `STABLE_ENVS_LARGE_RENDERER_MIN_ROWS`; the tuning cache that `stable-envs tune <Env> --batch N --device cuda` writes; and the value each class ships. Nothing is timed when you build an environment or draw a frame. `stable-envs tune` times both renderers at the batch size you give and stores the choice for that power-of-two batch range. A range above one that chose the CUDA renderer, or below one that chose the CPU renderer, follows it.

Because both renderers draw the same bytes, the switch point, a tuning cache entry and the batch size never change a frame: `render(state)` and `render(state[:10])` give the same pixels for the same rows. `tests/gpu/test_render_backends.py` checks this, with the default and with a sampled variation. A class gets a switch point only when its two renderers draw the same bytes. The DM Control CPU and Triton ray casters differ by one level in a few pixels, so DM Control draws every CUDA batch with Triton.

The shipped switch points were measured on 15 November 2026 on hosts with an NVIDIA RTX 3090 (AMD EPYC 7532 and 7413) and an NVIDIA RTX A5000 (AMD EPYC 7662), 8 CPU cores each. On each host, both renderers drew batches of 1 to 256 frames, from `render` and from `step`, after 1000 warm-up calls, and the speed benchmark (`benchmarks/speed.py`) drew stored rows on the RTX 3090 host. `python -m benchmarks.renderer_switch <Env>` times both renderers of a class at batches 1, 2, 4, and so on, the way `stable-envs tune` times them, and prints the smallest batch from which the CUDA renderer is faster at every larger batch. The values are 32 for PushT and 64 for TwoRoom.

Where the measurements disagree, a class ships the smaller value. The CPU renderer runs on the CPU cores of the process, beside other work such as data loading, and near the switch point the two renderers are about equally fast. The smaller value keeps the CPU renderer to the batches where it was faster on every host.

## Render ahead

`snv.FrameLoader` draws at most `Env.render_ahead_rows` stored states (1024 by default) in one `render` call, and fewer when the free device memory is smaller. On 15 November 2026 we timed one `render` call per batch for `PushT`, `TwoRoom`, `DMCReacher` and `OGBCube` at batches of 1 to 1024, on an Intel Xeon Platinum 8268 node (8 cores) and on an RTX A5000 host. The frame rate stopped growing by 256 rows for every class on both devices, except `PushT` on CUDA, which still grew at 1024 rows. A second GPU host was not measured. A larger value does not change the frames of a class whose renderer bytes do not depend on the batch size.

## Measure the fastest values

Run `stable-envs tune DMCCheetahRun --batch 64 --device cpu` to measure the fastest values for one environment on this machine. It caches them, and later runs with the same render path, frame size and range of batch sizes use them. A process reads the cache when it builds an environment, so a cache that another process writes takes effect at the next environment build. [`stable-envs tune`](api/cli.md#stable-envs-tune) explains how it picks values and lists every option.

## Hint and autotune

If a render path has no measured values, the library logs one `INFO` hint the first time the path renders a batch of 256 or more batch rows. The default `WARNING` verbosity hides it. It appears after `snv.set(verbose="INFO")` or with `STABLE_ENVS_VERBOSE=INFO`:

```text
`stable-envs tune DMCCheetahRun --batch 512 --size 64 --device cuda` may speed up the dmc-trackcom render path (about 1 min). To hide this message, set STABLE_ENVS_TUNING_HINT=0.
```

With `STABLE_ENVS_AUTOTUNE=1`, the library tunes instead of logging the hint. The first time such a path renders, it builds a separate environment with the same class, batch, size and device, tunes it and caches the values. That first render takes as long as a tune run. A lock file beside the cache lets only one process tune, and another process that finds the lock held uses the defaults without waiting. If a tune fails, the library logs a `WARNING` and the render continues with the defaults. Autotune is off by default.
