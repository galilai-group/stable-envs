# Performance tuning

Change the stable-envs tuning knobs only when rendering or stepping is slower than you expect, or a large batch runs out of GPU memory. The defaults need no setup. A knob trades speed against memory, and every [frame](terms.md#frame) and [state](terms.md#state) stays the same.

Set a knob for one block with `snv.tuning.override(...)`, or for the whole process with `snv.tuning.set(...)` or its [`STABLE_ENVS_<KNOB>` variable](environment-variables.md#tuning-knobs). [Tuning](api/tuning.md) lists every knob, its default and the order in which the library looks for a value.

## What to sweep

- Out of GPU memory: lower `dmc_ray_budget`, `nvd_budget_bytes`, `dmc_launch_rays`, `fetch_mjw_worlds`, `ogb_mjw_worlds` or `ogb_context_cache_bytes`.
- Slow CPU rendering of DM Control: try `dmc_cpu_ray_budget` (fixed-camera classes) in powers of two; smaller chunks stay in the CPU cache. With `backend="torch"`, also try `cpu3d_chunk` (body-tracking classes).
- Slow CPU stepping: try `workers` between 0 and the number of cores, and `worker_min_lanes` for small batches. For DM Control, try `dmc_threads` between 1 and the number of cores, and for Fetch with `simulation_backend="source"` when it steps without worker processes, `fetch_threads`.
- Several DM Control environments in one process, or one beside data-loader workers: set `dmc_threads` so that all their threads fit in the CPU affinity.
- Slow classic control CPU rendering without Numba, or slow Fetch (`backend="opengl"`), Miniworld, Roadster or `PyFlytRocketLanding` CPU rendering with `backend="pybullet"`: try `render_workers` between 0 and the number of cores, and `render_worker_min_rows` for small batches.
- Numba render kernels (DM Control, classic control, navigation, Fetch and PushT on CPU) use one thread for each CPU in the process affinity. To use fewer, set `NUMBA_NUM_THREADS` before Numba is imported, that is, before you construct the environment.
- Triton kernels on a new GPU: try the `*_warps` knobs (1, 2, 4, 8) and the `*_block` knobs in powers of two.

## Worker processes

On Linux with at least four CPUs available, some families use worker processes: classic `Acrobot` with `simulation_backend="gym"`, Fetch with `simulation_backend="source"` and `PyFlytRocketLanding` with `simulation_backend="pybullet"` step in them, PushT, `OGBCube` and `OGBScene` reset in them, and PushT draws its pygame frames in them. `worker_min_lanes` is the smallest batch that uses workers (64 by default), and `workers` caps the number of processes (8 by default). `worker_timeout` is the number of seconds that a worker can take to answer one request before the batch fails (120 by default). Set a larger value for environments whose reset or step takes minutes in one worker. Each worker runs one simulation thread and stays inside the process's CPU affinity. On a CUDA device the worker outputs move to the device before `step` returns. Most other families step every [batch row](terms.md#batch-row) in one call in the main process, such as DM Control, MS-Human, Fetch, Roadster and the OGBench classes in `mujoco.batch` or MuJoCo Warp, Miniworld and PushT in Numba or Triton, and `PyFlytRocketLanding` in Numba. DM Control's `mujoco.batch` call runs on at most `dmc_threads` threads. `OGBCube` and `OGBScene` draw OpenGL frames in their own pool of up to `ogb_gl_workers` processes.

While workers run the next known action, the main process renders the previous state. At most one transition is in flight, and every one finishes before `step` returns. The outputs match a run without workers. `set_state` checks the state in every worker before any worker restores it. If one batch row is invalid, it raises and every batch row stays as it was. If rendering fails while a transition is in flight, the workers stop and you need to construct a new environment.

PyBullet allows 1024 clients in one process. A Rocket reset or restore holds one client more than the batch rows, and the step frames use one more. One process holds at most 1022 Rocket batch rows, and a larger batch uses enough processes to fit, on every platform and whatever the knobs say, unless the environment runs inside a daemon process.

## Physics backends

The `simulation_backend` constructor argument picks the physics implementation of [DM Control](environments/dmcontrol.md#physics-backends), [classic control](environments/classic-control.md#simulation), [Fetch](environments/fetch.md#simulation), [Miniworld](environments/miniworld.md#physics-backends), [MS-Human](environments/mshuman.md#simulation), [OGBench](environments/ogbench.md#maze-physics), [PushT](environments/pusht.md#simulation), [PyFlyt](environments/pyflyt.md#simulation), [Roadster](environments/roadster.md#simulation), [TwoRoom](environments/navigation.md#tworoom-backends), `TwoRoomTarget`, `Piecewise` and `PiecewiseZones`. On CUDA, most of these step on the GPU by default, and with `"auto"`, batches below a cutoff use another backend: the CPU one below `mjwarp_min_envs` for the MuJoCo families, `torch_min_envs` for classic control and `triton_min_envs` for Miniworld and PushT, and Torch below `jax_min_envs` for TwoRoom. It's a constructor argument rather than a knob, because some choices change results: most GPU backends compute in float32, and their trajectories can differ from the CPU ones. The cutoffs are constructor arguments too, measured on one machine; measure on your hardware before you change them.

## Measure the fastest values

Run `stable-envs tune DMCCheetahRun --batch 64 --device cpu` to measure the fastest values for one environment on this machine. It caches them, and later runs with the same render path, frame size and range of batch sizes use them. [`stable-envs tune`](api/cli.md#stable-envs-tune) explains how it picks values and lists every option.

## Hint and autotune

If a render path has no measured values, the library logs one `INFO` hint the first time the path renders a batch of 256 or more batch rows. The default `WARNING` verbosity hides it. It appears after `snv.set(verbose="INFO")` or with `STABLE_ENVS_VERBOSE=INFO`:

```text
`stable-envs tune DMCCheetahRun --batch 512 --size 64 --device cuda` may speed up the dmc-trackcom render path (about 1 min). To hide this message, set STABLE_ENVS_TUNING_HINT=0.
```

With `STABLE_ENVS_AUTOTUNE=1`, the library tunes instead of logging the hint. The first time such a path renders, it builds a separate environment with the same class, batch, size and device, tunes it and caches the values. That first render takes as long as a tune run. A lock file beside the cache lets only one process tune, and another process that finds the lock held uses the defaults without waiting. If a tune fails, the library logs a `WARNING` and the render continues with the defaults. Autotune is off by default.
