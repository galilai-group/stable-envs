# Performance tuning

Every environment runs well with no setup. Chunk sizes, Triton block sizes, worker counts and memory
budgets have defaults, and the library computes some defaults from the hardware. You can change them to
trade speed against memory. Frames and State are the same for every valid value.

The `simulation_backend` constructor argument selects physics for [Reacher](environments/dmcontrol.md#reacher),
[TwoRoom](environments/navigation.md#tworoom), TwoRoomTarget, Piecewise, PiecewiseZones, PushT, PushTDiscrete and the
[classic control](environments/classic-control.md#physics-backends) classes. On CUDA, the classic control classes step on the
GPU by default. With `"auto"`, they step in NumPy on the host below `torch_min_envs` environments. Reacher chooses native CPU physics below a provisional
cutoff of 32 environments and MJX for larger CUDA batches. The four navigation classes use Numba on CPU. On CUDA, TwoRoom
uses Torch below `jax_min_envs` and JAX at or above it. Its cutoff defaults to 1 because current measurements favor JAX for
one environment. [PushT](environments/pusht.md#physics-backends) steps in Numba on CPU and in Triton on CUDA; with
`"auto"`, CUDA batches below `triton_min_envs` step in Numba. The cutoffs are constructor arguments and require measurements on your hardware to
choose the fastest implementation. Reacher's backends use different precision and can produce
different trajectories, so these arguments are separate from the tuning knobs below.

The library stores measured values per render path. For example, every fixed-camera DM Control cell
uses the `dmc-suite` path, so a value measured with `DMCCartpoleSwingup` also applies to
`DMCReacher`. The Triton block and warp knobs and `ogb_mjw_worlds` have one measured value for the
whole device instead of one per path.

## Where a value comes from

When the library reads a knob, it uses the first value it finds in this list:

1. `snv.tuning.override(...)` blocks, innermost first, then `snv.tuning.set(...)`.
2. The environment variable `STABLE_ENVS_<KNOB>`, for example `STABLE_ENVS_CPU3D_CHUNK=8`.
3. A value that `stable-envs tune` measured on this machine for the render path, frame size and batch
   range of the call.
4. A value derived from the hardware, for knobs that have a rule.
5. The call site's default. TrackCom cells use a per-cell `cpu3d_chunk` table measured on a Xeon
   Platinum 8268.
6. The fallback in the table below.

The library reads knobs each time it does the work, so a change applies to the next call. These
inputs raise `ValueError`:

- an unknown knob name
- a value that is not an int
- a value that is too small
- a value that is not a power of two, for a knob that needs a power of two
- a warp count that the knob does not allow

```python
import stable_envs as snv

snv.tuning.set(cpu3d_chunk=8)                   # whole process
with snv.tuning.override(nav_render_chunk=64):  # this block
    frames = env.render(state)
snv.tuning.get("cpu3d_chunk")                   # current value
snv.tuning.describe()                           # every knob: (value, source)
```

To use different values for two environments, wrap each environment's calls in its own
`snv.tuning.override` block.

Environment variables take an integer with an optional `K`, `M`, `G` or `T` suffix (powers of 1024),
for example `STABLE_ENVS_NVD_BUDGET_BYTES=4G`. The value `auto` skips the variable. See
[Environment variables](environment-variables.md) for every variable.

## Knobs

Values are ints. The "machine" column names the machine that measured the fallback, when the record
names one.

| knob | fallback | derived default | machine | what it controls |
|---|---|---|---|---|
| `workers` | 8 | min(value, CPUs in the process affinity, lanes) | not recorded | Most worker processes of a CPU simulator batch, of a PushT CUDA reset, or of a PushT render, on Linux. The value 0 steps and resets every lane, and draws every PushT frame, in the main process. |
| `worker_min_lanes` | 64 | none | not recorded | Smallest batch that uses worker processes, or Fetch simulator threads. For PushT, the smallest number of rendered States that uses render workers. |
| `worker_timeout` | 120 | none | not recorded | Seconds that a CPU simulator worker can take to answer one request before the batch fails. A worker can take up to the larger of this value and 300 seconds to start. Set a larger value for environments whose reset or step takes minutes in one worker. |
| `fetch_threads` | 4 | min(value, CPUs in the process affinity, lanes) | not recorded | Most simulator threads of an in-process Fetch batch with `simulation_backend="source"`. |
| `dmc_threads` | 8 | min(value, CPUs in the process affinity, lanes) | not recorded | Most `mujoco.batch` threads of a DM Control batch, for CPU and CUDA output. A batch uses at most one thread per two lanes, and about one thread for each 0.1 ms of MuJoCo work in a step call. MJX Reacher does not read it. |
| `cpu3d_chunk` | 16 | none (TrackCom cells use a per-cell table) | Xeon Platinum 8268 | Frames per chunk of the Torch CPU ray caster (`cpu3d`, `backend="torch"`). The Numba kernel draws all frames in one call. |
| `dmc_cpu_ray_budget` | 2^21 | 2^21 × L3 size / 35.75 MiB, rounded to a power of two, in [2^18, 2^21] | not recorded | Rays per CPU render chunk of the fixed-camera DMC cells, counting supersamples. |
| `dmc_ray_budget` | 2^24 | min(2^24, free GPU memory / 512), rounded down to a power of two | not recorded | Rays per CUDA render chunk of the fixed-camera DMC cells. |
| `dmc_launch_rays` | 256 × 224 × 224 | none | not recorded | Rays per MJWarp launch of the DMC TrackCom cells and of the DMC `backend="mjwarp"` renderer. This value sets worlds per launch (at most 256). The `worlds` render option overrides it. |
| `dmc_min_worlds` | 64 | none | not recorded | Fewest worlds per MJWarp launch of the DMC TrackCom cells and of the DMC `backend="mjwarp"` renderer. |
| `fetch_mjw_worlds` | 256 | none | not recorded | Most worlds per MJWarp launch of the Fetch cells. |
| `ogb_mjw_worlds` | 256 | none | not recorded | Most worlds per MJWarp launch of the OGBench Cube and Scene cells. |
| `ogb_gl_workers` | 8 | min(value, CPUs in the process affinity, rows) | not recorded | Most OpenGL worker processes of an OGBench Cube and Scene batch on Linux. A batch that uses fewer than 2 workers draws in the main process. The value 0 draws every batch in the main process. |
| `render_workers` | 8 | min(value, CPUs in the process affinity) | not recorded | Most worker processes of a CPU render batch on Linux, for classic control when Numba is not installed, for Fetch with `backend="opengl"` (software OpenGL) and for `PyFlytRocketLanding` with `backend="pybullet"` (TinyRenderer). Each Fetch worker has its own OpenGL context. The values 0 and 1 draw every frame in the main process. |
| `render_worker_min_rows` | 16 | none | not recorded | Smallest classic control, Fetch or `PyFlytRocketLanding` CPU render batch that uses render worker processes. `PyFlytRocketLanding` uses 2 when the knob is not set. |
| `ogb_context_cache_bytes` | none | min(total GPU memory / 4, free GPU memory / 2), per device | not recorded | GPU bytes the OGBench MJWarp context cache keeps. |
| `nvd_budget_bytes` | 6 GiB | min(6 GiB, free GPU memory / 2) | not recorded | GPU bytes one nvdiffrast chunk of the Rocket cells may use. |
| `nav_render_chunk` | 256 | none | not recorded | Frames per render chunk of the navigation cells. |
| `navdot_block` | 1024 | none | not recorded | Pixels per Triton tile of the navigation dot compositor (power of two). |
| `navdot_warps` | 4 | none | not recorded | Triton warps per tile of the navigation dot compositor. |
| `raster2d_block` | 128 | none | not recorded | Pixels per Triton program of the 2-D rasterizer without culling (power of two). |
| `raster2d_cull_warps` | 4 | none | not recorded | Triton warps per tile of the 2-D rasterizer with culling. |
| `gfxdraw_warps` | 1 | none | not recorded | Triton warps per tile of the classic-control rasterizer. |
| `raycast_block` | 256 | none | not recorded | Rays per Triton program of the analytic ray caster (power of two). |

The library computes each hardware rule once per process and device. GPU rules read
`torch.cuda.mem_get_info`. The CPU rule reads the L3 size from `sysconf`. If the library
cannot read the hardware, the fallback applies.

### What to sweep

- Out of GPU memory: lower `dmc_ray_budget`, `nvd_budget_bytes`, `dmc_launch_rays`,
  `fetch_mjw_worlds`, `ogb_mjw_worlds` or `ogb_context_cache_bytes`.
- Slow CPU rendering of DMC cells: try `dmc_cpu_ray_budget` (fixed-camera cells) in powers of two.
  Smaller chunks stay in the CPU cache. With `backend="torch"`, also try `cpu3d_chunk` (TrackCom cells).
  The Numba kernels use one thread for each CPU in the process affinity. To use fewer threads, set
  `NUMBA_NUM_THREADS` before the first render.
- Slow CPU stepping: try `workers` between 0 and the number of cores, and `worker_min_lanes`
  for small batches. The library uses worker processes only on Linux. For DM Control, try
  `dmc_threads` between 1 and the number of cores.
- Several DM Control envs in one process, or an env beside data-loader workers: set `dmc_threads`
  so that all of their threads fit in the CPU affinity.
- Slow classic control CPU rendering without Numba, or slow Fetch or `PyFlytRocketLanding` CPU rendering: try `render_workers`
  between 0 and the number of cores, and `render_worker_min_rows` for small batches.
- Classic control and navigation CPU rendering with Numba: the kernel uses one thread for each CPU in the process
  affinity. To use fewer threads, set `NUMBA_NUM_THREADS` before the first render.
- Triton kernels on a new GPU: try the `*_warps` knobs (1, 2, 4, 8) and the `*_block` knobs in powers of two.

## `stable-envs tune`

`stable-envs tune` measures the fastest values on this machine. Tuning is optional and covers one
environment at a time.

```bash
stable-envs tune DMCCheetahRun --batch 64 --device cpu
```

`stable-envs tune --all` tunes one registered environment for each render path that the device can
use. It lists the environments and asks before it starts. Pass `--yes` to start without the question,
for example in a batch job.

The command tunes one knob at a time, for each knob that the environment reads. It times the
default and a few powers of two around it. It keeps the best value of each knob while it tunes
the next knob. The command makes one warm-up call and five timed calls for each value, with CUDA
synchronized. The median counts.

The command skips a value that gives the same work split as another value, for example a chunk larger
than the batch. It also skips a value that would use more than half the free GPU memory. A run takes
about a minute.

The command checks that every value gives the same frames or State as the first. A difference is a bug
and stops the run.

The command prints a table and the location where it saved the values. Example output (the numbers depend on the machine):

```text
  knob                        default        tuned  speedup
  dmc_cpu_ray_budget          2097152       524288    1.12x
saved for: dmc-suite, 84 px, batch 64-127 (also covers: DMCCartpoleSwingup, DMCReacher, ...)
```

The command saves values for the render path, the frame size and the batch range. Batch ranges are
powers of two: 64 to 127, 128 to 255, and so on. For the knobs above that are not per path, the
command saves one value for the whole device.

The command saves the values in `~/.cache/stable-envs/tuning/<device name>.json`. When
`$XDG_CACHE_HOME` has a value, the command uses it instead of `~/.cache`. A file applies only on the
same device, driver, host CPU and stable-envs version. After an upgrade, tune again.

From Python, `snv.tune(env)` does the same with a constructed environment and returns the chosen values.
It restores the environment's State afterwards.

## Hint and autotune

If a render path has no measured values, the library logs a hint when the path first renders a
batch of 256 or more lanes. The hint is one `INFO` line on the `stable_envs.tuning` logger, for example
``"`stable-envs tune DMCCheetahRun --batch 512 --size 64 --device cuda` may speed up the dmc-trackcom render path
(about 1 min). To hide this message, set STABLE_ENVS_TUNING_HINT=0."``.
The library logs the hint at most once per path and device in a process. To hide it, set `STABLE_ENVS_TUNING_HINT=0`
or raise the `stable_envs.tuning` log level.

With `STABLE_ENVS_AUTOTUNE=1`, the library tunes instead of logging the hint. The first time such a
path renders, the library does these steps:

1. It builds a separate environment with the same class, batch, size and device.
2. It tunes that environment.
3. It caches the values.

That first render takes about a minute longer. A lock file beside the cache file lets only one process
tune. Another process that finds the lock held uses the defaults and does not wait. If a tune fails,
the library logs an `INFO` line and the render continues with the defaults. Autotune is off by default.

See [Command line](api/cli.md#stable-envs-tune) for every option.

## Reference

::: stable_envs.tuning
    options:
      members: [get, set, reset, override, describe, cache_path]
      show_root_heading: false
