# Tuning

`stable_envs.tuning` holds the knobs that set chunk sizes, Triton block sizes, worker counts and memory budgets. A knob changes speed and memory use, and every value gives the same [frames](../terms.md#frame) and [states](../terms.md#state). Each knob has a default and a fixed order of sources for its value, both listed below. See [Performance tuning](../performance-tuning.md) for when to change one.

```python
import stable_envs as snv

print(snv.tuning.get("nav_render_chunk"))               # 256
snv.tuning.set(nav_render_chunk=64)                     # for the whole process
with snv.tuning.override(nav_render_chunk=32):          # for this block only
    print(snv.tuning.describe()["nav_render_chunk"])    # (32, 'override')
print(snv.tuning.describe()["nav_render_chunk"])        # (64, 'set')
snv.tuning.set(nav_render_chunk=None)                   # clear it again
print(snv.tuning.describe()["nav_render_chunk"])        # (256, 'fallback')
```

## How a value is chosen

The library reads a knob each time it does the work, and a change applies to the next call. `workers`, `worker_min_lanes`, `worker_timeout`, `fetch_threads` and `dmc_threads` are read when the environment builds its worker processes or threads. The library uses the first value it finds:

| Source | Set by |
|---|---|
| `override` | `snv.tuning.override(...)` blocks, innermost first |
| `set` | `snv.tuning.set(...)` |
| `env` | The `STABLE_ENVS_<KNOB>` environment variable, such as `STABLE_ENVS_NAV_RENDER_CHUNK=64` |
| `cache` | `stable-envs tune` or `snv.tune(env)`, for this device, render path, frame size and batch range |
| `derived` | A rule that reads the hardware, for the knobs that have one |
| `default` | The call site's own value, such as the per-class `cpu3d_chunk` table of the DM Control body-tracking classes |
| `fallback` | `KNOBS[name].fallback` |

`describe()` returns the value and the source of every knob. Overrides live in a context variable and stay in the thread that set them. Wrapping each environment's calls in its own `override` block gives two environments different values. Knob values are per render path, the code that draws a group of classes: every fixed-camera DM Control class shares the `dmc-suite` path, and a value measured with `DMCCartpoleSwingup` also applies to `DMCReacher`. The Triton block and warp knobs and `ogb_mjw_worlds` have one value for the whole device instead.

An unknown name, a non-int value, a value that is too small, or a value of the wrong kind (a power of two or a Triton warp count) raises `ValueError`:

```python
try:
    snv.tuning.set(navdot_block=3)
except ValueError as e:
    print(e)  # tuning knob 'navdot_block' takes a power of two, got 3
```

## Knobs

Values are ints, and "batch rows" means the batch rows of one call. Each hardware rule runs once per process and device: GPU rules read `torch.cuda.mem_get_info` and the CPU rule reads the L3 size from `sysconf`. When reading the hardware fails, the fallback applies.

| Knob | Type | Fallback | Derived default | What it controls |
|---|---|---|---|---|
| `workers` | int (0 or more) | 8 | min(value, CPUs in the process affinity, batch rows) | Most worker processes of a CPU simulator batch, of a PushT, `OGBCube` or `OGBScene` reset, or of a PushT render, on Linux. The value 0 steps and resets every batch row, and draws every PushT frame, in the main process, except that a `PyFlytRocketLanding` batch of more than 1022 environments always uses processes. |
| `worker_min_lanes` | int | 64 | none | Smallest batch that uses worker processes, or Fetch simulator threads. For PushT, the smallest number of rendered states that uses render workers. |
| `worker_timeout` | int | 120 | none | Seconds that a CPU simulator worker can take to answer one request before the batch fails. A worker can take up to the larger of this value and 300 seconds to start. Set a larger value for environments whose reset or step takes minutes in one worker, for example with `snv.tuning.set(worker_timeout=600)`. |
| `fetch_threads` | int | 4 | min(value, CPUs in the process affinity, batch rows) | Most simulator threads of an in-process Fetch batch with `simulation_backend="source"`. |
| `dmc_threads` | int | 8 | min(value, CPUs in the process affinity, batch rows) | Most `mujoco.batch` threads of a DM Control batch, for CPU and CUDA output. A batch uses at most one thread per two batch rows, and about one thread for each 0.1 ms of MuJoCo work in a step call. With MuJoCo Warp, it only sizes the reset threads of `CheetahRun` and `QuadrupedWalk`, and MJX doesn't read it. |
| `cpu3d_chunk` | int | 16 | none (body-tracking classes use a per-class table) | Frames per chunk of the Torch CPU ray caster (`cpu3d`, `backend="torch"`). The Numba kernel draws all frames in one call. The fallback was measured on a Xeon Platinum 8268. |
| `dmc_cpu_ray_budget` | int, power of two | 2^21 | 2^21 × L3 size / 35.75 MiB, rounded to a power of two, in [2^18, 2^21] | Rays per CPU render chunk of the fixed-camera DM Control classes, counting supersamples. |
| `dmc_ray_budget` | int, power of two | 2^24 | min(2^24, free GPU memory / 512), rounded down to a power of two | Rays per CUDA render chunk of the fixed-camera DM Control classes. |
| `dmc_launch_rays` | int | 256 × 224 × 224 | none | Rays per MJWarp launch of the DM Control body-tracking classes and of the DM Control `backend="mjwarp"` renderer. This value sets the worlds per launch: at most 256, unless `dmc_min_worlds` is larger. The `worlds` render option overrides it. |
| `dmc_min_worlds` | int, power of two | 64 | none | Fewest worlds per MJWarp launch of the DM Control body-tracking classes and of the DM Control `backend="mjwarp"` renderer. |
| `fetch_mjw_worlds` | int, power of two | 256 | none | Most worlds per MJWarp launch of the Fetch classes. |
| `ogb_mjw_worlds` | int, power of two | 256 | none | Most worlds per MJWarp launch of `OGBCube` and `OGBScene`. |
| `ogb_gl_workers` | int (0 or more) | 8 | min(value, CPUs in the process affinity), and on CUDA also rows ÷ 8 | Most OpenGL worker processes of an `OGBCube` or `OGBScene` batch on Linux. A batch that would use fewer than 2 workers draws in the main process, and so does a CPU batch with fewer states than workers. The value 0 draws every batch there. |
| `render_workers` | int (0 or more) | 8 | min(value, CPUs in the process affinity) | Most worker processes of a CPU render batch on Linux, for classic control without Numba, for Fetch with `backend="opengl"` (software OpenGL), for Miniworld, for Roadster, for the `info.goal` images of SimplePointMaze and for `PyFlytRocketLanding` with `backend="pybullet"` (TinyRenderer). Each Fetch worker has its own OpenGL context. The values 0 and 1 draw every frame in the main process. |
| `render_worker_min_rows` | int | 16 | none | Smallest CPU render batch that uses render worker processes, in every family that has them. `PyFlytRocketLanding` uses 2 when the knob is unset. |
| `ogb_context_cache_bytes` | bytes | none | min(total GPU memory / 4, free GPU memory / 2), per device | GPU bytes the OGBench MJWarp context cache keeps. |
| `nvd_budget_bytes` | bytes | 6 GiB | min(6 GiB, free GPU memory / 2) | GPU bytes one nvdiffrast chunk of `PyFlytRocketLanding` may use. |
| `nav_render_chunk` | int | 256 | none | Frames per render chunk of the navigation classes. |
| `navdot_block` | int, power of two | 1024 | none | Pixels per Triton tile of the navigation dot compositor (power of two). |
| `navdot_warps` | 1, 2, 4, 8, 16 or 32 | 4 | none | Triton warps per tile of the navigation dot compositor. |
| `raster2d_block` | int, power of two | 128 | none | Pixels per Triton program of the 2D rasterizer without culling (power of two). |
| `raster2d_cull_warps` | 1, 2, 4, 8, 16 or 32 | 4 | none | Triton warps per tile of the 2D rasterizer with culling. |
| `gfxdraw_warps` | 1, 2, 4, 8, 16 or 32 | 1 | none | Triton warps per tile of the classic control rasterizer. |
| `raycast_block` | int, power of two | 256 | none | Rays per Triton program of the analytic ray caster (power of two). |

`snv.tuning.KNOBS` maps each knob name to its `Knob`, and `snv.tuning.RENDER_PATHS` maps an environment class or family name to the render and simulation paths whose measured values it shares:

```python
knob = snv.tuning.KNOBS["nav_render_chunk"]
print(knob.env_var, knob.fallback, knob.paths)  # STABLE_ENVS_NAV_RENDER_CHUNK 256 ('navigation',)
print(snv.tuning.RENDER_PATHS["navigation"])    # ('navigation', 'raster2d')
```

## Read and set knobs

::: stable_envs.tuning.get

::: stable_envs.tuning.lookup

::: stable_envs.tuning.set

::: stable_envs.tuning.override

::: stable_envs.tuning.reset

::: stable_envs.tuning.describe

::: stable_envs.tuning.cache_path

## Knob definitions

::: stable_envs.tuning.Knob
