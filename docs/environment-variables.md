# Environment variables

stable-envs reads the `STABLE_ENVS_*` environment variables below. Every one has a default, and setting them is optional.

```bash
STABLE_ENVS_CPU3D_CHUNK=8 STABLE_ENVS_TUNING_HINT=0 python train.py
stable-envs doctor    # lists the variables that are set and flags unknown names
```

An unparsable value raises `ValueError` with the variable's name when the library reads it. `stable-envs doctor` flags unknown names, such as a misspelling.

Integer values take an optional `K`, `M`, `G` or `T` suffix (powers of 1024). `auto` or an empty value uses the default. Flags take `1` or `0`.

Variables from other libraries, such as `MUJOCO_GL`, `PYOPENGL_PLATFORM` and `MUJOCO_EGL_DEVICE_ID`, are part of [installing each family](install.md#what-each-family-needs).

## Tuning knobs

Each variable sets one [tuning knob](api/tuning.md#knobs) for the whole process. [How a value is chosen](api/tuning.md#how-a-value-is-chosen) says when it wins.

| Variable | Sets |
|---|---|
| `STABLE_ENVS_WORKERS` | most worker processes |
| `STABLE_ENVS_WORKER_MIN_LANES` | smallest batch that uses worker processes |
| `STABLE_ENVS_WORKER_TIMEOUT` | seconds that a CPU simulator worker can take to answer one request |
| `STABLE_ENVS_FETCH_THREADS` | most simulator threads of an in-process Fetch batch |
| `STABLE_ENVS_DMC_THREADS` | most `mujoco.batch` threads of a DM Control batch |
| `STABLE_ENVS_CPU3D_CHUNK` | frames per chunk of the Torch CPU ray caster |
| `STABLE_ENVS_DMC_CPU_RAY_BUDGET` | rays per CPU render chunk of the fixed-camera DM Control classes |
| `STABLE_ENVS_DMC_RAY_BUDGET` | rays per CUDA render chunk of the fixed-camera DM Control classes |
| `STABLE_ENVS_DMC_LAUNCH_RAYS` | rays per MJWarp launch of the DM Control body-tracking classes |
| `STABLE_ENVS_DMC_MIN_WORLDS` | fewest worlds per MJWarp launch of the DM Control body-tracking classes |
| `STABLE_ENVS_FETCH_MJW_WORLDS` | most worlds per MJWarp launch of the Fetch classes |
| `STABLE_ENVS_OGB_MJW_WORLDS` | most worlds per MJWarp launch of `OGBCube` and `OGBScene` |
| `STABLE_ENVS_OGB_CONTEXT_CACHE_BYTES` | GPU bytes of the OGBench MJWarp context cache |
| `STABLE_ENVS_OGB_GL_WORKERS` | most OpenGL worker processes of `OGBCube` and `OGBScene` |
| `STABLE_ENVS_RENDER_WORKERS` | most CPU render worker processes |
| `STABLE_ENVS_RENDER_WORKER_MIN_ROWS` | smallest CPU render batch that uses render workers |
| `STABLE_ENVS_NVD_BUDGET_BYTES` | GPU bytes per nvdiffrast chunk of `PyFlytRocketLanding` |
| `STABLE_ENVS_NAV_RENDER_CHUNK` | frames per render chunk of the navigation classes |
| `STABLE_ENVS_NAVDOT_BLOCK` | pixels per Triton tile of the navigation dot compositor |
| `STABLE_ENVS_NAVDOT_WARPS` | Triton warps per tile of the navigation dot compositor |
| `STABLE_ENVS_RASTER2D_BLOCK` | pixels per Triton program of the 2D rasterizer |
| `STABLE_ENVS_RASTER2D_CULL_WARPS` | Triton warps per tile of the 2D rasterizer with culling |
| `STABLE_ENVS_GFXDRAW_WARPS` | Triton warps per tile of the classic control rasterizer |
| `STABLE_ENVS_RAYCAST_BLOCK` | rays per Triton program of the analytic ray caster |

## Tuning behavior

| Variable | Default | Effect |
|---|---|---|
| `STABLE_ENVS_TUNING_HINT` | 1 | Set to `0` to hide the one-time log line that suggests `stable-envs tune`. It logs at `INFO`, so it shows only with `STABLE_ENVS_VERBOSE=INFO` or lower. |
| `STABLE_ENVS_AUTOTUNE` | 0 | `1` tunes an untuned render path the first time it renders a large batch, then caches the values. |

## Logging

| Variable | Default | Effect |
|---|---|---|
| `STABLE_ENVS_VERBOSE` | `WARNING` | Default log level, with the same values as `snv.set(verbose=...)`. `snv.set` overrides it. |

## Environments

| Variable | Default | Effect |
|---|---|---|
| `STABLE_ENVS_MSHUMAN_HOME` | unset | Path to an existing MS-Human-700 checkout. When it's unset, stable-envs downloads the pinned commit into the cache (`$XDG_CACHE_HOME/stable-envs/ms-human-700`). |
| `STABLE_ENVS_DMC_CUDA_RENDER` | `auto` | `mjwarp` makes the fixed-camera DM Control environments with `backend="auto"` on CUDA draw with the MuJoCo Warp ray tracer, as the body-tracking ones do by default. See [DM Control](environments/dmcontrol.md#mujoco-warp-renderer). |
| `STABLE_ENVS_DMC_WARP_OPTIONS` | unset | A JSON object of default options for the DM Control MuJoCo Warp renderer, for example `{"shadows": "one"}`. The `render_options` argument overrides each key that it sets. |

## Development

| Variable | Default | Effect |
|---|---|---|
| `STABLE_ENVS_OGB_UNMEASURED` | 0 | `1` lets a Cube or Scene class without measured visual fidelity values register with placeholder values while you measure them. `OGBCube` and `OGBScene` are measured, so it doesn't change them. Leave it unset for normal use. The library reads it once, at import. |
