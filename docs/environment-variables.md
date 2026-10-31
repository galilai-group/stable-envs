# Environment variables

stable-envs reads the variables below. Each one has a default, so you do not need to set any of them. A value that does
not parse raises `ValueError` with the variable's name when the library reads it. `stable-envs doctor`
lists the variables that are set and flags names that stable-envs does not know, such as a typo.

Integer values take an optional `K`, `M`, `G` or `T` suffix (powers of 1024). `auto` or an empty value
uses the default. Flags take `1` or `0`.

[System setup](system-setup.md) describes variables from other libraries, such as `MUJOCO_GL`,
`PYOPENGL_PLATFORM` and `CUDA_VISIBLE_DEVICES`.

## Tuning knobs

Each tuning knob has a variable. The variable applies to the whole process. It has lower precedence
than `snv.tuning.set` and higher precedence than the values that `stable-envs tune` measured. See [Performance tuning](performance-tuning.md) for what each
knob does and its default.

| variable | type |
|---|---|
| `STABLE_ENVS_WORKERS` | int (0 or more) |
| `STABLE_ENVS_WORKER_MIN_LANES` | int |
| `STABLE_ENVS_FETCH_THREADS` | int |
| `STABLE_ENVS_CPU3D_CHUNK` | int |
| `STABLE_ENVS_DMC_CPU_RAY_BUDGET` | int, power of two |
| `STABLE_ENVS_DMC_RAY_BUDGET` | int, power of two |
| `STABLE_ENVS_DMC_LAUNCH_RAYS` | int |
| `STABLE_ENVS_DMC_MIN_WORLDS` | int, power of two |
| `STABLE_ENVS_FETCH_MJW_WORLDS` | int, power of two |
| `STABLE_ENVS_OGB_MJW_WORLDS` | int, power of two |
| `STABLE_ENVS_OGB_CONTEXT_CACHE_BYTES` | bytes |
| `STABLE_ENVS_OGB_GL_WORKERS` | int (0 or more) |
| `STABLE_ENVS_RENDER_WORKERS` | int (0 or more) |
| `STABLE_ENVS_RENDER_WORKER_MIN_ROWS` | int |
| `STABLE_ENVS_NVD_BUDGET_BYTES` | bytes |
| `STABLE_ENVS_NAV_RENDER_CHUNK` | int |
| `STABLE_ENVS_NAVDOT_BLOCK` | int, power of two |
| `STABLE_ENVS_NAVDOT_WARPS` | 1, 2, 4, 8, 16 or 32 |
| `STABLE_ENVS_RASTER2D_BLOCK` | int, power of two |
| `STABLE_ENVS_RASTER2D_CULL_WARPS` | 1, 2, 4, 8, 16 or 32 |
| `STABLE_ENVS_GFXDRAW_WARPS` | 1, 2, 4, 8, 16 or 32 |
| `STABLE_ENVS_RAYCAST_BLOCK` | int, power of two |

## Tuning behaviour

| variable | default | effect |
|---|---|---|
| `STABLE_ENVS_TUNING_HINT` | 1 | Set to `0` to hide the one-time log line that suggests `stable-envs tune`. |
| `STABLE_ENVS_AUTOTUNE` | 0 | `1` tunes an untuned render path the first time it renders a large batch, then caches the values. |

## Logging

| variable | default | effect |
|---|---|---|
| `STABLE_ENVS_VERBOSE` | `WARNING` | Default log level, with the same values as `snv.set(verbose=...)`. `snv.set` overrides it. |

## Environments

| variable | default | effect |
|---|---|---|
| `STABLE_ENVS_MSHUMAN_HOME` | unset | Path to an existing MS-Human-700 checkout. Unset downloads the pinned commit into the cache (`$XDG_CACHE_HOME/stable-envs/ms-human-700`). |

## Development

| variable | default | effect |
|---|---|---|
| `STABLE_ENVS_OGB_UNMEASURED` | 0 | `1` lets OGBench Cube and Scene cells without measured visual fidelity register with placeholder values. Use it while you measure these cells. Do not set it for normal use. The library reads it once, at import. |
