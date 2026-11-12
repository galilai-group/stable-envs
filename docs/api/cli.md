# CLI

The package installs one command, `stable-envs`, with `doctor` to check which families work on this machine and `tune` to measure the fastest tuning knobs. `python -m stable_envs` runs the same command.

## `stable-envs doctor`

```bash
stable-envs doctor          # text report
stable-envs doctor --json   # the same report as JSON
```

`doctor` checks which environment families work on this machine and what to fix for the rest. This is its report on a macOS laptop with only the core install:

```text
stable-envs 0.1.0 doctor

System
  ok    python                 3.11.14 on macOS-15.6.1-arm64-arm-64bit
  skip  nvidia driver          macOS has no CUDA; GPU render paths are unavailable
  ok    torch                  2.15.0, CPU-only build
  skip  warp                   not installed
  ok    tuning                 0 of 8 render paths tuned on this cpu device; others use derived defaults
  ok    STABLE_ENVS_*          none set

MuJoCo OpenGL
  skip  gl                     mujoco not installed; only the MuJoCo families need OpenGL

Families                 cpu   gpu
  ok    classic_control   ok    skip
  skip  navigation        not installed
                          install: pip install "stable-envs[navigation]"
  skip  pusht             not installed
                          install: pip install "stable-envs[pusht]"
  ...
  skip  roadster          not installed
                          install: pip install "stable-envs[roadster]"

All installed families can render here.
```

| Option | Effect |
|---|---|
| `--json` | Print the report as JSON instead of text. |

The exit code is 0 when every installed family works, and 1 when any installed family fails.

### What it checks

- System: the Python version, the NVIDIA driver and the CUDA version it supports, torch and its CUDA build, Warp, how many render paths have values from `stable-envs tune` on this machine, and which `STABLE_ENVS_*` variables are set.
- MuJoCo OpenGL: whether `MUJOCO_GL`, `PYOPENGL_PLATFORM`, `DISPLAY` and `MUJOCO_EGL_DEVICE_ID` agree, then each OpenGL backend. The backends are `egl`, `osmesa` and `glfw` on Linux, and `cgl` and `glfw` on macOS. Creating a context can hang or crash, and `doctor` tries each backend in a separate process with a 60-second timeout. It skips the OpenGL checks when MuJoCo is missing.
- Families: for each family, whether its extra is installed, whether its modules import, and whether its CPU and GPU render paths work. `doctor` skips families whose extra is missing.

Each check has a status: `ok`, `warn`, `fail` or `skip`. A `warn` or `fail` line also prints the fix, such as the extra to install or the environment variable to set.

When torch has no GPU kernels for the GPU's compute capability, CUDA calls would fail. The `torch` line then warns and lists the architectures the build has. A build for an older architecture runs on a newer GPU of the same major version, and PTX (`compute_XY`) runs on any equal or newer GPU.

### JSON output

`--json` prints one object. Paste it into bug reports.

| Key | Content |
|---|---|
| `ok` | `false` when any installed family fails. |
| `system` | System checks. Each has `name`, `status`, `detail` and `fix`. |
| `gl` | MuJoCo OpenGL checks, in the same form. |
| `families` | One entry per family, with `name`, `status`, `cpu`, `gpu`, `detail` and `fixes`. |

```text
{
  "ok": true,
  "system": [
    {
      "name": "python",
      "status": "ok",
      "detail": "3.11.14 on macOS-15.6.1-arm64-arm-64bit",
      "fix": ""
    },
    ...
```

To fix a failing check, see [what each family needs](../install.md#what-each-family-needs).

## `stable-envs tune`

```bash
stable-envs tune DMCCheetahRun                 # tune one environment's render path
stable-envs tune TwoRoom --batch 1024 --device cpu
stable-envs tune --show                        # list cached values
stable-envs tune --clear                       # delete cached values
stable-envs tune --all --device cpu            # one environment per render path
```

`tune` times the tuning knobs one environment reads and saves the fastest values for its render path, frame size and batch range on this device. Later runs of every environment on the same path use them. [Tuning](tuning.md#knobs) lists the knobs.

It tunes one knob at a time, for each knob with a measured unit that the environment reads on its device; `worker_min_lanes`, `render_worker_min_rows`, `ogb_gl_workers` and `ogb_context_cache_bytes` aren't tuned. It times the default and a few powers of two around it, and keeps the best value of each knob while it tunes the next. Each value gets one warm-up call and five timed calls, with CUDA synchronized, and the median counts. It skips values that give the same work split as another value, or that would use more than half the free GPU memory. A run takes about a minute. It stops if any value changes a frame or a state, which is a bug.

The cache is `~/.cache/stable-envs/tuning/<device name>.json`, or under `$XDG_CACHE_HOME` when that's set. Batch ranges are powers of two: 64 to 127, 128 to 255, and so on. A file applies only on the same device, driver, host CPU and stable-envs version. Tune again after an upgrade. `stable-envs tune --all` tunes one environment for each render path the device can use, and `snv.tune(env)` tunes a constructed environment from Python, then restores its state.

This is `stable-envs tune TwoRoom --batch 1024 --device cpu` followed by `stable-envs tune --show` on a macOS laptop. Your numbers will differ:

```text
  knob                        default        tuned  speedup
  nav_render_chunk                256         1024    1.18x
saved for: navigation, 224 px, batch 1024-2047 (also covers: Piecewise, PiecewiseZones, SimplePointMaze, TwoRoomTarget) (~/.cache/stable-envs/tuning/arm.json)
~/.cache/stable-envs/tuning/arm.json: arm, current
  navigation|224|1024-2047             nav_render_chunk=1024
```

| Option | Effect |
|---|---|
| `ENV` | Environment class name, such as `DMCCheetahRun`. |
| `--batch N` | Number of environments to time (default 64). The command saves values for the power-of-two range that holds `N`. |
| `--size S` | Frame size in pixels (default: the environment's). |
| `--device D` | `cpu`, `cuda` or `cuda:N` (default: the configured default device). |
| `--knob NAME` | Sweep only this knob. Repeat for more. |
| `--repeats R` | Timed calls per value after one warm-up call (default 5). The median counts. |
| `--show` | List the cached values of this machine's CPU and GPU, and whether they're current. |
| `--clear` | Delete the cached values of this machine's CPU and GPU. |
| `--all` | Tune one representative environment per render path. Prints the plan and an estimated time, then asks before starting. |
| `--yes` | With `--all`, start without asking. |

The command prints one row per knob (default, tuned value, speedup) and the paths it saved the values for. It exits with 1 when you decline the `--all` plan. It fails when a knob value changes [frames](../terms.md#frame) or [state](../terms.md#state). That is a bug; please report it.
