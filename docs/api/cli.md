# Command line

The package installs one command, `stable-envs`, with the subcommands `doctor` and `tune`. `python -m stable_envs` runs the same command.

## `stable-envs doctor`

```bash
stable-envs doctor          # text report
stable-envs doctor --json   # the same report as JSON
```

`doctor` checks which environment families work on this machine and what to fix for the rest.

| option | effect |
|---|---|
| `--json` | Print the report as JSON instead of text. |

The exit code is 0 when every installed family works, and 1 when any installed family fails.

### What it checks

- System: the Python version, the NVIDIA driver and the CUDA version it supports, torch and its
  CUDA build, and Warp.
- MuJoCo OpenGL: whether `MUJOCO_GL`, `PYOPENGL_PLATFORM`, `DISPLAY` and `MUJOCO_EGL_DEVICE_ID`
  agree, then each OpenGL backend. The backends are `egl`, `osmesa` and `glfw` on Linux, and `cgl` and
  `glfw` on macOS. `doctor` tries each backend in a separate process with a 60-second timeout, because
  creating a context can hang or crash. `doctor` skips the OpenGL checks when MuJoCo is not installed.
- Families: for each family, whether its extra is installed, whether its modules import, and
  whether its CPU and GPU render paths work. `doctor` skips a family that is not installed.

Each check has a status: `ok`, `warn`, `fail` or `skip`. A `warn` or `fail` line also prints the fix,
such as the extra to install or the environment variable to set.

### JSON output

`--json` prints one object:

| key | content |
|---|---|
| `ok` | `false` when any installed family fails. |
| `system` | System checks. Each has `name`, `status`, `detail` and `fix`. |
| `gl` | MuJoCo OpenGL checks, in the same form. |
| `families` | One entry per family, with `name`, `status`, `cpu`, `gpu`, `detail` and `fixes`. |

Paste this output into bug reports.

See [System setup](../system-setup.md) for what each family needs and how to fix a failing check.

The report also has a `tuning` line: how many render paths have values from `stable-envs tune` on
this machine. The other paths use the derived defaults.

When torch has no GPU kernels for the GPU's compute capability, the `torch` line warns and lists the
architectures the build has. CUDA calls then fail. A build for an older architecture runs on a newer
GPU of the same major version. PTX (`compute_XY`) runs on any equal or newer GPU.

## `stable-envs tune`

```bash
stable-envs tune DMCCheetahRun                 # tune one environment's render path
stable-envs tune TwoRoom --batch 1024 --device cuda
stable-envs tune --show                        # list cached values
stable-envs tune --clear                       # delete cached values
stable-envs tune --all --device cpu            # one environment per render path
```

`tune` times the tuning knobs one environment reads and saves the fastest values for its render path,
frame size and batch range on this device. Later runs of every environment on the same path use them.
See [Performance tuning](../performance-tuning.md) for the knobs and how the command chooses values.

| option | effect |
|---|---|
| `ENV` | Environment class name, such as `DMCCheetahRun`. |
| `--batch N` | Number of environments to time (default 64). The command saves values for the power-of-two range that holds `N`. |
| `--size S` | Frame size in pixels (default: the environment's). |
| `--device D` | `cpu`, `cuda` or `cuda:N` (default: the configured default device). |
| `--knob NAME` | Sweep only this knob. Repeat for more. |
| `--repeats R` | Timed calls per value after one warm-up call (default 5). The median counts. |
| `--show` | List the cached values of this machine's CPU and GPU, and whether they are current. |
| `--clear` | Delete the cached values of this machine's CPU and GPU. |
| `--all` | Tune one representative environment per render path. Prints the plan and an estimated time, then asks before starting. |
| `--yes` | With `--all`, start without asking. |

The command prints one row per knob (default, tuned value, speedup) and the paths that it saved the
values for. It exits with 1 when you do not approve the `--all` plan. It fails when a knob value
changes frames or State. Report this failure as a bug.
