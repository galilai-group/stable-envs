# Command line

The package installs one command, `stable-envs`. `python -m stable_envs` runs the same command.

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
  agree, then each OpenGL backend (`egl`, `osmesa` and `glfw` on Linux; `cgl` and `glfw` on macOS).
  Each backend is tried in a separate process with a 60-second timeout, because creating a context
  can hang or crash. The OpenGL checks are skipped when MuJoCo is not installed.
- Families: for each family, whether its extra is installed, whether its modules import, and
  whether its CPU and GPU render paths work. A family that is not installed is skipped.

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
