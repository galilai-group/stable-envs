# System setup

Besides the Python extras in [Install](install.md), the CUDA paths need an NVIDIA driver and the
MuJoCo families need an OpenGL backend. Check both with:

```bash
stable-envs doctor          # or: python -m stable_envs doctor
stable-envs doctor --json   # the same report for scripts; exit code 1 when an installed family cannot render
```

The command reports the driver, torch and Warp. It tries each MuJoCo OpenGL backend in a separate
process. It lists every family, whether its CPU and GPU render paths work here, and the one change
that fixes them.

## What each family needs

| family | CPU render | CUDA render |
|---|---|---|
| classic control, navigation, PushT | nothing beyond the extra | a CUDA build of torch |
| Craftax | nothing beyond the extra | CUDA builds of torch and JAX (`craftax-gpu`) |
| DM Control, OGBench mazes | MuJoCo OpenGL | MuJoCo OpenGL and CUDA torch. TrackCom domains also need MuJoCo Warp (`dmcontrol-gpu`) |
| Fetch | nothing beyond the extra (MuJoCo OpenGL for `backend="opengl"`) | MuJoCo OpenGL, CUDA torch and MuJoCo Warp (`robotics` + `mjwarp`) |
| OGBench Cube / Scene | MuJoCo OpenGL | NVIDIA OpenGL and CUDA torch. `supersample` 2 or more also needs MuJoCo Warp (`ogbench-manipulation-gpu`) |
| Rocket | nothing beyond the extra | CUDA torch, Triton and [nvdiffrast](install.md#nvdiffrast) |

The MuJoCo families draw each scene's static background once per appearance with MuJoCo's own
OpenGL renderer. They draw the moving bodies on top, on the CPU or on CUDA. So they need an OpenGL
backend on GPU machines too. MuJoCo picks the backend from `MUJOCO_GL`.

When `MUJOCO_GL` is unset, stable-envs picks the backend:

- On Linux, if `PYOPENGL_PLATFORM` names a backend (`egl` or `osmesa`), stable-envs uses that backend
  so the two agree.
- Otherwise on Linux, it uses `egl`.
- On macOS, it uses `cgl`.

## GPU machines: the driver

torch and Warp wheels are built against a CUDA toolkit. The driver must support at least that
CUDA version. If it does not, `torch.cuda.is_available()` is False, but Warp falls back to CPU
arrays with only a printed warning.

Warp 1.18 is built with CUDA 13 and needs driver 580 or newer. On an older driver, install
`warp-lang==1.17.0`. The project certifies the GPU extras with that version.

The GPU render paths refuse to run on that fallback and say why. `stable-envs doctor` prints the
driver's and Warp's CUDA versions side by side.

## Desktop Linux with an NVIDIA GPU

1. Install the NVIDIA driver from your distribution. The driver package also installs its OpenGL and
   EGL libraries.
2. Install a CUDA build of torch that matches the driver.

The default `MUJOCO_GL=egl` works. `MUJOCO_GL=glfw` also works in a desktop session.

## Headless Linux with an NVIDIA GPU (servers, clusters)

Keep `MUJOCO_GL=egl`. Make sure that EGL can reach the GPU:

```bash
sudo apt install libegl1          # the GLVND EGL loader
```

The driver supplies `libEGL_nvidia.so.0`. Driver installs made with `--no-opengl-files` omit it.
Some minimal cluster images also omit it. In that case, reinstall the driver with its GL libraries
or use OSMesa below.

`MUJOCO_EGL_DEVICE_ID` indexes every GPU on the machine. `CUDA_VISIBLE_DEVICES` does not change
this index. Leave it unset unless you need a specific physical GPU.

## Headless Linux without a GPU

Use Mesa's software renderer through OSMesa:

```bash
sudo apt install libosmesa6 libgl1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
```

`MUJOCO_GL` and `PYOPENGL_PLATFORM` must agree: MuJoCo refuses `MUJOCO_GL=egl` with
`PYOPENGL_PLATFORM=osmesa` and the reverse. `MUJOCO_GL=glfw` needs a display. Without a display,
run under `xvfb-run` or use OSMesa.

## Docker

With a GPU:

1. Start from an `nvidia/cuda` image.
2. Install `libegl1`.
3. Give the container the graphics capability. The NVIDIA container toolkit then mounts the
   driver's EGL library:

```bash
docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics ...
```

The default capabilities (`compute,utility`) give CUDA but no EGL. Without a GPU, follow the
headless no-GPU section inside the image.

## macOS

There is no CUDA on macOS, so every family runs its CPU path. MuJoCo uses its native `cgl`
backend with no setup.
