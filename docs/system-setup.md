# System setup

The Python extras in [Install](install.md) are half of a working setup. The other half is the
machine: an NVIDIA driver for the CUDA paths, and an OpenGL backend for the MuJoCo families.
Check both with

```bash
stable-envs doctor          # or: python -m stable_envs doctor
stable-envs doctor --json   # the same report for scripts; exit code 1 when an installed family cannot render
```

It reports the driver, torch and Warp, tries each MuJoCo OpenGL backend in a separate process, and
lists every family with whether its CPU and GPU render paths work here and the one change that
fixes them.

## What each family needs

| family | CPU render | CUDA render |
|---|---|---|
| classic control, navigation, PushT | nothing beyond the extra | a CUDA build of torch |
| Craftax | nothing beyond the extra | CUDA builds of torch and JAX (`craftax-gpu`) |
| DM Control, OGBench mazes | MuJoCo OpenGL | MuJoCo OpenGL and CUDA torch; TrackCom domains also MuJoCo Warp (`dmcontrol-gpu`) |
| Fetch | MuJoCo OpenGL (`backend="opengl"`) | MuJoCo OpenGL, CUDA torch and MuJoCo Warp (`robotics` + `mjwarp`) |
| OGBench Cube / Scene | MuJoCo OpenGL | MuJoCo OpenGL, CUDA torch and MuJoCo Warp (`ogbench-manipulation-gpu`) |
| Rocket | nothing beyond the extra | CUDA torch, Triton and [nvdiffrast](install.md#nvdiffrast) |

The MuJoCo families draw each scene's static background once per appearance with MuJoCo's own
OpenGL renderer and draw the moving bodies on top, on the CPU or on CUDA. So they need an OpenGL
backend on GPU machines too. MuJoCo picks it from `MUJOCO_GL`. When that is unset on Linux,
stable-envs uses the backend `PYOPENGL_PLATFORM` names (`egl` or `osmesa`) so the two agree, and
`egl` otherwise; on macOS it uses `cgl`.

## GPU machines: the driver

torch and Warp wheels are built against a CUDA toolkit, and the driver must support at least that
CUDA version. torch says so (`torch.cuda.is_available()` is False). **Warp does not**: a Warp
build whose toolkit is newer than the driver falls back to CPU arrays with only a printed warning.
Warp 1.18 is built with CUDA 13 and needs driver 580 or newer; on an older driver install
`warp-lang==1.17.0`, the version the GPU extras are certified with. The GPU render paths refuse to
run on that fallback and say why; `stable-envs doctor` shows the driver's and Warp's CUDA versions
side by side.

## Desktop Linux with an NVIDIA GPU

Install the NVIDIA driver from your distribution, which also installs its OpenGL and EGL
libraries, then a CUDA build of torch that matches it. The default `MUJOCO_GL=egl` works;
`MUJOCO_GL=glfw` also works in a desktop session.

## Headless Linux with an NVIDIA GPU (servers, clusters)

Keep `MUJOCO_GL=egl` and make sure EGL can reach the GPU:

```bash
sudo apt install libegl1          # the GLVND EGL loader
```

The driver supplies `libEGL_nvidia.so.0`. Driver installs made with `--no-opengl-files`, and some
minimal cluster images, leave it out; then reinstall the driver with its GL libraries or use
OSMesa below. `MUJOCO_EGL_DEVICE_ID` indexes every GPU on the machine, not the ones in
`CUDA_VISIBLE_DEVICES`; leave it unset unless you need a specific physical GPU.

## Headless Linux without a GPU

Use Mesa's software renderer through OSMesa:

```bash
sudo apt install libosmesa6 libgl1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
```

`MUJOCO_GL` and `PYOPENGL_PLATFORM` must agree: MuJoCo refuses `MUJOCO_GL=egl` with
`PYOPENGL_PLATFORM=osmesa` and the reverse. `MUJOCO_GL=glfw` needs a display; without one run
under `xvfb-run` or switch to OSMesa.

## Docker

With a GPU, start from an `nvidia/cuda` image, install `libegl1`, and give the container the
graphics capability so the NVIDIA container toolkit mounts the driver's EGL library:

```bash
docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics ...
```

The default capabilities (`compute,utility`) give CUDA but no EGL. Without a GPU, follow the
headless no-GPU section inside the image.

## macOS

There is no CUDA on macOS, so every family runs its CPU path. MuJoCo uses its native `cgl`
backend with no setup; Fetch's CPU path is `backend="opengl"`.
