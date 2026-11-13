# Install

Install stable-envs with uv or pip, then add one extra for each environment [family](terms.md#family) you want to run. The GPU paths also need an NVIDIA driver, and most MuJoCo families need an OpenGL backend.

=== "uv"

    ```bash
    uv add stable-envs
    uv add "stable-envs[pusht]"              # PushT simulation and CPU rendering
    uv add "stable-envs[dmcontrol,ogbench]"  # add family extras as needed
    ```

=== "pip"

    ```bash
    pip install stable-envs
    pip install "stable-envs[pusht]"              # PushT simulation and CPU rendering
    pip install "stable-envs[dmcontrol,ogbench]"  # add family extras as needed
    ```

stable-envs supports CPython 3.10, 3.11 and 3.12. It runs on Linux and macOS, and Windows isn't supported. After you install, run [`stable-envs doctor`](api/cli.md#stable-envs-doctor) to see what works on this machine and what to change.

## Missing extras

A missing extra affects only its own family. The core install registers 52 classes, and 6 of them run with nothing else installed: CartPole, Acrobot, TwoRoom, TwoRoomTarget, Piecewise and PiecewiseZones.

- The 27 OGBench, 12 Fetch and 2 PushT classes and `PyFlytRocketLanding` register without their extras and raise `ModuleNotFoundError` when you construct one.
- SimplePointMaze needs the `navigation` extra, and MountainCar, MountainCarContinuous and Pendulum need the `classic` extra for pygame.
- DM Control, Craftax, Miniworld, MS-Human and Roadster register only once you install their extra.

`snv.import_errors()` says what's missing. With only the core installed:

```python
import stable_envs as snv

print(snv.list_families())
# ['classic_control', 'fetch', 'navigation', 'ogbench', 'pusht', 'pyflyt']
for module, error in snv.import_errors().items():
    print(module, "->", error)
# craftax -> ImportError: Craftax needs the craftax extra: pip install 'stable-envs[craftax]'
# dmcontrol -> ModuleNotFoundError: No module named 'dm_control'
# miniworld -> ImportError: Miniworld needs the miniworld extra: pip install 'stable-envs[miniworld]'
# mshuman -> ModuleNotFoundError: No module named 'mujoco'
# roadster -> ImportError: Roadster needs the roadster extra: pip install 'stable-envs[roadster]'
```

The MuJoCo extras need conflicting MuJoCo versions, and there's no `[all]` extra. Fetch (`robotics`) needs `mujoco<3.12`, while `dmcontrol`, `mshuman`, `roadster` and `ogbench-manipulation` need 3.17. Install Fetch in its own environment; `ogbench` works with either. No single environment can run every class.

## What each family needs

Every family runs on CPU, so you can try any of them without a GPU. For CUDA, some families need a `-gpu` extra or extra packages.

| Family | Extra and what it adds | CPU | CUDA |
|---|---|---|---|
| classic control | none. `classic` adds pygame for MountainCar, MountainCarContinuous and Pendulum, and Numba for the compiled CPU renderer | nothing else | a CUDA build of torch |
| navigation | none. `navigation` adds Matplotlib and Numba for SimplePointMaze (its exact `info.goal` image), Piecewise's `rendering.render_bias_field` and the compiled CPU renderer | nothing else | a CUDA build of torch. TwoRoom also needs `navigation-gpu`, JAX's CUDA 12 build (Linux) |
| PushT | `pusht`: Pymunk 7.2.0, pygame, OpenCV, Shapely, Numba | nothing else | CUDA torch. Triton, which comes with CUDA torch on Linux, steps the physics and draws the frames on the GPU. `pusht-gpu` adds JAX's CUDA 12 build (Linux) for `simulation_backend="jax"` |
| Craftax | `craftax`: Craftax, JAX | nothing else | CUDA torch and `craftax-gpu`, which takes JAX's CUDA 12 build |
| PyFlyt rocket | `rocket`: PyFlyt 0.29.0, pybullet 3.2.7, Pillow, SciPy, Numba for the default physics and CPU renderer | nothing else | CUDA torch, Triton, [nvdiffrast](#nvdiffrast) and a CUDA toolkit. Numba needs the toolkit's libNVVM to compile the CUDA physics |
| DM Control | `dmcontrol`: `dm_control` 1.0.50 or later, MuJoCo 3.17, Numba for the compiled CPU renderer | MuJoCo OpenGL | MuJoCo OpenGL, CUDA torch and `dmcontrol-gpu`: MuJoCo, MuJoCo Warp and MJX 3.17, Warp, JAX's CUDA 12 build (Linux). MuJoCo Warp steps the default CUDA physics and draws the TrackCom floor reflections |
| OGBench mazes | `ogbench`: OGBench 1.2.1 or a later 1.2, MuJoCo 3.11 to 3.17, Numba for the compiled CPU light pass of `PointMazeLargeNavigate` | MuJoCo OpenGL | MuJoCo OpenGL, CUDA torch and `mjwarp` for the default MuJoCo Warp physics |
| OGBench Cube and Scene | `ogbench-manipulation`: OGBench 1.2.1, MuJoCo 3.17.0, `dm_control` 1.0.50 or later, Numba for the default MuJoCo rasterizer | nothing else; MuJoCo OpenGL for `backend="opengl"` | NVIDIA OpenGL and CUDA torch. `ogbench-manipulation-gpu` adds MuJoCo Warp 3.17.0 and Warp 1.17.0 for the default MuJoCo Warp physics and for `supersample` 2 or more |
| Fetch | `robotics`: Gymnasium Robotics 1.4.2 or a later 1.4, MuJoCo 3.11, OpenCV, Numba for the compiled CPU renderer | nothing else; MuJoCo OpenGL for `backend="opengl"` | MuJoCo OpenGL, CUDA torch and `mjwarp`: MuJoCo, MuJoCo Warp, Warp |
| Roadster | `roadster`: MuJoCo 3.17, Pillow | MuJoCo OpenGL | MuJoCo OpenGL, CUDA torch and `roadster-gpu`, which adds MuJoCo Warp 3.17 and Warp for the default MuJoCo Warp physics. Frames are drawn with MuJoCo OpenGL and copied to the GPU |
| MS-Human | `mshuman`: MuJoCo and MuJoCo Warp 3.17, Warp, filelock | MuJoCo OpenGL | MuJoCo OpenGL and CUDA torch. MuJoCo Warp steps the physics on the GPU, and frames are drawn on the host and copied over |
| Miniworld | `miniworld`: the [hlillemark/Miniworld](#miniworld) fork, pyglet 1.5, Numba for the CPU step | pyglet OpenGL (EGL on Linux) | the same, plus CUDA torch. Triton steps the physics on the GPU, and frames are drawn on the CPU and copied to the GPU |

The `pusht-reference` extra (gym-pusht) installs only the [upstream library](terms.md#upstream-library) that the PushT tests compare against. The `docs`, `dev`, `examples` and `hf` extras aren't needed to render: they add the docs build, tests, example helpers and Hugging Face `datasets`.

- MountainCar, MountainCarContinuous and Pendulum draw part of their scene with pygame, which the `classic` extra installs; the `pusht` extra also installs it.
- On CUDA, TrackCom domains draw the floor reflection only with MuJoCo Warp 3.17, which `dmcontrol-gpu` installs. With another MuJoCo Warp, they render without the reflection and log a warning.
- The `rocket` extra resolves numpy < 2, the limit PyFlyt 0.29.0 declares in its metadata.

The first construction, step or render of many classes compiles kernels: Numba on CPU, and Triton, Warp or JAX on CUDA. This can take from a few seconds to a few minutes for each class, and nothing shows progress during this time. Most compiled kernels stay in a cache on disk, so the next process starts faster. Do some warm-up calls before you measure speed.

The MuJoCo families draw some or all of their frames with MuJoCo's own OpenGL renderer, so they need an OpenGL backend on GPU machines too. MuJoCo picks the backend from `MUJOCO_GL`. If it's unset, stable-envs picks one: on Linux the backend that `PYOPENGL_PLATFORM` names (`egl` or `osmesa`), otherwise `egl` on Linux and `cgl` on macOS.

On Linux, `import mujoco` loads the libraries of that backend. Fetch, `OGBCube` and `OGBScene` don't draw with OpenGL on CPU, but they import MuJoCo, so they also need the packages of [Headless Linux with an NVIDIA GPU](#headless-linux-with-an-nvidia-gpu) or [Headless Linux without a GPU](#headless-linux-without-a-gpu).

Miniworld draws through pyglet, which runs headless on EGL on Linux unless `PYGLET_HEADLESS` is set. It needs an EGL driver, and the EGL steps below apply to it as well. Without a GPU, Mesa's llvmpipe works. pyglet also loads libGL and GLU, so on Debian and Ubuntu install these packages, with or without a GPU:

```bash
sudo apt install libegl1 libgl1 libglu1-mesa
```

## Miniworld

`MovingBlockWorld` lives in the [hlillemark/Miniworld](https://github.com/hlillemark/Miniworld) fork. Install the fork from GitHub before the extra:

=== "uv"

    ```bash
    uv add "miniworld @ git+https://github.com/hlillemark/Miniworld@bc7972050155b67e242c07bfdccfe1eeddbf9882"
    uv add "stable-envs[miniworld]"
    ```

=== "pip"

    ```bash
    pip install "miniworld @ git+https://github.com/hlillemark/Miniworld@bc7972050155b67e242c07bfdccfe1eeddbf9882"
    pip install "stable-envs[miniworld]"
    ```

## nvdiffrast

PyFlytRocketLanding draws on CUDA with nvdiffrast, which you install from source. It builds a CUDA extension against your local toolkit, and PyPI rejects direct-URL requirements, so no extra can install it:

```bash
# nvcc must match torch's CUDA major version.
# CPATH must include torch's bundled CUDA headers (the build needs <cusparse.h>).
CUDA_HOME=... CPATH=... TORCH_CUDA_ARCH_LIST=... \
  pip install --no-build-isolation \
  "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0"
```

For example, the PyPI torch wheel for CUDA 13.2 on an L40S (compute capability 8.9) on Ubuntu, with NVIDIA's CUDA apt repository:

```bash
sudo apt install cuda-nvcc-13-2 cuda-cudart-dev-13-2 build-essential
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
CUDA_HOME=/usr/local/cuda-13.2 PATH=/usr/local/cuda-13.2/bin:$PATH \
  CPATH=$SITE/nvidia/cu13/include TORCH_CUDA_ARCH_LIST=8.9 \
  pip install --no-build-isolation \
  "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0"
```

`python -c "import torch; print(torch.version.cuda)"` gives torch's CUDA version, and `nvidia-smi --query-gpu=compute_cap --format=csv` gives the compute capability.

## The NVIDIA driver

torch and Warp wheels are built against a CUDA toolkit, and the driver has to support at least that CUDA version. With an older driver, `torch.cuda.is_available()` returns `False`, while Warp falls back to CPU arrays with only a printed warning. The GPU render paths stop on that fallback with an error that says why, and `stable-envs doctor` prints the driver's and Warp's CUDA versions side by side.

Warp 1.18 is built with CUDA 13 and needs driver 580 or newer. On an older driver, install `warp-lang==1.17.0`, the version the project tests the GPU extras with.

## Desktop Linux with an NVIDIA GPU

Install the NVIDIA driver from your distribution, which also installs its OpenGL and EGL libraries, then a CUDA build of torch that matches the driver. The default `MUJOCO_GL=egl` works, and so does `MUJOCO_GL=glfw` in a desktop session.

## Headless Linux with an NVIDIA GPU

On servers and clusters, keep `MUJOCO_GL=egl` and make sure EGL can reach the GPU:

```bash
sudo apt install libegl1 libgl1   # the GLVND EGL loader and libGL
```

PyOpenGL loads `libGL.so.1` (or `libOpenGL.so.0`) as well as `libEGL.so.1`. Without them, the MuJoCo families don't import, and `import stable_envs` logs a warning for each of them.

The driver supplies `libEGL_nvidia.so.0`. Driver installs made with `--no-opengl-files` and some minimal cluster images leave it out. If yours is missing it, reinstall the driver with its GL libraries or use OSMesa as in the next section.

`MUJOCO_EGL_DEVICE_ID` counts every GPU on the machine, whatever `CUDA_VISIBLE_DEVICES` says. Leave it unset unless you need a specific physical GPU.

## Headless Linux without a GPU

Use Mesa's software renderer through OSMesa:

```bash
sudo apt install libosmesa6 libgl1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
```

`MUJOCO_GL` and `PYOPENGL_PLATFORM` must agree: MuJoCo refuses `MUJOCO_GL=egl` with `PYOPENGL_PLATFORM=osmesa` and the reverse. `MUJOCO_GL=glfw` needs a display; without one, run under `xvfb-run` or use OSMesa.

## Docker

With a GPU, start from an `nvidia/cuda` image, install `libegl1` and `libgl1`, and give the container the graphics capability so the NVIDIA container toolkit mounts the driver's EGL library:

```bash
docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics ...
```

The default capabilities (`compute,utility`) give you CUDA but no EGL. Without a GPU, follow [Headless Linux without a GPU](#headless-linux-without-a-gpu) inside the image.

## macOS

On macOS every family runs its CPU path, and MuJoCo uses its native `cgl` backend with no setup.
