# Install

```bash
pip install stable-envs
pip install "stable-envs[pusht]"              # PushT simulation and CPU rendering
pip install "stable-envs[dmcontrol,ogbench]"  # add family extras as needed
```

Then run `stable-envs doctor`. It checks the NVIDIA driver, torch, Warp and MuJoCo's OpenGL backends.
For each installed family it reports what works on this machine and what to change. See
[System setup](system-setup.md) for driver and OpenGL setup.

There is no `[all]` extra. A missing extra disables only that family. `snv.import_errors()` lists the
modules that did not load.

```python
import stable_envs as snv
snv.import_errors()     # {module: "ModuleNotFoundError: ..."}
snv.list_families()     # family ids that registered
```

Supported CPython versions are 3.10, 3.11 and 3.12.

CUDA is required for the 3-D families and most MuJoCo families. PushT, classic control, navigation
and several 2-D families run on CPU. Device support per family: [Limitations](concepts/limitations.md).

For an editable install from a clone, see
[Contributing](https://github.com/galilai-group/stable-envs/blob/main/CONTRIBUTING.md).

## Extras

Some extras are needed to construct and render a family. Others install only the official reference
package, which tests use for comparison.

| extra | needed for | adds |
|---|---|---|
| *(none)* | classic control, navigation | nothing |
| `pusht` | PushT, PushTDiscrete | Pymunk 7.2.0, pygame, OpenCV, Shapely |
| `pusht-reference` | reference checks only | gym-pusht |
| `ogbench` | OGBench mazes | `ogbench`, MuJoCo |
| `ogbench-manipulation` / `-gpu` | OGBCube, OGBScene | MuJoCo 3.12.0, OGBench 1.2.1, dm_control; `-gpu` adds MuJoCo Warp 3.12.0 and Warp 1.17.0 |
| `dmcontrol` | DM Control | `dm_control`, MuJoCo |
| `dmcontrol-gpu` | DM Control on CUDA with TrackCom floor reflections | MuJoCo and MuJoCo Warp 3.12 |
| `robotics` | Fetch | gymnasium-robotics, MuJoCo < 3.12, OpenCV |
| `mjwarp` | Fetch on CUDA | MuJoCo Warp |
| `rocket` | PyFlytRocketLanding | PyFlyt 0.29.0, pybullet 3.2.7, Pillow, SciPy |
| `docs` / `dev` / `examples` / `hf` | not needed to render | docs, tests, example helpers, HuggingFace |

With `dmcontrol-gpu` on MuJoCo 3.11, TrackCom domains render without the floor reflection.

The `rocket` extra resolves numpy < 2 because of PyFlyt's metadata.

PushT renders with pygame on CPU. `device="cuda"` places the output tensors on the GPU. The drawing
still runs on CPU.

## Reference profiles

Extras declare compatible version ranges. Exact parity with an upstream renderer is checked only under
that family's reference profile. These are the `pin-<profile>` dependency groups in `pyproject.toml`:
`pin-classic`, `pin-navigation`, `pin-pusht`, `pin-dmc`, `pin-fetch`, `pin-ogbench` and `pin-rocket`.
They are for development, cannot be combined, need Python 3.12, and resolve in the single `uv.lock`:

```bash
uv sync --python 3.12 --extra dmcontrol --group pin-dmc --group test
```

`uvx nox -s "test(dmc)"` builds the same environment in `.venvs/dmc-py3.12` and runs the family's tests.

## Extra conflicts

Fetch (`robotics`) needs `mujoco<3.12`. DM Control and OGBench accept `mujoco<3.13`. Do not install
`robotics` in the same environment as an extra that needs MuJoCo 3.12.

## nvdiffrast

nvdiffrast is not a PyPI extra. It builds a CUDA extension at install time against a local toolkit.

```bash
# nvcc must match torch's CUDA major version.
# CPATH must include torch's bundled CUDA headers (the build needs <cusparse.h>).
CUDA_HOME=... CPATH=... TORCH_CUDA_ARCH_LIST=... \
  pip install --no-build-isolation \
  "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0"
```
