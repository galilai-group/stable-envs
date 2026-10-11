# Install

```bash
pip install stable-envs
pip install "stable-envs[pusht]"              # source-aligned PushT simulation and CPU rendering
pip install "stable-envs[dmcontrol,ogbench]"  # add family extras as needed
```

There is no `[all]` extra. A missing optional extra costs **that family**, not the package.
`snv.import_errors()` names the modules that did not load.

Supported CPython is **3.10, 3.11, and 3.12** — the range CI exercises. Newer versions are not
claimed until a job runs there.

```python
import stable_envs as snv
snv.import_errors()     # {module: "ModuleNotFoundError: ..."}
snv.list_families()     # stable family ids that registered
```

CUDA is required for the 3-D and most MuJoCo families. PushT, classic control, navigation and
several 2-D families run on CPU.

From a clone, use an editable install — see [Contributing](https://github.com/galilai-group/stable-envs/blob/main/CONTRIBUTING.md).

## Extras

Two kinds, and they are easy to mix up.

**Runtime** extras install what that family needs to **construct and render** in stable-envs.

**Reference** extras install the official CPU renderer / env package used for parity tests, the gallery, and certification. They are not required to draw our frames.

| extra | needed to render in stable-envs? | what the extra actually adds |
|---|---|---|
| *(none)* | classic control, navigation | core install is enough |
| `pusht` | PushT / PushTDiscrete | Pymunk 7.2.0, pygame, OpenCV and Shapely |
| `pusht-reference` | no | historical gym-pusht corpus checks, in a separate reference environment |
| `ogbench` | OGBench mazes (and manip, with CUDA) | `ogbench` + MuJoCo |
| `ogbench-manipulation` / `-gpu` | Cube / Scene | the exact MuJoCo 3.12.0 / OGBench 1.2.1 (and MuJoCo Warp 3.12.0 / Warp 1.17.0) their renderers check, plus dm_control |
| `dmcontrol` | DM Control cells | `dm_control` + MuJoCo |
| `dmcontrol-gpu` | DM Control on CUDA with TrackCom floor reflections | MuJoCo and MuJoCo Warp 3.12; on 3.11 TrackCom renders without the reflection |
| `robotics` | Fetch | gymnasium-robotics (MuJoCo pin), OpenCV for source-sized OpenGL output |
| `mjwarp` | Fetch / some visual 3-D cells | MuJoCo Warp render backend |
| `craftax` | Craftax | jax |
| `craftax-semantics` / `-gpu` | registered Craftax semantics | Craftax, JAX, Flax, Gymnax, Chex; construction checks the exact Craftax 1.5.0 / JAX 0.10.1 profile (`pin-craftax`) on Python 3.12 |
| `rocket` | PyFlyt rocket (CUDA + nvdiffrast) | PyFlyt 0.29.0 / pybullet 3.2.7 / pillow / scipy; pip resolves numpy<2 from PyFlyt's metadata |
| `docs` / `dev` / `examples` / `hf` | no | documentation, tests, gallery helpers, HuggingFace |

`pip install "stable-envs[pusht]"` is required to construct public `PushT` and
`PushTDiscrete`. Both now use the source-aligned task and complete replay State.
Their current public renderer is CPU Pygame. CUDA selects output storage;
it does not turn that renderer into a GPU renderer.

## Reference profiles

Extras declare compatible ranges. Bit-exact or MSE-bounded parity with an upstream renderer is
certified only under that family's reference profile: the `pin-<profile>` dependency groups in
`pyproject.toml` (`pin-classic`, `pin-navigation`, `pin-pusht`, `pin-dmc`, `pin-fetch`,
`pin-ogbench`, `pin-rocket`, `pin-craftax`, `pin-craftax-render`). They are development-only,
mutually exclusive, Python 3.12 only, and all resolved in the repository's single `uv.lock`:

```bash
uv sync --python 3.12 --extra dmcontrol --group pin-dmc --group test
```

## Extra incompatibilities

Fetch (`robotics`) needs `mujoco<3.12`. DM Control and OGBench accept
`mujoco<3.13`. Do not install `robotics` into the same environment as extras that want 3.12.

The renderer-only ports kept in the repository's `contrib/` directory are not released and have no
extras; their requirements files are described in
[contrib/README.md](https://github.com/galilai-group/stable-envs/blob/main/contrib/README.md).

## nvdiffrast

nvdiffrast is **not** a PyPI extra. PyPI rejects direct-URL requirements, and the package
builds a CUDA extension at install time against a matching local toolkit.

```bash
# nvcc must match torch's CUDA major version.
# CPATH must include torch's bundled CUDA headers (the build needs <cusparse.h>).
CUDA_HOME=... CPATH=... TORCH_CUDA_ARCH_LIST=... \
  pip install --no-build-isolation \
  "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0"
```

## Device honesty

Device support depends on the family and backend. Cube, Scene and Fetch have
software OpenGL CPU paths; their CUDA renderers require an NVIDIA GPU. Cube/Scene
physics still runs on CPU. See [backend selection](api/env.md) and
[limitations](concepts/limitations.md); CPU measurements must not use a hardware GL context.
