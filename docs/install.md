# Install

```bash
pip install stable-envs
pip install "stable-envs[pusht]"              # gym_pusht + pymunk + pygame — parity tests only
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
| *(none)* | PushT, classic control, navigation | core install is enough |
| `pusht` | no | official `gym-pusht` / pygame / pymunk — **parity tests only** |
| `box2d` | LunarLander / CarRacing reference path | Gymnasium Box2D |
| `minigrid` | MiniGrid / BabyAI | MiniGrid |
| `ogbench` | OGBench mazes (and manip, with CUDA) | `ogbench` + MuJoCo |
| `dmcontrol` | DM Control cells | `dm_control` + MuJoCo |
| `mujoco-locomotion` | Gymnasium MuJoCo locomotion | Gymnasium MuJoCo |
| `robotics` | Fetch | gymnasium-robotics (MuJoCo pin) |
| `mjwarp` | Fetch / some visual 3-D cells | MuJoCo Warp render backend |
| `metaworld` | Meta-World | scenes come from the package's own MjModel |
| `humanoidbench` | HumanoidBench remainder | env package is not on PyPI |
| `craftax` / `crafter` | Craftax / Crafter | jax / Crafter |
| `overcooked` | Overcooked | needs its **own interpreter** |
| `rocket` | PyFlyt rocket (CUDA + nvdiffrast) | pillow / pybullet / scipy |
| `highway` / `minihack` / `sokoban` / `griddly` / `jumanji` / `pgx` / `procgen` | those families | those packages |
| `docs` / `dev` / `examples` / `hf` | no | documentation, tests, gallery helpers, HuggingFace |

`pip install "stable-envs[pusht]"` is **not** required to call `snv.PushT()` and `env.render()`. It is required to compare against the official pygame renderer.

## Extra incompatibilities

Fetch (`robotics`) needs `mujoco<3.12`. DM Control, OGBench and MuJoCo locomotion accept
`mujoco<3.13`. Do not install `robotics` into the same environment as extras that want 3.12.

Overcooked's `jaxmarl` pin is incompatible with the rest of the stack. Use a dedicated
interpreter for that extra.

Meta-World pins `mujoco==3.3.0` on paper and runs on 3.11. Install it with `--no-deps`:

```bash
pip install --no-deps metaworld
pip install "stable-envs[metaworld]"
```

HumanoidBench is not on PyPI. Install the env from git with `--no-deps`, then
`pip install "stable-envs[humanoidbench]"`. The family registers only when
`STABLE_ENVS_HUMANOIDBENCH=1` is set: `mujoco_warp`'s batched render is not bit-reproducible
across a long conformance session.

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

The measured source install, including the header-path failure mode, is in
[nvdiffrast](research/nvdiffrast.md#1-install).

## Device honesty

OGBCube, OGBScene and PyFlytRocketLanding refuse `device="cpu"` (and `device="auto"` on a CPU-only
machine) rather than silently falling back to CUDA. Fetch has no CPU backend. Details: [limitations](concepts/limitations.md).
