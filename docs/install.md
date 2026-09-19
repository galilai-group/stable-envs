# Install

```bash
pip install stable-envs
pip install "stable-envs[pusht]"              # gym_pusht + pymunk + pygame — parity tests only
pip install "stable-envs[dmcontrol,ogbench]"  # add family extras as needed
```

There is no `[all]` extra. A missing optional extra costs **that family**, not the package.
`snv.import_errors()` names the modules that did not load.

```python
import stable_envs as snv
snv.import_errors()     # {module: "ModuleNotFoundError: ..."}
snv.list_families()     # modules that did load
```

CUDA is required for the 3-D and most MuJoCo families. PushT, classic control, navigation and
several 2-D families run on CPU.

From a clone, use an editable install — see [Contributing](https://github.com/galilai-group/stable-envs/blob/main/CONTRIBUTING.md).

## Extras

Each family extra installs what that family's **reference** needs (parity, CPU-vs-GPU benches,
gallery). Most extras are not required to *render*.

| extra | what it adds |
|---|---|
| `pusht` | official PushT renderer (`gym-pusht`, pygame, pymunk) |
| `box2d` | Gymnasium Box2D |
| `minigrid` | MiniGrid / BabyAI |
| `ogbench` | OGBench mazes and manipulation (`mujoco>=3.11,<3.13`) |
| `dmcontrol` | DM Control (`mujoco>=3.11,<3.13`) |
| `mujoco-locomotion` | Gymnasium MuJoCo locomotion |
| `robotics` | Fetch (`mujoco>=3.11,<3.12`) |
| `mjwarp` | MuJoCo Warp render backend |
| `metaworld` | Meta-World (install the package itself with `--no-deps`) |
| `humanoidbench` | HumanoidBench remainder (the env package is not on PyPI) |
| `craftax` / `crafter` | Craftax (needs jax) / Crafter |
| `overcooked` | Overcooked (needs its **own interpreter**) |
| `rocket` | PyFlyt rocket landing |
| `highway` / `minihack` / `sokoban` / `griddly` / `jumanji` / `pgx` / `procgen` | those families |
| `swm` / `interop` | `stable-worldmodel` |
| `docs` / `dev` / `examples` / `hf` | documentation, tests, gallery helpers, HuggingFace |

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

OGBCube, OGBScene and PFRocketLanding refuse `device="cpu"` rather than silently falling
back to CUDA. Fetch has no CPU backend. Details: [limitations](concepts/limitations.md).
