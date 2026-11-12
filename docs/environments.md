# Overview

stable-envs ships 80 environment classes in 11 [families](terms.md#family). Each family has a page, and each class on it shows a clip of random actions and, for most classes, a clip of sampled [variation](terms.md#variation), the action, the [state](terms.md#state) keys and the variation leaves.

```python
import stable_envs as snv

print(snv.list_envs(family="classic_control"))  # ['Acrobot', 'CartPole', 'MountainCar', 'MountainCarContinuous', 'Pendulum']
print(snv.list_envs(family="ogbench", domain="pointmaze")[:2])  # ['PointMazeGiantNavigate', 'PointMazeGiantStitch']
with (
    snv.CartPole(num_envs=8, device="cpu") as cartpole,
    snv.TwoRoom(num_envs=8, device="cpu") as tworoom,
):
    print(cartpole.action_spec.shape, tworoom.action_spec.shape)  # (1,) (2,)
```

`snv.list_envs()` lists the classes registered on this interpreter, and the extras you installed decide which ones those are. [Missing extras](install.md#missing-extras) says what the core install registers and runs.

| Family | Classes | Examples |
|---|---:|---|
| [`classic_control`](environments/classic-control.md) | 5 | `CartPole`, `Pendulum` |
| [`craftax`](environments/craftax.md) | 4 | `CraftaxClassicPixels`, `CraftaxSymbolic` |
| [`dmcontrol`](environments/dmcontrol.md) | 17 | `DMCReacher`, `DMCCheetahRun` (`domain="reacher"`, `"cheetah"`, ...) |
| [`fetch`](environments/fetch.md) | 12 | `FetchReach`, `FetchPush`, ... |
| [`miniworld`](environments/miniworld.md) | 1 | `MiniWorldMovingBlockWorld` |
| [`mshuman`](environments/mshuman.md) | 1 | `MSHumanLocomotion` |
| [`navigation`](environments/navigation.md) | 5 | `TwoRoom`, `Piecewise`, `SimplePointMaze` |
| [`ogbench`](environments/ogbench.md) | 27 | `PointMazeLargeNavigate`, `OGBCube` (`domain="pointmaze"`, `"cube"`, ...) |
| [`pusht`](environments/pusht.md) | 2 | `PushT`, `PushTDiscrete` |
| [`pyflyt`](environments/pyflyt.md) | 1 | `PyFlytRocketLanding` |
| [`roadster`](environments/roadster.md) | 5 | `RoadsterTrafficJam`, `RoadsterCarHop` (`domain="traffic"`, `"carhop"`, ...) |
| **Total** | **80** | |

## Comparison with the upstream library

`reference_id` names the environment in the [upstream library](terms.md#upstream-library) that the tests compare a class with, such as `CartPole-v1` or `dm_control/reacher-hard`. Each class table on a family page has "Renderer on CPU" and "Renderer on CUDA" rows that name the code drawing the [frames](terms.md#frame). PushT on CPU, for example, reproduces the pygame drawing of gym-pusht in Numba and gives the same bytes, while `PyFlytRocketLanding` on CPU draws with PyFlyt's own TinyRenderer. The [renderers](contributing/renderers.md) follow the kind of family: 2D families build a `Scene2D` and rasterize it, OGBench mazes ray-cast analytic primitives, and MuJoCo families use MuJoCo's OpenGL renderer, MJWarp, or a stable-envs ray caster or rasterizer on CPU.

The tests measure the per-frame mean squared error (MSE) between each class's frames and the frames it declares as its comparison target. They compare it with a recorded value in the [frame regression checks](contributing/testing.md#frame-regression-checks).

Classes that declare a fidelity record report it in `env.describe()["visual_fidelity"]`: the measured MSE as `mse_mean`, `mse_p95` and `mse_max` on the 0-255 scale, with the measurement setup. The DM Control body-tracking classes declare one because they draw with a different rasterizer from the upstream library, and `PyFlytRocketLanding` declares one for its CUDA frames, which nvdiffrast draws. `OGBCube` and `OGBScene` declare one against the stable-envs MuJoCo OpenGL path. `env.describe()["visual_note"]` says where the frames differ.

??? note "Page data"

    The family pages read their tables from `docs/environments/_data/<Class>.json` and their clips from `docs/assets/media/envs/<Class>/`. `scripts/gen_env_docs.py` writes both. To refresh one class, run the script in an interpreter that registers it:

    ```bash
    python scripts/gen_env_docs.py PushT --meta --devices --media
    ```

    - `--meta` writes the metadata of a one-environment batch built on CPU.
    - `--devices` builds the class on each available device and records the result.
    - `--media` renders the two clips on the default device of the class. It needs `imageio-ffmpeg` or an `ffmpeg` executable.

    `tests/unit/test_env_docs_snapshot.py` checks each snapshot against the live class.
