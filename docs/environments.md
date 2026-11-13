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

`reference_id` names the environment in the [upstream library](terms.md#upstream-library) that the tests compare a class with, such as `CartPole-v1` or `dm_control/reacher-hard`. Each class table on a family page has "Renderer on CPU" and "Renderer on CUDA" rows that name the code drawing the [frames](terms.md#frame). PushT on CPU, for example, reproduces the pygame drawing of gym-pusht in Numba and gives the same bytes, and `PyFlytRocketLanding` on CPU draws with a Numba port of PyFlyt's TinyRenderer. The [renderers](contributing/renderers.md) follow the kind of family: 2D families build a `Scene2D` and rasterize it, OGBench mazes ray-cast analytic primitives, and MuJoCo families use MuJoCo's OpenGL renderer, MJWarp, the stable-envs [MuJoCo rasterizer](contributing/renderers.md#mujoco-rasterizer), or a stable-envs ray caster on CPU.

The tests measure the per-frame mean squared error (MSE) between each class's frames and the frames it declares as its comparison target. They compare it with a recorded value in the [frame regression checks](contributing/testing.md#frame-regression-checks).

Classes that declare a fidelity record report it in `env.describe()["visual_fidelity"]`: the measured MSE as `mse_mean`, `mse_p95` and `mse_max` on the 0-255 scale, with the measurement setup. `env.describe()["visual_note"]` says where the frames differ. These classes draw with a different rasterizer from the upstream library:

- Fetch, `OGBCube` and `OGBScene` with the default backend. The default is the stable-envs MuJoCo rasterizer (`backend="raster"`) on CPU and on CUDA. On CUDA, `OGBCube` and `OGBScene` use it at `supersample=1` only.
- The DM Control body-tracking classes. They declare a fidelity record.
- `PyFlytRocketLanding` on CUDA, which nvdiffrast draws. It declares a fidelity record for these frames.

`backend="opengl"` draws Fetch, `OGBCube` and `OGBScene` with MuJoCo OpenGL, the renderer of the upstream library. On CPU, these frames are bit-exact and repeatable. On CUDA, the NVIDIA OpenGL driver can change a pixel by 1/255 between two draws of the same state, on a few pixels of a few frames in each pass; the comparison renderer shows the same variation. For `OGBCube` and `OGBScene`, the OpenGL fidelity record compares two OpenGL instances with the same compiled appearance. It is not a comparison with the default OGBench appearance. At `supersample=2` or more, MJWarp is measured against that same stable-envs OpenGL path. See [OGBench](environments/ogbench.md#cube-and-scene-render-backends).

## Frame size

`native_size` is the only `size` that a class draws, or None when the class draws any size. When it is set, it is the size that the upstream library draws, and it is also `default_size`. The classic control, navigation and Craftax classes set it, because their upstream libraries draw each primitive in fixed pixels. Another `size` raises `ValueError`. `snv.info(name)["native_size"]` reads it without building the class.

??? note "Page data"

    The family pages read their tables from `docs/environments/_data/<Class>.json` and their clips from `docs/assets/media/envs/<Class>/`. `scripts/gen_env_docs.py` writes both. To refresh one class, run the script in an interpreter that registers it:

    ```bash
    python scripts/gen_env_docs.py PushT --meta --devices --media
    ```

    - `--meta` writes the metadata of a one-environment batch built on CPU.
    - `--devices` builds the class on each available device and records the result.
    - `--media` renders the two clips on the default device of the class. It needs `imageio-ffmpeg` or an `ffmpeg` executable.

    `tests/unit/test_env_docs_snapshot.py` checks each snapshot against the live class.
