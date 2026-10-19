# Environments

`snv.list_envs()` lists the environments registered **on this interpreter**. The package discovers
`stable_envs/envs/*` automatically. If an optional dependency is missing, that family does not register,
and `snv.import_errors()` records why.

```python
import stable_envs as snv

snv.list_envs()                      # sorted class names that registered
snv.list_envs(family="classic_control")
snv.list_envs(family="ogbench", domain="pointmaze")
snv.list_families()                  # family ids
snv.info("PushT")                    # metadata; does not need a device
snv.import_errors()                  # {module: exception} for families this machine lacks
```

`info(name)` does not construct the env. Construction needs a device, and some families need a GPU, an
EGL context or a compiled scene.

`info` returns `class_name`, `id`, `class`, `module`, `family`, `domain`, `reference_id`,
`render_state_dim`, `render_state_names`, `default_size`, `action_dim`, `has_dynamics` and
`requires_cuda`. `class` is the class object and `module` is the name of its module.

`snv.info` returns metadata. It does not return an `snv.Spec`. An `snv.Spec` describes tensors
(`env.frame_spec`, `env.state_spec`, …).

Registration uses the `@register` decorator in `stable_envs.registry`. Duplicate class names are an
error. Each class declares `family=` explicitly. Discovery skips modules whose names start with `_`
unless another module imports them.

## Concrete classes are canonical

```python
env = snv.PushT(num_envs=8)
env = snv.CartPole(num_envs=8)
# snv.DMCReacherHard, snv.OGBCube, snv.CraftaxClassicPixels, snv.PointMazeLargeNavigate, …
```

`dir(snv)` lists the documented namespace and the registered class names.

`reference_id` names the official simulator or renderer that a class is compared against
(`gym_pusht/PushT-v0`, `dm_control/reacher-hard`, …). It is not a `make()` alias.

## Families

`list_envs(family=…)` matches the class attribute `family`. Each family has a page. A page shows a
clip of each class with random actions and a clip with sampled Variation. It also lists the action, the
State keys and the Variation leaves of each class.

| family | environments | examples |
|---|---:|---|
| [`classic_control`](environments/classic-control.md) | 5 | `CartPole`, `Pendulum` |
| [`craftax`](environments/craftax.md) | 4 | `CraftaxClassicPixels`, `CraftaxSymbolic` |
| [`dmcontrol`](environments/dmcontrol.md) | 18 | `DMCReacherHard`, `DMCCheetahRun` (`domain="reacher"`, `"cheetah"`, …) |
| [`fetch`](environments/fetch.md) | 12 | `FetchReach`, `FetchPush`, … |
| [`navigation`](environments/navigation.md) | 5 | `TwoRoom`, `Piecewise`, `SimplePointMaze` |
| [`ogbench`](environments/ogbench.md) | 27 | `PointMazeLargeNavigate`, `OGBCube` (`domain="pointmaze"`, `"cube"`, …) |
| [`pusht`](environments/pusht.md) | 2 | `PushT`, `PushTDiscrete` |
| [`pyflyt`](environments/pyflyt.md) | 1 | `PyFlytRocketLanding` |

With every extra installed, these 74 classes register. Without PyFlyt, 73 classes register.

## Comparison with the source

`reference_id` names the source that the tests compare a class with. The "Renderer on CPU" and
"Renderer on CUDA" rows of each class on its family page name the code that draws the frames. For
example, PushT uses the source's own pygame drawing, and PyFlytRocketLanding uses TinyRenderer on CPU.

For some classes, the tests compare the frames by the mean absolute difference, the fraction of
pixels within 8/255 and the SSIM. For these classes, `env.describe()["visual_fidelity"]` gives the
measured values and the measurement setup. `env.describe()["visual_note"]` tells where the frames differ. See
[Limitations](concepts/limitations.md#agreement-with-the-source).

## API reference

- [Env](api/env.md)
- [Registered environments](api/envs.md)

## Page data

The family pages read their tables from `docs/environments/_data/<Class>.json`. The clips are in
`docs/assets/media/envs/<Class>/`. `scripts/gen_env_docs.py` writes both. To refresh one class, run
the script in an interpreter that registers the class:

```bash
python scripts/gen_env_docs.py PushT --meta --devices --media
```

- `--meta` writes the metadata from a one-lane env on CPU.
- `--devices` builds the class on each available device and records the result.
- `--media` renders the two clips with the default device of the class. It needs `imageio-ffmpeg` or
  an `ffmpeg` executable.

`tests/unit/test_env_docs_snapshot.py` checks each snapshot against the live class.
