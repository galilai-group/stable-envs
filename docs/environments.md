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

`info` returns `class_name`, `id`, `family`, `domain`, `reference_id`, `render_state_dim`,
`render_state_names`, `default_size`, `fidelity_tier`, `action_dim`, `has_dynamics` and `requires_cuda`.

`snv.Spec` is a tensor contract (`env.frame_spec`, `env.state_spec`, …). `snv.info` returns metadata.
They are different things.

Registration uses the `@register` decorator in `stable_envs.registry`. Duplicate class names are an
error. Each class declares `family=` explicitly. Discovery skips modules whose names start with `_`
unless another module imports them.

## Concrete classes are canonical

```python
env = snv.PushT(num_envs=8)
env = snv.CartPole(num_envs=8)
# snv.DMCReacherHard, snv.OGBCube, snv.PointMazeLargeNavigate, …
```

`dir(snv)` lists the documented namespace and the registered class names.

`reference_id` names the official simulator or renderer that a class is compared against
(`gym_pusht/PushT-v0`, `dm_control/reacher-hard`, …). It is not a `make()` alias.

## Families

`list_envs(family=…)` matches the class attribute `family`.

| family | environments | examples |
|---|---:|---|
| `classic_control` | 5 | `CartPole`, `Pendulum` |
| `dmcontrol` | 18 | `DMCReacherHard`, `DMCCheetahRun` (`domain="reacher"`, `"cheetah"`, …) |
| `fetch` | 12 | `FetchReach`, `FetchPush`, … |
| `navigation` | 5 | `TwoRoom`, `Piecewise`, `SimplePointMaze` |
| `ogbench` | 27 | `PointMazeLargeNavigate`, `OGBCube` (`domain="pointmaze"`, `"cube"`, …) |
| `pusht` | 2 | `PushT`, `PushTDiscrete` |
| `pyflyt` | 1 | `PyFlytRocketLanding` |

These 70 register with every extra installed. Without PyFlyt there are 69.

## Fidelity

Every class declares one tier: `parity` or `visual`. `info(name)["fidelity_tier"]` reports it.
Visual-tier frames are a different render of the same scene. Use them for training data. Do not use
them in place of official pixels at evaluation. See [Fidelity tiers](concepts/fidelity-tiers.md).

## API reference

- [Env](api/env.md)
- [Registered environments](api/envs.md)
