# Environment listing

`snv.list_envs()` is the current answer **on this interpreter**. The package auto-discovers `stable_envs/envs/*`; a missing optional dependency costs that family and is recorded in `snv.import_errors()`, never silent.

```python
import stable_envs as snv

snv.list_envs()                      # sorted canonical class names that registered
snv.list_envs(family="classic_control")
snv.list_envs(family="ogbench", domain="pointmaze")
snv.list_families()                  # stable family ids, not module paths
snv.info("PushT")                    # facts that do not need a device
snv.import_errors()                  # {module: exception}, families this machine lacks
```

`info(name)` does **not** construct the env. Construction needs a device, and for several families a GPU, an EGL context or a compiled scene.

Returned fields include `class_name`, `id` (slug), `family`, `domain`, `reference_id`, `render_state_dim`, `render_state_names`, `default_size`, `fidelity_tier`, `render_decoder`, `action_dim`, `has_dynamics`, `requires_cuda`.

`snv.Spec` is a tensor contract (`env.frame_spec`, `env.state_spec`, …). `snv.info` is metadata. They are different types with similar English names on purpose — do not confuse them.

Registration is the `@register` decorator in `stable_envs.registry`. Duplicate class names are an error. A class must declare `family=` explicitly; it is never inferred from the file path. Modules whose name starts with `_` are skipped by discovery unless another module imports them. Import failures are recorded, never swallowed.

## Concrete classes are canonical

```python
env = snv.PushT(num_envs=8)
env = snv.CartPole(num_envs=8)
# snv.DMCReacherHard, snv.OGBCube, snv.CraftaxClassic, snv.PointMazeLargeNavigate, …
```

`dir(snv)` lists the documented namespace **and** the registered class names. `stable_envs.testing` still imports for developers but is not advertised at `snv.*`.

`reference_id` is provenance of the official simulator or renderer a cell is measured against (`gym_pusht/PushT-v0`, `dm_control/reacher-hard`, …). It is not a `make()` alias.

Per-id certification measurements live in the [research archive](research/index.md).

## Families are declared, not derived

`list_envs(family=…)` matches the class attribute `family`, which does not change if a file moves.

| family | examples |
|---|---|
| `pusht` | `PushT` |
| `classic_control` | `CartPole`, `Pendulum` |
| `dmcontrol` | `DMCReacherHard`, `DMCCheetahRun` (`domain="reacher"`, `"cheetah"`, …) |
| `ogbench` | `PointMazeLargeNavigate`, `OGBCube` (`domain="pointmaze"`, `"cube"`, …) |
| `mujoco` | `MuJoCoAnt`, `MuJoCoHalfCheetah` |
| `pyflyt` | `PyFlytRocketLanding` |
| `fetch` | `FetchReach`, `FetchPush`, … |

When in doubt, `list_families()` and `info(name)["family"]` / `["domain"]` are the source of truth.

## Fidelity

Every class declares exactly one of `parity` or `visual`. `info(name)["fidelity_tier"]` reports it. Visual-tier frames are a different render of the same scene, suitable for live training data and **not** a substitute for official pixels at eval time. See [Fidelity tiers](concepts/fidelity-tiers.md).

## API reference by family

Generated from the module docstrings:

- [Env contract](api/env.md)
- [Registered families](api/envs.md)
