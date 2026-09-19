# Environment listing

`snv.list_envs()` is the current answer **on this interpreter**. The package auto-discovers `stable_envs/envs/*`; a missing optional dependency costs that family and is recorded in `snv.import_errors()`, never silent.

```python
import stable_envs as snv

snv.list_envs()                      # sorted class names that registered
snv.list_envs(family="classic_control")
snv.list_families()                  # modules that contributed at least one class
snv.spec("PushT")                    # facts that do not need a device
snv.spec("swm/PushT-v1")             # SWM aliases resolve the same way as make()
snv.import_errors()                  # {module: exception}, families this machine lacks
```

`spec(name)` does **not** construct the env. Construction needs a device, and for several families a GPU, an EGL context or a compiled scene.

Returned fields include `name`, `module`, `family`, `state_dim`, `state_names`, `default_size`, `reference_id`, `fidelity_tier`, `render_decoder`, `action_dim`, `has_dynamics`, `swm_ids`.

Registration is the `@register` decorator in `stable_envs.registry`. Duplicate class names are an error. Modules whose name starts with `_` are skipped by discovery unless another module imports them. Import failures are recorded, never swallowed.

## Concrete classes are canonical

```python
env = snv.PushT(num_envs=8, device="cpu")
env = snv.CartPole(num_envs=8, device="cpu")
# snv.DMCReacherHard, snv.OGBCube, snv.CraftaxClassic, … if that family imported
```

`dir(snv)` lists the documented namespace **and** the registered class names. `stable_envs.testing` and `stable_envs.interop` still import for developers but are not advertised at `snv.*`.

## The 42 stable-worldmodel ids

`snv.SWM_IDS` is the frozen 42-id table: 40 pixel-producing, 2 symbolic / renderer-free. `make("swm/PushT-v1")` is an alias for the PushT class. Ids that map onto several cells (for example `swm/CartpoleDMControl-v0` → Balance and Swingup) refuse to pick one silently — ask for the class name.

The two symbolic ids raise from `make` / `spec` with that reason rather than "unknown env".

Per-id measurements live in the [research archive](research/index.md).

## Family filter caveat

`list_envs(family=…)` matches `cls.__module__.rsplit(".", 1)[-1]`. That last segment is not always the family name a reader would guess:

| what you might type | what actually registered |
|---|---|
| `"pusht"`, `"classic_control"`, `"navigation"`, `"craftax"` | those names |
| `"ogbench"` | maze cells whose `__module__` ends in `ogbench` |
| `"maze"` / `"manipulation"` | OGBench cells defined in those submodules |
| `"reacher"` | a DM Control task module, not the whole DMC family |

When in doubt, `list_families()` and `spec(name)["module"]` are the source of truth. The registry does not invent a second taxonomy.

## Fidelity

Every class declares exactly one of `parity` or `visual`. `spec(name)["fidelity_tier"]` reports it. Visual-tier frames are a different render of the same scene, suitable for live training data and **not** a substitute for official pixels at eval time. See [Fidelity tiers](concepts/fidelity-tiers.md).

## API reference by family

Generated from the module docstrings:

- [Env contract](api/env.md)
- [Registered families](api/envs.md)
