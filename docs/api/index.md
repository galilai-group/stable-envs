# stable_envs

`import stable_envs as snv` exposes every registered class, such as `snv.CartPole`, and the functions below. Environment methods are on [Env](env.md).

```python
import stable_envs as snv

print(snv.__version__)                    # 0.1.0
env = snv.make("CartPole", num_envs=4, device="cpu")
print(type(env).__name__, env.num_envs)   # CartPole 4
env.close()
print(snv.info("CartPole")["reference_id"])  # CartPole-v1
print(snv.get_config())  # {'verbose': 'WARNING', 'default_device': 'auto'}
```

| Name | What it does |
|---|---|
| `snv.make(name, **kwargs)` | Build a registered environment by class name. |
| `snv.list_envs(family=, domain=)` | List registered class names. |
| `snv.list_families()` | List the family IDs that registered a class. |
| `snv.info(name)` | Read class metadata without building anything. |
| `snv.import_errors()` | Say why a family didn't register. |
| `snv.set(...)`, `snv.get_config()` | Change and read the log level and the default device. |
| `snv.tune(env)` | Measure the fastest tuning knob values for an environment's render path. |
| `snv.__version__` | The installed version. Put it in bug reports next to `stable-envs doctor --json`. |
| `snv.Env`, `snv.State`, `snv.StepResult`, `snv.VariationLike` | The environment base class and its types; see [Env](env.md). |
| `snv.Spec`, `snv.CompositeSpec` | Shape and dtype descriptions; see [Specs and spaces](specs.md). |
| `snv.tuning` | The tuning knobs; see [Tuning](tuning.md). |

## Build an environment

`snv.make("CartPole", num_envs=4)` and `snv.CartPole(num_envs=4)` build the same thing. Use `make` when the class name comes from a config file or a command line. The name is the Python class name: `"CartPole"` works, and the [upstream library](../terms.md#upstream-library)'s id `"CartPole-v1"` is an unknown name. An unknown name raises `KeyError` with the list of registered names. An unregistered attribute, such as `snv.DMCReacher` without the `dmcontrol` extra, raises `AttributeError` that points you to `snv.import_errors()`.

::: stable_envs.make

## Find environments

A family registers only if its module imports. The extras you install decide which classes appear, starting from [what the core install lists](../install.md#missing-extras). `snv.info(name)` reads metadata from the class, and it works for registered classes that need a GPU or an extra you haven't installed. It returns `class_name`, `id`, `class`, `module`, `family`, `domain`, `render_state_dim`, `render_state_names`, `default_size`, `reference_id`, `action_dim`, `has_dynamics` and `requires_cuda`. Values that depend on an instance, such as the width of a Fetch render row, are `None`. For shapes and dtypes, build the environment and read its [specs](../terms.md#spec).

::: stable_envs.list_envs

::: stable_envs.list_families

::: stable_envs.info

::: stable_envs.import_errors

## Global settings

`snv.set` changes two process-wide settings: `verbose`, the lowest level the library logs, and `default_device`, the device a constructor uses when you omit `device` on a class whose default is the global setting. [Env](env.md#construction) lists which classes those are. Constructors read `default_device` when they run. Environments you already built stay on their device when you change it. The starting log level comes from `STABLE_ENVS_VERBOSE`.

::: stable_envs.set

::: stable_envs.get_config

## Tune a render path

`snv.tune(env)` is the Python form of [`stable-envs tune`](cli.md#stable-envs-tune). Pass `write=False` to leave the cache untouched.

::: stable_envs.tune
