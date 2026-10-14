<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/logo-dark.png">
  <img src="docs/assets/brand/logo.png" alt="stable-envs" width="340">
</picture>

**A batched environment library for research: explicit state, dynamics, and variation, with fast state-driven rendering across CPU and GPU.**

[![License: MIT](https://img.shields.io/badge/license-MIT-black.svg)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-black.svg)](pyproject.toml)

Store the full State. Render frames from it when you need them.

Authors: **Randall Balestriero**, **Sami BuGhanem**.

## Install

```bash
pip install stable-envs
pip install "stable-envs[pusht]"              # PushT simulation and CPU rendering
pip install "stable-envs[dmcontrol,ogbench]"  # add family extras as needed
stable-envs doctor                            # check what works on this machine
```

A missing extra disables only that family. `snv.import_errors()` lists the modules that did not load.

CUDA is required for the 3-D families and most MuJoCo families. PushT, classic control, navigation and
several 2-D families run on CPU. Extras, version pins and the nvdiffrast build: [Install](docs/install.md).
Editable install from a clone: [Contributing](CONTRIBUTING.md).

## Quickstart

```python
import stable_envs as snv

with snv.CartPole(num_envs=64) as env:          # device="auto": CUDA if available, else CPU
    state = env.reset(seed=0)                # State
    pixels = env.render(state)               # Frame: (64, H, W, 3) uint8
    action = env.action_spec.sample(64, env.rng)
    out = env.step(action)
    state = out["state"]
    env.set_state(state)                     # restores the full State; returns None
```

Class names are the canonical API. `make` takes the same names as strings:

```python
env = snv.make("CartPole", num_envs=64)
print(snv.list_envs())
print(snv.info("CartPole"))
snv.set(verbose="INFO", default_device="auto")
```

`step(actions, states=slice(None), frames=slice(-1, None))` advances the live environments. Actions are
`[B,A]` for one step or `[B,T,A]` for a sequence. The two slices select which States and frames to
return; `None` turns either off. The final State is always returned. See
[Stepping](docs/concepts/dynamics.md). There is no `env.rollout(policy=...)`. Planning, datasets and
training belong to the sibling [stable-*](https://github.com/galilai-group) projects.

Full walkthrough: [Quickstart](docs/quickstart.md).

## Public API

```text
snv.Env, snv.State, snv.StepResult, snv.VariationLike
snv.Spec, snv.CompositeSpec
snv.make, snv.info, snv.list_envs, snv.list_families, snv.import_errors
snv.set, snv.get_config, snv.__version__
snv.CartPole / snv.Pendulum / …                  concrete env classes
```

`stable_envs.testing` is a public measurement library. It is not exported at `snv.*`.

## Environments

With every extra installed, `snv.list_envs()` returns 70 environments:

| family | environments |
|---|---:|
| `classic_control` | 5 |
| `dmcontrol` | 18 |
| `fetch` | 12 |
| `navigation` | 5 |
| `ogbench` | 27 |
| `pusht` | 2 |
| `pyflyt` | 1 |

A family registers only when its dependencies import. Without `mujoco`, DM Control, Fetch and OGBench
manipulation do not register. Without PyFlyt the count is 69.

Device support differs by family and backend. For example, Fetch on CPU needs `backend="opengl"`.
See [Limitations](docs/concepts/limitations.md).

## Documentation

| page | content |
|---|---|
| [Quickstart](docs/quickstart.md) | `reset`, `step`, `render`, specs |
| [Install](docs/install.md) | extras, pins, nvdiffrast |
| [Environments](docs/environments.md) | listing, metadata, fidelity tier |
| [State, Variation, Frame](docs/concepts/state.md) | what each value holds |
| [Stepping](docs/concepts/dynamics.md) | action sequences and output selection |
| [Lifecycle](docs/concepts/lifecycle.md) | `close()` and the context manager |
| [Limitations](docs/concepts/limitations.md) | devices and backends |
| [Benchmarks](docs/benchmarks.md) | how throughput is measured |
| [Adding a component](docs/adding/index.md) | environment, task and renderer guides |
| [Testing](docs/testing.md) | test layout and CI profiles |

Build the site locally:

```bash
pip install "stable-envs[docs]"
mkdocs serve
```

## Examples

```bash
python examples/render.py --help
python examples/batched_env.py --help
python examples/action_sequences.py
```

See [examples/README.md](examples/README.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
