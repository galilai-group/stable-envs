<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/logo-dark.png">
  <img src="docs/assets/brand/logo.png" alt="stable-envs" width="340">
</picture>

**A batched environment library for research: explicit state, dynamics, and variation, with state-driven rendering across CPU and GPU.**

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

The 3-D families and most MuJoCo families need CUDA. PushT, classic control, navigation, Roadster and
several 2-D families run on CPU. [Install](docs/install.md) covers extras, version pins and the nvdiffrast build.
[Contributing](CONTRIBUTING.md) covers an editable install from a clone.

## Quickstart

```python
import stable_envs as snv

env = snv.CartPole(num_envs=64)          # device="auto": CUDA if available, else CPU
state = env.reset(seed=0)                # State
pixels = env.render(state)               # Frame: (64, H, W, 3) uint8
action = env.action_spec.sample(64, env.rng)
out = env.step(action)
state = out["state"]
env.set_state(state)                     # restores the full State; returns None
env.close()
```

`env.close()` stops the simulator's worker processes and frees the environment's own
buffers. Renderer contexts that other environments share stay cached. An environment is also a
context manager: `with snv.CartPole(num_envs=64) as env:` closes it at the end of the block.

Class names are the canonical API. `make` takes the same names as strings:

```python
env = snv.make("CartPole", num_envs=64)
print(snv.list_envs())
print(snv.info("CartPole"))
snv.set(verbose="INFO", default_device="auto")
```

`step(actions, *, states=slice(None), frames=slice(-1, None))` advances the live environments. Actions
are `[B,A]` for one step or `[B,T,A]` for a sequence. The two keyword-only slices select which States
and frames `step` returns. `None` turns either one off. `step` always returns the final State. See
[Stepping](docs/concepts/dynamics.md).

There is no `env.rollout(policy=...)`. The sibling [stable-*](https://github.com/galilai-group) projects
handle planning, datasets and training.

The [Quickstart](docs/quickstart.md) has the full walkthrough.

## Public API

```text
snv.Env, snv.State, snv.StepResult, snv.VariationLike
snv.Spec, snv.CompositeSpec
snv.make, snv.info, snv.list_envs, snv.list_families, snv.import_errors
snv.set, snv.get_config, snv.__version__
snv.tune, snv.tuning
snv.CartPole / snv.Pendulum / …                  concrete env classes
```

`stable_envs.testing` is a public measurement library. `snv.*` does not export it.

## Environments

The extras together provide 79 environment classes. One Python environment cannot hold every extra.
The `robotics` extra for Fetch pins MuJoCo < 3.12, and the `ogbench-manipulation` extra for `OGBCube`
and `OGBScene` pins MuJoCo 3.12.0. Install them in separate environments.

| family | environments |
|---|---:|
| `classic_control` | 5 |
| `craftax` | 4 |
| `dmcontrol` | 18 |
| `fetch` | 12 |
| `navigation` | 5 |
| `ogbench` | 27 |
| `pusht` | 2 |
| `pyflyt` | 1 |
| `roadster` | 5 |

A family registers only when its module imports. `snv.import_errors()` names each module that did not
import. Craftax needs the `craftax` extra to register. `PyFlytRocketLanding` needs the `rocket` extra,
and without it the count is 78. DM Control compiles a test model at import, so it registers only with
`dm_control` and MuJoCo installed. The Fetch and OGBench modules import MuJoCo and their simulator
packages inside functions. These classes register without them and raise an error when you use them.

Device support differs by family and backend. See [Limitations](docs/concepts/limitations.md).

## Documentation

| page | content |
|---|---|
| [Quickstart](docs/quickstart.md) | `reset`, `step`, `render`, specs |
| [Install](docs/install.md) | extras, pins, nvdiffrast |
| [System setup](docs/system-setup.md) | driver, OpenGL and `stable-envs doctor` |
| [Environments](docs/environments.md) | listing, metadata, comparison with the source |
| [State, Action, Variation and Frame](docs/concepts/state.md) | what each value holds |
| [Stepping](docs/concepts/dynamics.md) | action sequences and output selection |
| [Lifecycle](docs/concepts/lifecycle.md) | `close()` and the context manager |
| [Limitations](docs/concepts/limitations.md) | devices and backends |
| [Benchmarks](docs/benchmarks.md) | how the harnesses measure throughput |
| [Adding a component](docs/adding/index.md) | environment, task and renderer guides |
| [Testing](docs/testing.md) | test layout and CI profiles |
| [Command line](docs/api/cli.md) | `stable-envs doctor` |
| [Changelog](CHANGELOG.md) | changes in each release |

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
