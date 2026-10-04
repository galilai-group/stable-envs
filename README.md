<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/logo-dark.png">
  <img src="docs/assets/brand/logo.png" alt="stable-envs" width="340">
</picture>

**A batched environment library for research: explicit state, dynamics, and variation, with fast state-driven rendering across CPU and GPU.**

[![License: MIT](https://img.shields.io/badge/license-MIT-black.svg)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-black.svg)](pyproject.toml)

Keep exhaustive State for replay and branching. Render frames from State when needed. Compact RenderState is a private renderer projection and does not replace a simulator checkpoint.

Authors: **Randall Balestriero**, **Sami BuGhanem**.

## Install

```bash
pip install stable-envs
pip install "stable-envs[pusht]"              # source-aligned PushT simulation and CPU rendering
pip install "stable-envs[dmcontrol,ogbench]"  # add family extras as needed
```

A missing optional extra costs **that family**, not the package. `snv.import_errors()` names the modules that did not load.

CUDA is required for the 3-D and most MuJoCo families. PushT, classic control, navigation and several 2-D families run on CPU.

Family extras that cannot share an interpreter, the Fetch MuJoCo pin, and the nvdiffrast source build: [Install](docs/install.md).

Editable install (from a clone) lives under [Contributing](CONTRIBUTING.md).

## Quickstart

```python
import stable_envs as snv

with snv.CartPole(num_envs=64) as env:          # device="auto" → CUDA if available, else CPU
    state = env.reset(seed=0)                # State
    pixels = env.render(state)               # Frame: (64, H, W, 3) uint8
    action = env.action_spec.sample(64, env.rng)
    out = env.step(action)                   # stateful
    state = out["state"]
    env.set_state(state)                    # restores full State; returns None
```

Concrete classes are canonical. Strings are those class names:

```python
env = snv.make("CartPole", num_envs=64)
print(snv.list_envs())
print(snv.info("CartPole"))
snv.set(verbose="INFO", default_device="auto")
```

`step(actions, states=slice(None), frames=slice(-1, None))` advances live environments. Actions are `[B,A]` for one step or `[B,T,A]` for a sequence. Independent slices select State history and rendered frames; `None` disables either history. The final State is always returned. See [selection semantics](docs/concepts/dynamics.md). There is no `env.rollout(policy=...)`. Planning, datasets and training belong to sibling [stable-*](https://github.com/galilai-group) projects.

Full walkthrough: [Quickstart](docs/quickstart.md).

## What the public API is

```text
snv.Env, snv.State, snv.StepResult, snv.VariationLike
snv.Spec, snv.CompositeSpec
snv.make, snv.info, snv.list_envs, snv.list_families, snv.import_errors
snv.set, snv.get_config, snv.__version__
snv.CartPole / snv.Pendulum / …                  concrete env classes
```

`stable_envs.testing` is a public measurement library; it is not advertised at `snv.*`. There is no runtime interop contract with other stable-* packages.

| topic | page |
|---|---|
| Environment listing | [docs/environments.md](docs/environments.md) |
| State, Action, Variation, Frame | [docs/concepts/state.md](docs/concepts/state.md) |
| Stepping, action sequences and capture | [docs/concepts/dynamics.md](docs/concepts/dynamics.md) |
| `close()` / context manager | [docs/concepts/lifecycle.md](docs/concepts/lifecycle.md) |
| Sibling projects | [docs/concepts/worldmodel.md](docs/concepts/worldmodel.md) |
| Backend and device limitations | [docs/concepts/limitations.md](docs/concepts/limitations.md) |
| How throughput numbers were taken | [docs/benchmarks.md](docs/benchmarks.md) |

## Examples

These three scripts use only the public `snv` surface and degrade honestly when an extra is missing:

```bash
python examples/render.py --help
python examples/batched_env.py --help
python examples/functional_dynamics.py --help
```

Historical training loops, GPU sweeps and paper tables stay under [`docs/research/`](docs/research/). They are evidence, not the getting-started path.

## Keep your training loop. Delete your image dataset.

`render(full_state)` produces frames without advancing simulation. Learner-visible
observation selection and dataset filtering belong to stable-worldmodel. Renderer
parity and exact simulation/restoration are separate acceptance gates.

For contributor-only compact corpus handling, see the
[renderer guide](docs/adding/renderer.md). Public applications keep full State
snapshots; compact rows cannot be used to restore omitted simulator fields.

Side-by-side against the official renderer, same recorded states: [Gallery](docs/gallery.md).

## Optional families

`len(snv.list_envs())` is the current answer **on this interpreter**. The union over every extra is larger; a machine without `mujoco` simply does not register DM Control, Fetch, OGBench manipulation, and so on.

```python
snv.import_errors()  # {module: "ModuleNotFoundError: ..."}
snv.list_families()  # stable family ids that registered
```

Device honesty is part of the contract: OGBCube, OGBScene and PyFlytRocketLanding refuse CPU rather than silently falling back; Fetch has no CPU backend; HumanoidBench registers only under `STABLE_ENVS_HUMANOIDBENCH=1`. Details: [limitations](docs/concepts/limitations.md).

## Documentation

```bash
pip install "stable-envs[docs]"
mkdocs serve
```

| | |
|---|---|
| [Home](docs/index.md) | claim, quickstart, where to go |
| [Install](docs/install.md) | extras, pins, nvdiffrast |
| [Adding a component](docs/adding/index.md) | environment, task and renderer guides |
| [Testing](docs/testing.md) | what each check rules out |
| [Research archive](docs/research/index.md) | per-family studies, including the no-gos |

## License

MIT. See [LICENSE](LICENSE).
