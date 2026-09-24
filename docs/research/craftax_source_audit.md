# Craftax source, ownership and candidate-State audit

Starting clean checkpoint: `bcf8699d9983f95e97b3c71607521a61c354247b`, the exact
Cube/Scene promotion/report commit. Worktree `/private/tmp/stable-envs-semantic-recovery`,
branch `repair/environment-semantics`. The committed matrix is 35 VERIFIED / 7 INCOMPLETE.
Both Cube/Scene acceptance chains and all three GPU success markers resolve;
173 GPU and 193 final manifest entries verify without rerunning their mutable scripts.
No push occurred. [Starting record](semantic_recovery_evidence/craftax_source_audit/start.json).

## Authority and ownership

The immutable SWM oracle is `176d54532b3c0a251a4f509b74f2fdacc143a63d`.
Its `stable_worldmodel/envs/__init__.py:194–212` registers four discrete Gymnasium IDs.
The registration helper forwards to Gymnasium with no `max_episode_steps` override.
Normal `gym.make` supplies OrderEnforcing/PassiveEnvChecker, not TimeLimit or AutoReset.
The concrete wrapper lives entirely in `stable_worldmodel/envs/craftax/craftax.py` (152 lines).
It calls the upstream factory with `auto_reset=False`. There is no private game fork.

Authority is [Craftax 1.5.0 on PyPI](https://pypi.org/project/craftax/1.5.0/),
[official repository](https://github.com/MichaelTMatthews/Craftax), tag
[`v1.5.0`](https://github.com/MichaelTMatthews/Craftax/tree/v1.5.0), commit
`47fff7a0b44b2842006661a2caab9ee6505fa210`.
All **35 upstream Python files** match the installed oracle dependency, the isolated
research installation and the official tag byte for byte. The SWM lock's wheel hash
is `96826abc7d1fcf5ffbdd022100a6815b38c2b9bd22eaf43932a2252d2d7304d8`;
its sdist hash is `a8deddea60167934c23723a83df882e07a683c0e3ec7fb3a3e0a6364543401a9`.
[Provenance](semantic_recovery_evidence/craftax_source_audit/provenance.json) records
absolute paths, hashes, sizes and source definition line ranges.

| Historical registration | SWM class | Upstream concrete class | Engine / output |
|---|---|---|---|
| `swm/CraftaxClassicPixels-v1` | `CraftaxClassicPixelsWrapper` | `CraftaxClassicPixelsEnvNoAutoReset` | Classic / float32 63×63×3 pixels |
| `swm/CraftaxClassicSymbolic-v1` | `CraftaxClassicSymbolicWrapper` | `CraftaxClassicSymbolicEnvNoAutoReset` | Classic / float32 1345-vector |
| `swm/CraftaxPixels-v1` | `CraftaxPixelsWrapper` | `CraftaxPixelsEnvNoAutoReset` | Full / float32 130×110×3 pixels |
| `swm/CraftaxSymbolic-v1` | `CraftaxSymbolicWrapper` | `CraftaxSymbolicEnvNoAutoReset` | Full / float32 8268-vector |

Each pair uses the same `craftax_step`, world generation, EnvState/EnvParams and
StaticEnvParams. Only `get_obs` and declared observation metadata differ. Full's
upstream declared pixel H/W are swapped; SWM explicitly supplies the actual shape.
These are two engines × two output configurations, not four dynamics implementations.

| Owner | Module / methods | Semantic responsibility |
|---|---|---|
| SWM | `CraftaxWrapper.__init__`, `_reset`, `_step` | Upstream factory, default params, owned key splitting, outer JIT/backend selection |
| SWM | `reset`, `step`, `info` | Seed replacement, cached last output/info, NumPy float32 output, Python reward/bools; truncation always false |
| SWM | `render` | Pixels only, `rgb_array` only; clip normalized cached output, multiply by 255, truncate to uint8 |
| Upstream | `craftax_env.make_craftax_env_from_name` | Four exact no-auto-reset classes |
| Upstream | `environment_base/environment_bases.py` | JIT reset/step; no hidden auto-reset |
| Upstream | `craftax_classic/envs/*`, `craftax/envs/*` | Reset/step/output orchestration, terminal/discount and info |
| Upstream | `craftax_classic/world_gen.py`, `craftax/world_gen/*` | Procedural worlds and initial state |
| Upstream | Both `game_logic.py` trees and full `util/*` | Actions, crafting, mining, combat, mobs, plants, inventory, achievements, rewards |
| Upstream | Both `renderer.py` modules and PNG assets | Symbolic encoding, all pixel arithmetic, night noise, textures |
| Proposed stable-envs | Narrow codec/adapter | Named numeric State, key ownership, lifecycle, atomic restoration, batch/device bridge, RenderState projection |

Classic has 17 actions, full 43, including NOOP=0, movement 1–4, DO=5,
sleep=6, placement 7–10 and crafting 11–16. Full adds rest/floor transitions,
diamond equipment, projectiles/spells, potions/books/enchanting and attributes.
SWM casts a scalar action with `int(action)`; upstream does not supply a robust
invalid-action validator. The proposed public boundary validates discrete integer
IDs and refuses out-of-range or fractional values rather than relying on JAX's
out-of-bounds indexing behavior. Batched actions must be explicit per lane.

Classic reward is new achievement count plus 0.1×health change. Full uses the
upstream achievement-weight map plus 0.1×health change. Classic terminates on
step limit (10000), death or lava; full on step limit (100000), death or defeated
boss. SWM maps all these to `terminated`, never `truncated`, and never auto-resets.
The adapter must preserve those flags and the explicit terminal State. No
post-terminal auto-reset may be introduced.

## Candidate numeric State and PRNG

[Complete measured engine schema](semantic_recovery_evidence/craftax_source_audit/candidate-engine-schema.json):
49 numeric Classic leaves (20,952 bytes) and 84 full leaves (273,276 bytes).
Keys retain source semantic names under `engine`, including nested inventory and
entity records; shapes and dtypes are measured from real reset results. Four
`fractal_noise_angles=None` entries are fixed structural constants in this bounded
configuration, encoded by compatibility signature, not arbitrary public objects.
Every numeric leaf receives a leading public batch axis `[B]`; scalar engine
leaves become `[B]`. No engine dataclass, pytree, pickle or JIT executable is exposed.

All engine leaves are exhaustive dynamic snapshot data, even when absent from
current observations: complete maps/all nine levels, collision map, entity positions,
health/masks/cooldowns/types, projectiles, plant ages, hidden intrinsics, inventory,
achievements, potion permutation, spells, enchantments, boss state, timestep and
lighting. Offscreen maps, cooldowns, hunger/recovery accumulators and procedural
keys cannot be recovered from either a symbolic vector or a pixel frame.

The full public proposal adds:

- `rng.wrapper_key`: the two uint32 words of the *measured* legacy Threefry key.
- `meta.signature`: 32 uint8 bytes identifying engine, output configuration,
  package/profile versions, PRNG settings, static/default parameters and schema.
- Named numeric fixed parameter fields: exact EnvParams and StaticEnvParams;
  structural None values remain encoded in the signature.
- Lifecycle fields for the last reward and termination/truncation state.

The wrapper initializes `jax.random.PRNGKey(seed)`. On every reset or step:
`key1, key2 = jax.random.split(wrapper_key)`; `key2` goes to upstream, `key1`
is retained. An explicitly seeded reset replaces the retained key first; an
unseeded reset consumes its current value. The separate `engine.state_rng` is
also a legacy uint32[2] key. Upstream game logic splits its supplied key for
interactions/mobs/spawns and stores a final split in state. Night rendering reads
that key functionally; it does not mutate or consume the wrapper key. No typed-key
conversion is assumed. The current configuration is Threefry2x32 with partitionable
Threefry enabled and x64 disabled; incompatible PRNG settings must refuse.

Restore order: validate the entire batch's exact keys/shapes/dtypes, signature,
fixed parameters, finite/index/range and lifecycle invariants; decode every lane
into independent upstream immutable dataclasses and keys; only then swap all
owned lanes and lifecycle fields atomically. No world generation, history replay,
`step`, global RNG mutation or simulator reconstruction belongs in `set_state`.
A RenderState must have a distinct signature/key set and fail dynamics restoration.

No Variation is proposed: SWM exposes `variation_space=None`, and registered
parameters are fixed. Seeds remain reset input and retained key State. Parameter
sweeps, alternate static sizes and fractal-noise options remain outside this scope.

## Research replay and output boundary

[Research replay](semantic_recovery_evidence/craftax_source_audit/replay-results.json)
passes independently for all four concrete wrappers. Each uses seeds 0/17/42,
independent different-history instances, detached numeric TensorDict snapshots,
seven-step continuation and unseeded reset continuation. Eight bounded source
fixtures exercise tree mining, real crafting, mob combat, received damage, night,
death, near-time-limit completion and stochastic grass interaction. Assertions
check actual inventory/reward/health/terminal effects, not only initial movement.
Complete per-fixture encoded input arrays and expected outputs are retained.
All engine leaves and wrapper keys compare exactly after continuation.

The first probe failed in its test helper by comparing TensorDict key-view objects
rather than key sets. The corrected helper and original failure log are retained;
this was not a source-state mismatch. The successful run is
[replay-probe.log](semantic_recovery_evidence/craftax_source_audit/replay-probe.log).

A separate [output-boundary measurement](semantic_recovery_evidence/craftax_source_audit/output-boundary.json)
compares SWM's cached JIT output with pure upstream `get_obs` from the resulting
State. Classic pixel floats differ by at most 5.960464477539063e-08 in these
fixtures, but all eight uint8 frame comparisons are exact. Full's eight float
and frame comparisons are exact. Standalone symbolic recomputation is exact in
the recorded replay end states. These bounded results authorize implementation
and further acceptance tests; they are not a universal floating-point identity
claim. Exact uint8 source parity must continue to be measured over gameplay,
lighting, batching and device cases. No threshold may be relaxed if it fails.

The historical stable-envs atlas renderers use a separate Craftax 1.6.1 eager,
rounded-uint8 contract. SWM uses 1.5.0 JIT normalized output and truncates to uint8.
The new semantic adapters will delegate rendering to the actual upstream source;
the legacy atlas implementations and evidence remain intact. A new source baseline
must be independently configured from the pinned SWM wrapper, not inferred from
the legacy renderer's previous measurements.

## Architecture and profile decision

The existing Env/State/RenderState/Simulator interfaces can represent this model.
Use four explicit family configurations over two shared functional adapters;
keep learner Observation selection outside stable-envs. Symbolic source output
is checked through the upstream engine in tests and must never substitute for
public State. Symbolic registrations retain their source's explicit pixel-render
refusal. No Observation class, filtering layer or shared-contract change is requested.

The isolated local research profile is recorded in
[profile.json](semantic_recovery_evidence/craftax_source_audit/profile.json):
Python 3.12.11, Craftax 1.5.0, JAX/JAXlib 0.10.1, Flax 0.12.7, Gymnax 0.0.9,
NumPy 2.2.6, Torch 2.14.0, TensorDict 0.14.2; CPU backend, x64 false,
Threefry2x32, partitionable true. SWM's Python-3.10 installation instead uses
JAX/JAXlib 0.6.2 and Flax 0.10.7; no cross-version identity claim follows.
All collecting/timing probes synchronize with `block_until_ready` or NumPy materialization.
Upstream reset/step and pixel renderer are JIT paths, with an additional SWM outer
JIT. Scalar SWM makes no vmap promise. New public batching must prove per-lane
order/chunk independence without assuming vmap preserves float rounding.

Source audit and bounded replay are complete; production acceptance, comprehensive
CPU output/lifecycle tests, GPU tests and performance characterization remain pending.
No registration is promoted by this audit. Matrix remains 35 VERIFIED / 7 INCOMPLETE.
