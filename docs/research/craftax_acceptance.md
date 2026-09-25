# Craftax: independent CPU and GPU acceptance

All four Craftax registrations are accepted independently. The semantic-recovery
matrix is **39 VERIFIED / 3 INCOMPLETE**. Oscar job **6679082** completed on
`gpu2110` with exit `0:0` in 58m35s. Its four registration markers, 92 GPU tests,
44 saved source/candidate array cases and all 79 remote evidence hashes were
independently verified after download. No sibling result substitutes for another
registration's gate.

This report supersedes the pending-status statements in the sealed
[CPU checkpoint](craftax_recovery_checkpoint.md) and
[submission checkpoint](craftax_gpu_submission.md). Those reports, their evidence,
and the accepted Cube/Scene evidence remain unchanged.

## Checkpoints and commits

| Checkpoint | Exact SHA |
|---|---|
| Original Craftax start; Cube/Scene promotion/report containing commit | `bcf8699d9983f95e97b3c71607521a61c354247b` |
| Completed CPU checkpoint | `28e23907f887c74edb56471f4011bbb819cdda5e` |
| Starting HEAD for this retrieval/promotion, clean | `efb62d119ac94de2b4a090c88521cce60bbb4be8` |
| Tested immutable source | `b6a8242f5ff5a9cae5e9e883a15611d779c2c99c` |
| Downloaded GPU evidence and independent verification | `59ba52300cc2e9c05da2eea0a58b64c57008036f` |

Branch: `repair/environment-semantics`. Worktree:
`/private/tmp/stable-envs-semantic-recovery`. The final containing SHA is returned
in the handoff; this report's containing commit records the promotion. The final
[checkpoint inventory](semantic_recovery_evidence/craftax_gpu_6679082/checkpoint.json)
records every preceding Craftax commit, changed path and validation command.
No push occurred. No new job was needed in this retrieval sequence.

## Registration-specific acceptance

| Registration / accepted class | Engine | Presentation | CPU tests | GPU tests | Exact cases per device | Acceptance |
|---|---|---|---:|---:|---:|---|
| `swm/CraftaxClassicPixels-v1` / `CraftaxClassicPixels` | Classic | pixels | 28 | 28 | 11 | [Manifest](semantic_recovery_evidence/craftax_gpu_6679082/CraftaxClassicPixels-acceptance.json) |
| `swm/CraftaxClassicSymbolic-v1` / `CraftaxClassicSymbolic` | Classic | symbolic | 17 | 17 | 11 | [Manifest](semantic_recovery_evidence/craftax_gpu_6679082/CraftaxClassicSymbolic-acceptance.json) |
| `swm/CraftaxPixels-v1` / `CraftaxPixels` | Full | pixels | 28 | 28 | 11 | [Manifest](semantic_recovery_evidence/craftax_gpu_6679082/CraftaxPixels-acceptance.json) |
| `swm/CraftaxSymbolic-v1` / `CraftaxSymbolic` | Full | symbolic | 19 | 19 | 11 | [Manifest](semantic_recovery_evidence/craftax_gpu_6679082/CraftaxSymbolic-acceptance.json) |

Every manifest links its own CPU and GPU arrays, schemas, pytest XML/logs, device
profiles and terminal markers with SHA256 hashes. All 92 tests on each device have
zero skips/failures/errors. Saved cases cover reset seeds 0/17/42, stochastic grass,
mining, crafting, combat, damage, night, near-limit and terminal states. Tests also
cover hidden-state continuation, independent restore, atomic batch validation,
owned RNG replay/isolation, all 17 Classic and 43 full action IDs, reward,
achievements, lifecycle, output nonmutation, batching/order/chunk invariance and
unsupported-mode refusal.

GPU test durations were 728.01s, 294.24s, 1321.44s and 582.97s respectively.
JAX emitted upstream `jit(backend=...)` deprecation warnings; no test was skipped.
Craftax does not use EGL; the logged absence of MuJoCo/EGL accounting does not
replace or weaken the upstream JAX renderer checks.

## Source ownership, State and public contracts

Authority is [official Craftax 1.5.0](https://github.com/MichaelTMatthews/Craftax/tree/v1.5.0),
commit `47fff7a0b44b2842006661a2caab9ee6505fa210`, with pinned SWM oracle
`176d54532b3c0a251a4f509b74f2fdacc143a63d`. The
[source audit](craftax_source_audit.md), its
[provenance and hashes](semantic_recovery_evidence/craftax_source_audit/provenance.json),
and the corrected [CPU checkpoint](craftax_recovery_checkpoint.md) provide complete
registration/factory/wrapper paths, installed source locations and ownership tables.
All 35 upstream Python files and 202 released assets were source-aligned; wheel
SHA256 is `96826abc7d1fcf5ffbdd022100a6815b38c2b9bd22eaf43932a2252d2d7304d8`.

The adapter delegates directly to the real concrete upstream environment. Upstream
owns reset/world generation, actions, game logic, rewards, achievements, termination
and rendering. stable-envs owns the numeric codec, compatibility signature, wrapper
key, lifecycle, atomic restore and device/batch bridge. Ordinary step calls upstream
directly and never set_state. Production does not import stable-worldmodel; no Craftax
game or renderer logic was copied. There is no shared State/RenderState/Simulator
change, Observation abstraction, filtering layer or shared-contract change request.

The [complete Classic schemas](semantic_recovery_evidence/craftax_cpu/final/CraftaxClassicPixels/schema.json)
and [full schemas](semantic_recovery_evidence/craftax_cpu/final/CraftaxPixels/schema.json)
contain all 49/84 named engine leaves plus fixed parameters, static configuration,
owned keys, lifecycle and a 32-byte numeric signature. Per-leaf shapes/dtypes are
listed exactly; every public field adds leading batch dimension B.
[Full Symbolic](semantic_recovery_evidence/craftax_cpu/final/CraftaxSymbolic/schema.json)
also owns the exact source wrapper's `source_cache.last_symbolic`, float32 `[B,8268]`.
The [source proof](craftax_cached_symbolic_audit.md) establishes why this output cannot
always be recomputed bitwise from engine State. It supplements exhaustive engine
State and does not make a symbolic observation sufficient to restore dynamics.
[Classic Symbolic's complete schema](semantic_recovery_evidence/craftax_cpu/final/CraftaxClassicSymbolic/schema.json)
has no such extra cache.

`rng.wrapper_key` and `engine.state_rng` are lossless int64 `[B,2]` encodings of
legacy uint32 Threefry key words in `[0,2^32)`. The wrapper splits once per reset/step,
retaining the first key and passing the second. Explicit seed replaces the wrapper
key; unseeded reset consumes it. Full Symbolic additionally splits its supplied reset
key inside upstream before world generation. Thus equal public seeds need not produce
the same full Pixels/Symbolic world. Rendering consumes neither owned key. Batch seed
i is seed+i modulo 2^32. Typed-key or alternative PRNG claims are not made.

Restore first validates the complete batch: exact keys, shapes/dtypes, finite/range/
enum/invariant checks, fixed parameters, signature, key representation and lifecycle.
It decodes all immutable upstream lanes and allocates keys/lifecycle/cache before
atomically replacing the batch. No reset, history replay, game reconstruction or
partial lane mutation occurs. Time limit and death are source `terminated`; truncation
is false, with explicit reset required after terminal states. No auto-reset is added.

Pixel output is exact source uint8 (clip normalized source pixels, multiply by 255,
truncate), 63×63×3 Classic and 130×110×3 full. Full State and compact RenderState frames
are exactly equal; compact RenderState cannot restore dynamics. Symbolic output is
exact float32, lengths 1345/8268, including terminal output and source values exceeding
the advertised Box upper bound. Symbolic registrations refuse pixel rendering.
No universal exact normalized pixel-float identity is claimed. Legacy atlas classes,
their frozen crosswalk and historical readiness inventory are preserved separately;
`accepted_classes` and the semantic matrix identify these recovered configurations.

## Device and performance scope

CPU and CUDA semantics are accepted independently against their same-device oracle.
**CPU/GPU exactness and cross-device restoration are not claimed.** Default execution
remains `DELEGATED_CPU`; accepted CUDA execution delegates to real upstream JAX and
uses explicit host NumPy/Torch transfers. This is not a native Torch game engine.
Public execution is JIT, matching SWM, using scalar kernels per batch lane (tested B=1/3).
No public eager or vmap equivalence is claimed.

GPU profile: Python 3.12.8, Craftax 1.5.0, JAX/JAXlib 0.10.1, Flax 0.12.7,
Gymnax 0.0.9, Chex 0.1.92, NumPy 2.2.6, Torch 2.14.0+cu126, TensorDict 0.14.2;
NVIDIA RTX 3090, driver 575.57.08. x64 is false, Threefry2x32 partitionable is true,
`JAX_PLATFORMS=cuda`, preallocation false, and no XLA_FLAGS override. The complete
[installed dependency freeze](semantic_recovery_evidence/craftax_gpu_6679082/remote/evidence/freeze.txt)
and immutable hashed dependency lock are retained. CPU uses Python 3.12.11 and
Torch 2.14.0 on macOS; all other listed package versions match.

Warm GPU throughput below is **batch calls/second**, median of ten synchronized
samples after one first call in the initialized process. Multiply step calls by B
for transitions/second. get/set/render include bridge and validation cost. Symbolic
throughput times upstream recomputation, not the full-symbolic cache lookup.
Raw first-call/median/p95 timings and samples are in each GPU `result.json`.

| Configuration | B | Step/s | Reset/s | get_state/s | set_state/s | Symbolic kernel/s | Pixel render/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| CraftaxClassicPixels | 1 | 35.26 | 43.61 | 387.94 | 40.51 | 2799.27 | 26.38 |
| CraftaxClassicPixels | 3 | 21.77 | 28.52 | 308.56 | 23.97 | 1048.37 | 12.95 |
| CraftaxClassicSymbolic | 1 | 38.34 | 46.87 | 392.09 | 40.61 | 2555.54 | unsupported |
| CraftaxClassicSymbolic | 3 | 22.56 | 30.21 | 311.29 | 23.45 | 1045.98 | unsupported |
| CraftaxPixels | 1 | 27.38 | 31.43 | 263.61 | 30.26 | 2111.13 | 17.41 |
| CraftaxPixels | 3 | 15.00 | 16.86 | 202.33 | 14.95 | 755.10 | 8.13 |
| CraftaxSymbolic | 1 | 29.15 | 32.61 | 259.45 | 30.49 | 2188.80 | unsupported |
| CraftaxSymbolic | 3 | 15.50 | 17.87 | 195.73 | 15.70 | 756.02 | unsupported |

CPU throughput and synchronized fresh-process eager/JIT cold/warm diagnostics are
preserved in the [CPU checkpoint](craftax_recovery_checkpoint.md). GPU first calls
are not a controlled fresh-process cold-JIT comparison; no GPU eager comparison is
claimed. These bounded GPU rates are lower than CPU rates in the existing measurements;
no speedup claim or algorithm-changing optimization follows from acceptance.

Instance close releases owned State/key/cache references. The bounded kernel cache
and upstream texture/JIT caches have process lifetime and are not semantic State.
Scope remains fixed registered defaults: no Variation, arbitrary parameters, MPS,
alternate rendering sizes, supersampling or historical dataset snapshot sampling.

## Verification and preserved evidence

| Artifact | SHA256 |
|---|---|
| Submitted source archive | `49235ccf34ee8ba947addbf866225f1fd87ddbefa4bec68b60bc7eeb9a206834` |
| Submitted oracle archive | `c2253051f8ab584b9702e12c9fcc919f6cb8372cd0dbd1c85e270a05fc9d6b9a` |
| Downloaded GPU archive | `8f26f97ca5509606a32ca88f65224836ee70dbe67f759a80a2c1d4add6e2e932` |
| Download evidence manifest | `4922bf19d854ee6cd1f0209e22a311a134214eaf5d33bb6133fdc95d262b5655` |
| Existing CPU manifest | `d12c41b1ca918231b64776bede3f35e944d9f4287855d7adbfefdc45e20f2bfb` |

The original archive and raw logs are retained byte-for-byte. Safe extraction omitted
only pytest convenience symlinks; their names/targets are recorded, and all regular
files required by the remote manifest were extracted. The initial strict extraction
refusal and correction are documented; no acceptance data was changed.

The [independent GPU verification](semantic_recovery_evidence/craftax_gpu_6679082/verified-gpu.json)
checks exact arrays and all manifest entries, not just success strings. The
[terminal log](semantic_recovery_evidence/craftax_gpu_6679082/slurm-6679082.log) and
[SLURM accounting](semantic_recovery_evidence/craftax_gpu_6679082/remote-terminal.txt)
resolve each result. Source and test files are unchanged from the tested bundle.

Pre-promotion preservation passed for all 35 prior accepted rows, all 38 non-Craftax
rows, 2,455 original files, 170 prior artifacts, 11 terminal logs, both Cube/Scene
acceptance chains and 74 recorded source files. Post-promotion checks additionally
resolve the sealed CPU/submission/download manifests, independent acceptance
manifests and exact 39/3 matrix. Historical mutable index/matrix/map hashes are
resolved at their original commits; immutable outputs are never regenerated.

Shared State/harness/readiness/taxonomy/matrix checks, formatting/lint, maintained-file
whitespace and strict docs are recorded with exact commands/results in the final
checkpoint evidence. Raw upstream logs retain original trailing whitespace.
No production or shared code changed during GPU retrieval/promotion. No accepted
family outside Craftax changed, and no PushT or Rocket work was performed.

The remaining incomplete registrations are:

- `swm/PushT-v1`
- `swm/PushT-Discrete-v1`
- `swm/PFRocketLanding-v0`

No Craftax acceptance gate remains pending in this bounded scope. No shared-contract
change is requested. No push occurred. Stop for review after this Craftax sequence.
