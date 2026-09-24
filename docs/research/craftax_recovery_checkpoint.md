# Craftax recovery: CPU implementation checkpoint

The four Craftax configurations delegate to the real upstream engines. GPU acceptance
is pending Oscar SSH authentication; no registration is promoted. The matrix remains
**35 VERIFIED / 7 INCOMPLETE**. PushT and Rocket are outside this sequence.

## Checkpoint and authority

Starting clean HEAD and Cube/Scene containing promotion/report commit:
`bcf8699d9983f95e97b3c71607521a61c354247b`. Branch
`repair/environment-semantics`, worktree `/private/tmp/stable-envs-semantic-recovery`.
The exact ending containing commit is returned in the final handoff; the tested source commit
is recorded in the immutable GPU bundle metadata.

- `f5b3b1a2232bcb7447349a2ba5ceec202e868109`: source/dependency/candidate-State audit and replay.
- `692d7562d950f3496b8784e851f49ddc0617f203`: delegated production semantic adapter.
- `067ea24e`: retained CPU development evidence, including the two corrected harness/bridge failures.
- `8feb5960`: upstream renderer integration, profile signature, tests and GPU runner.
- `f8c0a37b`: terminal symbolic oracle JIT boundary and preservation verifier.
- `79d1e4ac`: measured cached-output mismatch and source-only replay proof.
- `2953735e`: full-symbolic numeric source-output cache; exact reset contract recovered.
- `b6a8242f`: final collector preserves both source and candidate arrays before assertions.

Full commit IDs through the tested source and subsequent evidence commits are indexed
in the final command/checkpoint records. No commit was pushed.

The sealed [initial source audit](craftax_source_audit.md), together with the corrections
below and the cached-output supplement, records the registration/factory/
wrapper trace, method ownership table, source-read inventory, lifecycle, action and
PRNG analysis. [Provenance](semantic_recovery_evidence/craftax_source_audit/provenance.json)
records exact source paths, hashes and source-definition locations. Authority is
[official Craftax 1.5.0](https://github.com/MichaelTMatthews/Craftax/tree/v1.5.0), commit
`47fff7a0b44b2842006661a2caab9ee6505fa210`; the pinned SWM oracle is
`176d54532b3c0a251a4f509b74f2fdacc143a63d`. All 35 upstream Python files match
both installations and the official tag; all 202 released assets match. Wheel SHA256:
`96826abc7d1fcf5ffbdd022100a6815b38c2b9bd22eaf43932a2252d2d7304d8`.

| Historical registration | New canonical stable-envs class | Engine | Presentation |
|---|---|---|---|
| `swm/CraftaxClassicPixels-v1` | `CraftaxClassicPixels` | Classic | upstream pixels, uint8 frame 63×63×3 |
| `swm/CraftaxClassicSymbolic-v1` | `CraftaxClassicSymbolic` | Classic | upstream float32 symbolic vector, 1345 |
| `swm/CraftaxPixels-v1` | `CraftaxPixels` | Full | upstream pixels, uint8 frame 130×110×3 |
| `swm/CraftaxSymbolic-v1` | `CraftaxSymbolic` | Full | upstream float32 symbolic vector, 8268 |

The initial audit's statement that only `get_obs` differs was too broad. Both engines
share game logic and world-generation implementation, but **full Symbolic's upstream
`reset_env` splits the supplied key again and gives the second result to world
generation**; full Pixels and both Classic classes pass it directly. Therefore full
Pixels/Symbolic do not map the same public seed to the same world. Each concrete
upstream class has been delegated from the first implementation, so this source
behavior was already preserved and independently tested. The final comparison retains
all differences; the component proof checks the transformed-key relationship and
six-step continuation from an identical engine State. No production change follows
from this audit correction. The original sealed audit is retained unchanged.

Existing `CraftaxClassic`/`CraftaxFull` atlas classes, their renderer contracts and
frozen crosswalk are preserved. The matrix separately identifies the new candidates;
none is an accepted replacement until its own GPU gate passes.

## State, restoration and delegation

The complete final schemas are the per-registration `schema.json` artifacts linked
below. Each listed shape is per lane, with an additional leading public dimension
`[B]`. Source-semantic names appear under `engine`: all 49 Classic or 84 full engine
leaves, including unseen maps/levels, entities, inventory, cooldowns, intrinsic
accumulators, achievements, procedural state and RNG. Numeric `params` and `static`
leaves record the exact fixed source configuration. Four structural None values
remain fixed signature data. `meta.signature` is 32 uint8 bytes over engine,
presentation, schema, profile, device backend, PRNG and complete fixed parameters.
`lifecycle` contains last reward (float32), terminated and truncated (bool).
Full Symbolic alone also owns `source_cache.last_symbolic`, float32 `[B,8268]`: the
exact upstream return cached by the SWM wrapper. Its fused reset output cannot always
be recomputed bitwise from engine State. The [supplemental source audit](craftax_cached_symbolic_audit.md)
records the failure and the independent source replay proof that preceded this bounded
correction. This cache is a named numeric adjunct to complete State, never a substitute
for engine data or a new Observation abstraction. Its shape/dtype/finiteness and
signature validate atomically; retained values are restored exactly without an
approximate recomputation check.

Final PRNG encoding supersedes the audit's candidate direct-uint32 encoding:
`rng.wrapper_key` and `engine.state_rng` are each **int64 `[B,2]`, range [0,2^32)**.
They losslessly encode the measured legacy uint32 Threefry words. Direct Torch uint32
failed TensorDict clone; the failure and correction are retained. Decode casts to
upstream uint32. Every reset/step splits the wrapper key into retained first key and
supplied second key, exactly as SWM. Explicit reset seed replaces that key first;
unseeded reset consumes the current one. Full Symbolic then performs the additional
upstream reset split described above; it owns no extra persistent key. Rendering reads engine RNG functionally
and consumes neither key. Seeds are validated integers in [0,2^32); a public batch
uses seed+i modulo 2^32 for independent lanes.

Restore validates all keys, shapes, dtypes, batch, signature, fixed parameters,
finite/range/enum/source-invariant constraints, key ranges and lifecycle agreement.
It then decodes every lane into immutable upstream dataclasses, allocates keys,
lifecycle arrays and the full-symbolic source cache, and swaps the complete batch only after success. It performs no
reset, world generation, history replay or simulator reconstruction. Ordinary step
calls upstream directly and never calls set_state. State and input snapshots own
storage. Symbolic vectors and compact RenderState explicitly cannot restore dynamics.

Classic uses all 17 source action IDs; full uses all 43. Public actions require
integer IDs per lane (int32/int64 accepted), refusing bool/fractional/out-of-range
values atomically. Upstream owns every action effect, world-generation operation,
reward, achievement and terminal calculation. Both source time limits are
`terminated`, not `truncated`; no auto-reset is introduced. Stepping a terminal lane
requires explicit reset. No Variation or reset options are exposed for these fixed
registrations; upstream experimental parameters are not silently promoted.

Production imports upstream Craftax, never stable-worldmodel. No game logic,
world-generation algorithm, combat/crafting/reward calculation or pixel arithmetic
was copied. No Observation class, filtering layer, process-global monkeypatch or
shared State/RenderState/Simulator contract change was introduced or requested.

## Output, execution and resource scope

Pixel RenderState holds a distinct numeric signature and the named source-read
subset of engine fields. It excludes collision-only data, entity health/cooldowns,
wrapper RNG and lifecycle. Full State and compact RenderState produce identical
frames through the unchanged upstream JIT renderer. The SWM frame conversion is
clip to [0,1], multiply by 255, then truncate to uint8. In Classic, standalone
normalized floats can differ from SWM's cached step output by 5.96e-8; the required
uint8 comparison is exact over the recorded gates. No universal float identity is
claimed. The legacy 1.6.1 eager rounded atlas baseline is a different preserved
contract, not silently relabeled as this 1.5.0 SWM contract.

Symbolic selection remains upstream/test-side. Its exact float32 vector, including
terminal output, is compared against pinned SWM. Full symbolic values can exceed
SWM's advertised [0,1] Box (books/enchantments); source values are preserved without
clamping. Symbolic registrations explicitly refuse pixel rendering and Frame specs.

Public execution is JIT, matching SWM. Batch execution uses scalar upstream kernels
per lane; no untested vmap equivalence is assumed. CPU timing synchronizes JAX
results with block_until_ready or host materialization. CPU and GPU compatibility
signatures differ; cross-device restoration/equality is not claimed. The bridge
materializes named numeric tensors through host NumPy; even on GPU it is a delegated
JAX engine with explicit transfer cost, not a native Torch implementation.

Python 3.12 is required. CPU profile: Python 3.12.11, Craftax 1.5.0, JAX/JAXlib 0.10.1,
Flax 0.12.7, Gymnax 0.0.9, Chex 0.1.92, NumPy 2.2.6, Gymnasium 1.2.3,
Torch 2.14.0, TensorDict 0.14.2. PRNG is Threefry2x32, partitionable true, x64 false.
The GPU lock additionally pins CUDA-enabled JAX and Torch 2.14.0+cu126, with complete
package hashes for Linux manylinux_2_34/Python 3.12. Actual GPU driver/device/profile
must be recorded by the pending job. No GPU or CPU/GPU exactness result exists yet.

Instance close releases owned State/key/cache references. The bounded eight-entry
engine/presentation/backend cache and upstream texture/JIT caches remain process
lifetime runtime resources, excluded from semantic State. Only CPU/CUDA are exposed;
MPS, alternate sizes, supersampling, renderer backends, arbitrary snapshot sampling
and unsupported source configuration changes refuse explicitly.

## Acceptance, measurements and evidence

Final commands, per-registration test counts, exact arrays, timings, hashes and
preservation results are indexed in the checkpoint evidence. Earlier development
runs are clearly labeled in `development.json`, not substituted for final acceptance.

**92 final tests passed, zero skips/failures/errors**, in 371.45 seconds.
All **44 saved source/candidate array cases** were independently rechecked after collection.

| Registration | Tests | Saved exact cases | CPU result | GPU result | Complete schema / arrays |
|---|---:|---:|---|---|---|
| `CraftaxClassicPixels` | 28 | 11 | PASS | Not submitted | [Schema](semantic_recovery_evidence/craftax_cpu/final/CraftaxClassicPixels/schema.json), [result](semantic_recovery_evidence/craftax_cpu/final/CraftaxClassicPixels/result.json) |
| `CraftaxClassicSymbolic` | 17 | 11 | PASS | Not submitted | [Schema](semantic_recovery_evidence/craftax_cpu/final/CraftaxClassicSymbolic/schema.json), [result](semantic_recovery_evidence/craftax_cpu/final/CraftaxClassicSymbolic/result.json) |
| `CraftaxPixels` | 28 | 11 | PASS | Not submitted | [Schema](semantic_recovery_evidence/craftax_cpu/final/CraftaxPixels/schema.json), [result](semantic_recovery_evidence/craftax_cpu/final/CraftaxPixels/result.json) |
| `CraftaxSymbolic` | 19 | 11 | PASS | Not submitted | [Schema](semantic_recovery_evidence/craftax_cpu/final/CraftaxSymbolic/schema.json), [result](semantic_recovery_evidence/craftax_cpu/final/CraftaxSymbolic/result.json) |

Each registration has an independent `*-cpu-acceptance.json` artifact manifest.
The [acceptance summary](semantic_recovery_evidence/craftax_cpu/acceptance-summary.json)
and [final pytest XML](semantic_recovery_evidence/craftax_cpu/final-acceptance.xml)
record counts and scope. [Component proof](semantic_recovery_evidence/craftax_cpu/component-proof/result.json)
passes both engines from identical State. The [same-seed comparison](semantic_recovery_evidence/craftax_cpu/cross-presentation-engine-comparison.json)
preserves full-family world differences caused by its source reset-key policy.

Warm throughput below is **batch calls per second**, median of ten timed samples
after a first timed call in an already initialized process. Multiply step calls by
B for environment transitions/second. Raw samples, p95 and first-call times are in
each result JSON. Symbolic throughput measures the upstream recomputation kernel,
not the full-symbolic retained-cache lookup. Pixel rendering includes validation and
host/Torch bridging. These are bounded CPU measurements, not GPU benchmarks.

| Configuration | B | Step/s | Reset/s | get_state/s | set_state/s | Symbolic kernel/s | Pixel render/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| CraftaxClassicPixels | 1 | 186.7 | 182.6 | 2670.4 | 185.3 | 26272.6 | 132.9 |
| CraftaxClassicPixels | 3 | 163.6 | 142.3 | 2000.8 | 135.6 | 13773.3 | 87.9 |
| CraftaxClassicSymbolic | 1 | 190.7 | 180.2 | 2681.1 | 184.9 | 19223.0 | unsupported |
| CraftaxClassicSymbolic | 3 | 178.1 | 146.0 | 1986.6 | 139.1 | 13706.4 | unsupported |
| CraftaxPixels | 1 | 157.0 | 119.3 | 1780.9 | 160.8 | 9693.1 | 95.9 |
| CraftaxPixels | 3 | 106.2 | 62.4 | 1226.2 | 100.0 | 5201.0 | 53.2 |
| CraftaxSymbolic | 1 | 173.5 | 128.3 | 1703.6 | 157.0 | 9650.2 | unsupported |
| CraftaxSymbolic | 3 | 140.9 | 72.0 | 1204.9 | 101.3 | 5079.9 | unsupported |

Separate fresh-process raw-source eager/JIT timing uses two calls per operation.
First includes compilation/runtime warmup; second is a single warm diagnostic sample.
Both modes are synchronized. Public/SWM execution remains JIT only.

| Engine | Mode | Reset first / second (ms) | Step first / second (ms) |
|---|---|---:|---:|
| Classic | jit | 559.341 / 0.804 | 1125.290 / 0.371 |
| Classic | eager | 4017.576 / 53.007 | 3135.493 / 342.313 |
| Full | jit | 1959.656 / 3.103 | 2943.727 / 0.710 |
| Full | eager | 12837.526 / 593.647 | 5156.945 / 598.780 |

Raw eager/JIT engine leaves match in this bounded experiment. Full symbolic step
outputs differ in five values by up to 5.96e-8; [both arrays and their comparison](semantic_recovery_evidence/craftax_cpu/eager-jit-array-comparison.json)
are retained. No universal eager/JIT or cross-device identity follows.

Shared State/harness/readiness/matrix/repository/taxonomy checks: **131 passed,
11 optional-dependency skips**, matching the previous accepted checkpoint.
[Preservation](semantic_recovery_evidence/craftax_cpu/preservation-final.json) verifies
2,455 unchanged starting files, all 35 accepted rows and all 38 non-Craftax rows,
170 prior artifacts, 11 terminal logs, both Cube/Scene acceptance chains, their
173-entry GPU and 193-entry final manifests, the 51-entry source-audit manifest,
and all 74 recorded source files. No old mutable evidence script was rerun.

The [command index](semantic_recovery_evidence/craftax_cpu/commands.json),
[changed-file inventory](semantic_recovery_evidence/craftax_cpu/changed-files.json),
and [final evidence manifest](semantic_recovery_evidence/craftax_cpu/SHA256SUMS)
provide the remaining reproducibility details.


## GPU blocker and resumption

Oscar was reachable but rejected fresh SSH authentication with
`Permission denied (keyboard-interactive)` when reading its required workflow skills.
No Craftax upload, SLURM submission or GPU execution occurred. Reconnect and keep the
control connection open:

```sh
ssh -M -S /private/tmp/stable-envs-oscar-control sboughan@sshcampus.ccv.brown.edu
```

The final immutable local bundle is `/private/tmp/stable-envs-craftax-b6a8242f-gpu`; source commit
`b6a8242f5ff5a9cae5e9e883a15611d779c2c99c`. It contains 2863 archive members,
including every required fixture, source/test runner and hashed dependency lock.
[Bundle metadata](semantic_recovery_evidence/craftax_gpu_prepared/bundle.json) records
all expected files, tests and terminal markers. Source archive SHA256:
`49235ccf34ee8ba947addbf866225f1fd87ddbefa4bec68b60bc7eeb9a206834`; oracle archive SHA256:
`c2253051f8ab584b9702e12c9fcc919f6cb8372cd0dbd1c85e270a05fc9d6b9a`. The earlier pre-cache bundle is explicitly
superseded and remains preserved; it must not be submitted.

After authentication, read the six current `~/.claude/skills/` Oscar skills, verify
and upload the prepared immutable bundle, and submit its runner. The runner uses one
nonpreemptible `gpu`/`norm-gpu` allocation, one RTX 3090 or L40S, eight CPUs, 64 GB,
60 minutes, no requeue, and an isolated scratch virtualenv. Installation uses the
fully hashed lock with explicit `--torch-backend cu126`; no heavy login-node work.
Each registration must have its expected pytest count with zero skips/failures/errors
and an independent exact-array gate before its success marker is emitted. Preserve
all raw remote evidence and failed cases. Download and independently validate hashes,
arrays, markers and actual terminal state before any registration-specific promotion.

All 35 previous accepted registrations remain intact. The incomplete set remains
the four Craftax registrations plus `swm/PushT-v1`, `swm/PushT-Discrete-v1`, and
`swm/PFRocketLanding-v0`. No push occurred. Stop for review after this sequence.
