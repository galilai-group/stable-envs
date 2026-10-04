# Rocket acceptance: 38/38 non-Craftax registrations

`swm/PFRocketLanding-v0` is accepted for its bounded public State, source
simulation, restoration and CPU/CUDA rendering/online correctness gates.
The non-Craftax matrix is **38 VERIFIED / 0 INCOMPLETE**. Including the four
historically accepted Craftax rows, the stored matrix is 42/0; Craftax was not
retested or changed in this campaign. Only Rocket's registration row is promoted.

The tested production code is `2d1875e1`; the containing checkpoint before this
promotion is `ebf82e6eeed8b2b9c7560e4dff6f9a6b5c95e95b` on `sami-dev`.
The immutable submission manifest identifies all 1,349 executed source/test files.
Oscar job **6996439** completed with exit **0:0** in **52m54s**, on gpu3004
(AMD EPYC 9554, two L40S GPUs). This is correctness evidence, not RTX 3090 timing.

## Independently verified gates

| Gate | Result |
|---|---|
| Public CPU and CUDA State/restore/render | 204 checkpoints per device, 41 visual groups each |
| Online source comparison | 64 cases / 224 frames per device; B2, T1/T8, float32/64, flight/contact/success/horizon, capture selectors |
| Different-history source suffixes | 108 cases, exact numeric continuation and source pixels |
| Persistent Variation | 90 leaf/representation cases, source reset/step/restore/suffix agreement |
| CPU unit/shared rendering checks | 67 passed, zero skipped |
| CUDA coverage and device checks | 3 passed, zero skipped; includes cuda:1 while cuda:0 is current |

The local final verifier checked current production hashes, terminal/lane records,
case coverage and all 856 raw source/candidate frame pairs. It recomputed both
480px visual-group criteria and 96px online image errors. The frozen visual
criteria and **0.001 per-frame normalized RGB MSE** threshold were unchanged.

| Comparison against source CPU renderer | Frames | Mean normalized MSE ± sample SD | Worst frame |
|---|---:|---:|---:|
| Our CPU, public 480px | 204 | 0 ± 0 | 0 |
| Our CUDA, public 480px | 204 | 0.0000462286 ± 0.0000223428 | 0.0000970718 |
| Our CPU, online 96px | 224 | 0 ± 0 | 0 |
| Our CUDA, online 96px | 224 | 0.0000645899 ± 0.0000787594 | 0.0003739141 |

Every measured frame also falls below 0.0005. This describes the recorded corpus,
not every possible State, resolution or configuration. Source rendering here is
CPU TinyRenderer; CUDA output placement is not a source GPU rendering backend.

## Implementation and accepted scope

The public `PyFlytRocketLanding` calls the migrated source task on official
PyFlyt **0.29.0** / PyBullet **3.2.7**. The tested profile uses NumPy **2.5.3**,
Gymnasium **1.3.0**, Torch **2.14.0+cu126**, and TensorDict **0.14.2**. The pinned
SWM oracle is commit `176d54532b3c0a251a4f509b74f2fdacc143a63d`; its Rocket source
SHA256 is `5ede3640ce1e2add32b6b0a6c36e190bb1130e38bb258ab182af12a6fbedee23`.
Production has no SWM runtime import and does not reimplement the physics engine.

State contains typed numeric reset recipes, exact action dtype/history, owned RNG,
Variation, lifecycle and current diagnostic fields. `set_state()` validates all
lanes, reconstructs independent official engines through replay, checks their
complete numeric diagnostics, and swaps only after every lane passes. Rejected
inputs leave the live batch unchanged. Snapshots own their storage. See the
[State schema and restore order](rocket_replay_state.md).

The GPU change is local to Rocket's triangle coverage: floating-point coverage
preserves thin fin faces that the prior rasterizer omitted, with near-plane
clipping and deterministic depth/face selection. Shared shading remains in use.
Other families retain the default rasterizer. Optional Triton is imported only
when the CUDA renderer is used, and launches select the input tensor's device.
The [investigation](rocket_lowres_coverage_checkpoint.md) retains negative controls,
failed and cancelled runs, and the original pixel witnesses.

## Limits and remaining performance work

- Dynamics remain **DELEGATED_CPU**. Acceptance does not establish a universal
  process-pipeline speedup or GPU physics.
- Replay requires the certified source/runtime math profile. A measured BLAS
  dispatch difference changed derived source fields by up to 3.55e-15; strict
  restoration correctly rejects that mismatch. Cross-profile exact replay is not
  promised, and no global BLAS setting or tolerance was changed to hide it.
- Restore costs O(recorded physical history). The registered configuration's
  source-derived capacity is 1,202 physical actions; ordinary stepping does not
  replay history. State retains the named source goal-image cache.
- Completed B1 replay costs and short saved-state L40S renderer comparisons remain
  descriptive evidence. B16/full GPU cost tables and broad source-paired online
  throughput acceptance remain open. This promotion makes no performance claim.

The legacy replacement-readiness inventory still carries separate VariationMap
classifications; its PARTIAL label is not the semantic matrix's acceptance count.
No shared State/step API or taxonomy was redesigned. The new
[testing infrastructure plan](../testing-design.md) proposes how to make these
different claims explicit and easier to maintain.

## Evidence and verification

- [Acceptance manifest](semantic_recovery_evidence/final_three_20261004/rocket/closure-v4/acceptance.json)
- [Independent final verification](semantic_recovery_evidence/final_three_20261004/rocket/closure-v4/verification/result.json)
- [Raw final gate artifacts](semantic_recovery_evidence/final_three_20261004/rocket/closure-v4/final/)
- [Scheduler terminal record](semantic_recovery_evidence/final_three_20261004/rocket/closure-v4/final-terminal.txt)
- [Frozen submission](semantic_recovery_evidence/final_three_20261004/rocket/closure-v4/submission/)

Recompute into a **new** output directory in a compatible installed environment:

```sh
PYTHONPATH=. python scripts/research_rocket_production/verify_final_acceptance.py \
  --input docs/research/semantic_recovery_evidence/final_three_20261004/rocket/closure-v4/final \
  --terminal docs/research/semantic_recovery_evidence/final_three_20261004/rocket/closure-v4/final-terminal.txt \
  --output /tmp/rocket-acceptance-reverification
```

Prior accepted rows and their artifacts were preserved. Shared CPU State/harness/
readiness/matrix checks passed 57 cases both before and after promotion. The four
pre-existing strict-docs archive-link
warnings documented in the testing review remain repository cleanup work; this
checkpoint does not claim a globally clean release build.
