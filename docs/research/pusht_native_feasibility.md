# PushT native restoration feasibility

**Outcome: EXACT_RESTORE_NOT_YET_FEASIBLE.** Continuous and discrete PushT remain
INCOMPLETE. The physical restoration prerequisite failed, so separate discrete
acceptance and production implementation did not begin. This extends the
[preserved Phase A audit](pusht_family_checkpoint.md), with its pinned ownership,
RNG/reset, Variation, reward, lifecycle and action-quantizer findings unchanged.
No production dependency, adapter, shared contract or matrix changed.

## Pinned source and native scope

SWM is `176d54532b3c0a251a4f509b74f2fdacc143a63d`. Continuous PushT owns the task,
reset, rendering and PD actions; discrete PushT subclasses it and dequantizes
16×16 polar-bin centers before the same physical step. Neither source contains a
replacement physics solver. Ordinary step performs ten authoritative Munk2D steps.

[Pymunk](https://github.com/viblo/pymunk) **7.2.0**, release commit
`3ef8f6e304fdc09e16c8ca2cfb1a3aa88a9399bc`, binds
[Munk2D](https://github.com/viblo/Munk2D) commit
`ade7ed72849e60289eefb7a41e79ae6322fefaf3` (2.0.1 runtime). Source paths and
hashes are in `build-identity.json`; original SWM and binding hashes remain in
`pusht_phase_a/audit.json`. Existing public properties do not expose a validated
complete numeric contact graph. The prior `Space.copy()` attempt failed through
the SWM pygame callback graph; it is not an exhaustive-State baseline.

The new `pusht_fields.c` builds an isolated accessor library against the pinned
headers. It does not replace or modify Munk2D stepping or the installed Pymunk.
Private addresses are used only inside the research call boundary, never stored
as snapshot leaves. Body fields are copied to float64 arrays:

- center of gravity, position, velocity, force (two coordinates each);
- angle, angular velocity and torque;
- cached transform's six components;
- sleeping idle time;
- bias velocity's two components and bias angular velocity.

The last three bias scalars are consumed by position integration before velocity
damping and are therefore future-relevant even with the task's damping zero.
A second accessor exports the active arbiter order as stable shape-hash pairs and
matches/reorders existing target arbiters. This is a diagnostic 64-entry capacity,
not a finalized public schema or a proven bound for arbitrary Variation.

The prototype extends the previous numeric reconstruction, which restores
body/task/RNG values, space counters, inspected arbiter/contact fields and warm
start impulses after a target-only detection step. It does not replay source
history. That detection procedure has not been proved atomic or sufficient and
is not proposed as public `set_state`.

## Controlled witnesses

Each case runs five variants: independent same-history positive control, previous
numeric restoration, added body bias, all 21 body fields, and all body fields plus
active-arbiter order. Every restoration target is independently constructed and
has 16 divergent actions. Source continuation is untouched, with 32 prescribed
public actions, exact output and future candidate/body-cache comparisons, plus
immediate pixels. All 55 records completed without collector exceptions.

| Case | Previous numeric restoration | Add bias / full body / active order |
|---|---|---|
| Free motion | Exact | Exact |
| First contact | Exact | Exact |
| Sustained contact | Exact | Exact |
| Separation and re-contact | Fails | Exact bounded continuation |
| Corner approach/contact | Fails | Exact bounded continuation |
| Edge approach/contact | Fails | Exact bounded continuation |
| Resting contact | Exact | Exact |
| Success fixture | Exact | Exact bounded continuation |
| Preserved isolated wall contact | Exact | Exact bounded continuation |
| Preserved isolated agent/block contact | Exact | Exact bounded continuation |
| Preserved combined contact | Fails | **Still fails** |

All eleven same-history independent controls are exact in outputs and future
native body fields. The corner/edge labels denote approach fixtures, not proof of
all contact geometry classes. Histories and contact counts are retained. The
success fixture uses a source-supported explicit goal equal to the reset state;
it does not claim complete threshold-neighbour, TimeLimit, RNG or Variation
acceptance. These would follow physical feasibility, which failed.

For the strongest combined-contact attempt, immediate candidate/body fields,
inspected contact values, active order and rendered pixels agree. Nevertheless:

- maximum output difference over 32 actions: **0.04628662980150011**;
- maximum reward difference: **0.04140655071182664**;
- termination/truncation agree on this witness, which does not excuse reward or physics drift;
- future native body fields and candidate State diverge.

The previous numeric reconstruction's maximum output difference is
0.04810707691274274. Restoring more immediate fields reduces a failure; it does
not establish exact different-history continuation. Full arrays and frames are
retained in `final/*.npz`, with no object/pickle payload.

## Remaining engine state

Pinned `cpSpaceStep.c` empties the active arbiter list on the next step, integrates
positions, reindexes dynamic shapes and rebuilds contacts using the spatial
index. It then handles sleeping components, ages cached arbiters, computes
pre-step terms, applies cached impulses and iterates the solver in arbiter order.
Consequently copying the current active order cannot preserve the next order if
broadphase traversal differs.

`cpBBTree.c` retains cached/fattened AABBs, binary node topology, leaf stamps,
threaded overlap pairs and collision IDs. The prototype does not export/import
this complete state. Cached-arbiter membership/order, persistent contacts and
pair collision IDs, shape caches, body sleeping/component links, and contact
buffer/space sequencing remain closure obligations. Constraint impulses also
matter for configurations containing constraints; the fixed default fixture has
none. Sleeping is disabled by the default infinite threshold, but that does not
justify dropping activation fields from arbitrary compatible configurations.
The original Python task, goal/render-goal, action cache, lifecycle and RNG audit
is preserved rather than rediscovered.

This source trace identifies concrete remaining candidates; it does **not** prove
which missing field explains every residual error. For the default topology of
four walls, one agent shape and two block polygons, at most 21 unordered distinct
shape pairs and two points per arbiter suggest a 42-contact numeric allocation.
A production bound still requires validating shape IDs, cached membership,
broadphase nodes/pairs and supported Variation topology. That work is not
complete. No public exhaustive schema is finalized from the diagnostic capacity.

## Packaging, licensing and measured cost

The complete native addition is `scripts/research_native/pusht_fields.c`; no
engine source patch or large fork is vendored. It was compiled with Apple clang
against exact private Munk2D struct layouts, exercised only on macOS arm64 with
CPython 3.12 and the accepted Pymunk 7.2.0 profile. `build-identity.json` records
the binary and source hashes. The library is not committed or installed into a
production environment.

A distributable extension would need the same precision/options/layout as each
Pymunk wheel, robust ABI checks, Python/platform wheels for all claimed systems,
Linux x86_64 and macOS arm64 tests, and safe validated/atomic import. A typed
upstream API would reduce the burden compared with compiling against private
layouts. Supporting full broadphase and contact graphs substantially increases
maintenance. Neither a minimal sufficient patch nor an engine fork requirement
has been proved. Pymunk and Munk2D license notices are retained alongside the
results; their permissive MIT terms require preservation of copyright/license
notices in distributions. No upstream acceptance or packaging portability is
claimed.

At the preserved combined-contact checkpoint, 10 warmups + 200 Apple M5 samples:

| Research subset operation | Median µs |
|---|---:|
| Export two bodies' 21 fields each | 3.833 |
| Import those body fields | 4.000 |
| Export active contact order | 2.083 |
| Restore existing active contact order | 1.667 |

Raw samples/p95/throughput and numeric subset bytes are in `performance/result.json`.
These are **NATIVE_PARTIAL** diagnostic lower bounds, not a cost for full cache
reconstruction or exhaustive State. The authoritative task has no sufficient
public native snapshot interface; no `NATIVE_EQUIVALENT` claim is made.

## Reproduction and review boundary

```sh
clang -shared -fPIC -O2 -I /private/tmp/pusht-pymunk-source-7.2.0/Munk2D/include scripts/research_native/pusht_fields.c -o /private/tmp/pusht-native-research/fields.dylib
PYTHONPATH=.:/Users/samib/stable-worldmodel SDL_VIDEODRIVER=dummy /private/tmp/pusht-authority-swm-lock/bin/python scripts/research_native/pusht_native_probe.py --library /private/tmp/pusht-native-research/fields.dylib --output /private/tmp/pusht-new-probe
PYTHONPATH=.:/Users/samib/stable-worldmodel SDL_VIDEODRIVER=dummy /private/tmp/pusht-authority-swm-lock/bin/python scripts/research_native/native_cost.py --engine pusht --library /private/tmp/pusht-native-research/fields.dylib --output /private/tmp/pusht-new-cost
.venv/bin/python scripts/research_native/verify_native_evidence.py --output /private/tmp/new-native-verification.json
```

The verifier recomputes all 55 output/native/frame comparisons and the Rocket
baseline fidelity check from retained arrays. Earlier development runs with a
collector observation-dict mistake are not engine evidence. Final result JSON
identifies all completed cases. Evidence is in
`semantic_recovery_evidence/pusht_native_feasibility/` with a SHA256 seal.

**Stop before any patched production dependency.** Both registrations stay
INCOMPLETE; no shared-contract change is requested and no promotion follows from
immediate numeric equality.
