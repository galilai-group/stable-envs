# State performance and native restoration checkpoint

The final matrix remains **39 VERIFIED / 3 INCOMPLETE**. All 39 accepted
registrations now have individual descriptive State-operation measurements.
Rocket and PushT native prototypes remain **EXACT_RESTORE_NOT_YET_FEASIBLE**:
preserved different-history contact continuations still diverge. No production
implementation, dependency, shared contract, acceptance row or promotion changed.
No shared-contract change is requested. **No push occurred.**

The three incomplete registrations are `swm/PFRocketLanding-v0`, `swm/PushT-v1`
and `swm/PushT-Discrete-v1`. Discrete PushT's independent acceptance is deferred
because the shared continuous physical-restoration prerequisite failed.

## Git checkpoint and preservation

- Starting HEAD / Rocket audit: `13e51037cdb4afb9ce31aba3a4837ccf5c95adaf`.
- Branch: `repair/environment-semantics`.
- Worktree: `/private/tmp/stable-envs-semantic-recovery`; the user's separate
  `/Users/samib/stable-envs` worktree was not changed.
- Starting worktree was clean. The ending checkpoint is the commit first adding
  this report, titled `docs: close State performance and native feasibility checkpoint`.
  Its exact SHA and clean status are supplied in the final handoff. Resolve it
  locally with `git log --diff-filter=A --format=%H -- docs/research/state_performance_checkpoint.md`.
- Benchmark evidence containing commit: `0d3cf34f149c62407b7356e87d19493aa4f4fd90`.
- All created commits are listed below; the containing commit cannot include its
  own SHA without changing that SHA.

The starting record resolved every accepted registration's manifests, baselines,
terminal evidence and covering implementation. Final preservation checks verify
208 distinct accepted artifact hashes and 112 production-file hashes, and byte
identity of the matrix and implementation map. The independent Rocket-audit
verifier also checks 2,851 prior files, six Craftax/Cube/Scene acceptance manifests,
the 102-entry Craftax containing-commit seal and 87-entry download seal, 170 earlier
artifact hashes and 11 terminal logs. Production remains identical to the last
accepted Craftax checkpoint `badc193af3ee450ac486a75270bc88f47d9a1816`.

The Rocket Ruff bookkeeping mistake is corrected in
`semantic_recovery_evidence/state_performance_start/correction.json`: both commands
actually exited zero, with `All checks passed!` and `3 files already formatted`.
The erroneous long marker came from matching `verify_rocket_audit` inside Ruff's
arguments. Superseding metadata and raw outputs are retained; the original audit
and its seal were not overwritten.

## Measurements and interpretation

The [full report](semantic_recovery_evidence/state_performance_report/report.md)
has family and per-registration tables, State payload/NPZ bytes, step/get/set and
branch costs, oracle/native comparisons, GPU rendering, cold starts and a chart.
Its [machine-readable summary](semantic_recovery_evidence/state_performance_report/summary.json)
includes all operations, p95, operations/second, raw samples, versions, hardware,
source hashes and explicit unsupported reasons. Exact report reproduction is in
the [report README](semantic_recovery_evidence/state_performance_report/README.md);
[harness documentation](state_operation_benchmarks.md) explains new coverage.

Coverage is 39 independent CPU registration records at B1 and B16, except B4 for
Cube/Scene. Matching GPU records provide all 37 available renderers, and public
GPU dynamics/State measurements for all four Craftax registrations. Other families
delegate dynamics to CPU engines; a GPU renderer does not imply GPU dynamics.
Symbolic Craftax has no renderer. B64 is deferred for runtime, not emulated or
declared unsupported. Every warm measured operation has at least ten warmups and
fifty samples; fast operations adapt up to 1,000 samples. Asynchronous work is
synchronized and outputs retained. Sixteen separate fresh-process Craftax cases
cover CPU/GPU × public/oracle × four registrations.

| Family | Registrations | B1 numeric bytes | Get/step range | Set/step range | Closest native class |
|---|---:|---:|---:|---:|---|
| Classic control | 5 | 104–129 | 0.560–0.714 | 0.558–0.630 | NONE |
| DMC | 12 | 5,058–5,690 | 0.260–0.603 | 0.328–0.677 | NATIVE_PARTIAL |
| Craftax | 4 | 21,151–306,516 | 0.083–0.143 | 1.016–1.205 | NATIVE_PARTIAL |
| Fetch | 12 | 560–712 | 0.151–0.204 | 0.231–0.301 | NATIVE_PARTIAL |
| OGBench | 3 | 5,145–10,485 | 0.382–0.608 | 0.920–146.427 | NATIVE_PARTIAL |
| Navigation | 3 | 16–33 | 0.210–0.425 | 0.189–32.680 | NONE |

These ranges are CPU B1 medians on Apple M5, spanning registrations rather than
independent backend implementations. The largest serialized B1 State is full
CraftaxSymbolic: 337,490 NPZ bytes (306,516 numeric bytes); B16 is 4,935,230 NPZ bytes.
Cube and Scene restore take 289.755 and 337.373 ms, respectively, or 141.92× and
146.43× their steps. Their branch cycles take 310.715 and 405.181 ms. SimplePointMaze
restore is 10.881 ms, 32.68× step, with a 13.438 ms branch. These remain observations
for post-42/42 optimization; no fast paths were added.

MuJoCo integration-state APIs are partial because wrapper/task/RNG/configuration
and output caches remain outside them. DMC's explicit integration-signature
supplement is used as its closest measured subset; the narrower default-signature
measurements remain available. Craftax's explicit deep engine-pytree copy/install
omits wrapper ownership and output state. **No NATIVE_EQUIVALENT baseline was
established; no public/native ratio is like-for-like exhaustive State.** Rocket's
opaque snapshots are `OPAQUE_SAME_PROCESS_NON_EQUIVALENT`; PushT's research body
and contact-order accessors are `NATIVE_PARTIAL`. Sources, fields and hashes are
in `state_performance_final/native-baseline-audit.json` and the native reports.

Oscar jobs were 6693308 (failed profile isolation, preserved), 6694195 (corrected
18-case retry, COMPLETED 0:0), and 6694865 (eight cold cases, COMPLETED 0:0).
Twenty-one valid normal-profile records from the initial job remain in use.
No duplicate active jobs were submitted and no task job remains active. Two CPU
dependency-profile failures were also retained and replaced by fresh successful
OGBMaze/SimplePointMaze runs. Submission scripts, exact commands, source archive
hashes, terminal logs and downloaded manifests are committed and verified.

## Native feasibility outcomes

[Rocket research](rocket_native_feasibility.md) isolates a useful cached-link
restore fix, making thrust and independent file-snapshot continuation exact.
Skipping importer reconstruction alone fails. Adding collider/interpolation,
velocity, activation and empty-manifold synchronization still fails contacts.
The best retained empty-manifold memory route has near-contact reward error
0.1697551170328957 and crash reward error 4.312914337371227, with crash truncation
differences on two steps. The serializer does include contact manifolds; complete
importer/broadphase/solver closure and a safe fixed bound remain unresolved.
Fresh-link reads also change ordinary oracle trajectories, so they are a semantic
migration diagnostic, not a recovery fix.

Exact native diffs:

- [Bullet v1](semantic_recovery_evidence/rocket_native_feasibility/cache-extension-v1.patch)
- [Bullet v2](semantic_recovery_evidence/rocket_native_feasibility/cache-extension-v2.patch)
- [PushT accessor addition](semantic_recovery_evidence/state_performance_final/pusht-fields-addition.patch)

[PushT research](pusht_native_feasibility.md) restores bias velocities and fixes
bounded re-contact/corner/edge continuations. The strongest combined-contact
witness still has maximum output error 0.04628662980150011 and reward error
0.04140655071182664, despite exact immediate numeric fields, inspected contacts,
arbiter order and pixels. Broadphase tree/pair persistence and full contact graph
closure remain unresolved. Eleven independent same-history controls pass.

Both experiments used isolated macOS arm64/CPython 3.12 research builds, with
source hashes, exact field lists, licenses, compile logs and performance samples.
PyBullet is 3.2.7 with bundled Bullet header version 326; Pymunk is 7.2.0 with
Munk2D 2.0.1. Neither prototype changes ordinary engine stepping. Rocket's text
transport/import hook is not production-safe or atomic; PushT accesses private
layouts. Supported wheels, Linux/other architectures, ABI checks, robust bounds,
atomic validation and ongoing internal-engine maintenance would require separate
work. No portable wheel or minimal sufficient extension is claimed. No patched
production dependency was introduced or proposed for acceptance.

## Validation and limitations

Shared regression selection: **134 passed, 11 skipped, 1 warning** (optional
`ale-py` absent). It covers State contracts, semantic harness, readiness,
matrix/taxonomy, repository contracts, Variation, manipulation acceptance and
benchmark accounting. Raw command arrays, actual exit codes and logs are in
`semantic_recovery_evidence/state_performance_final/commands.json`.

Native evidence verification independently checks 5,096 original Rocket baseline
array-leaf comparisons, 550 new Rocket array pairs, all 55 PushT records and their
11 independent controls. This reproduces the counterexamples; it is not a passing
physical-restoration gate. Exact native patch reconstruction also passed.
Final Ruff, formatting, strict documentation, evidence-seal/source-hash checks
and changed-file inventory validation are recorded in
`semantic_recovery_evidence/state_performance_final/final-checks.json`.

Raw upstream Pymunk license CRLF, exact Bullet patch contexts, native GPU status
output and the generated SVG retain their original whitespace. Whitespace
validation excludes those seven enumerated artifacts and its own raw diagnostic
logs; `state_performance_final/whitespace-exceptions.json` records the exact paths
and reasons. Source code and hand-written documentation pass the final check.
The unfiltered diagnostics remain available.

Timing is descriptive on the stated hardware/profiles. Thirty-eight CPU operation
cells have coefficient of variation above 0.25; all samples and p95 values remain.
Cold results are single fresh-process observations, not distributions. Classic
Craftax's second step after first restoration still has initialization-scale
cost on both CPU and GPU; only repeated warm results represent steady state.
Pixel Craftax intrinsically constructs observations in delegated step, so that
required output cost is included, separately from standalone rendering.
No contact, reward, lifecycle or independent-instance equality was weakened.

## Complete changed-file inventory and commits

The [complete path inventory](semantic_recovery_evidence/state_performance_final/changed-files.txt)
contains every added or modified path relative to the starting SHA, including
raw logs normally excluded by generic Git ignore rules. Evidence-directory seals
and final closure metadata accompany it. No production paths changed.

- `009065f0b7838dc94e6a8d408db25ea8d60a36ec` — docs: establish State benchmark checkpoint and correct Rocket Ruff metadata
- `8d153702708a5b481b3809203206e993eabb1060` — bench: measure per-registration State operations and scoped native baselines
- `3f1c4e67de79aff52bfe89bbb5a096362a4d2580` — research(rocket): isolate native cache fixes and preserve contact blockers
- `0a13971759aa3d675c4af61b29a7bc146cfa6417` — research(pusht): isolate bias cache effects and retain combined-contact failure
- `8dc26634406bbff836d0a73079b3921b54d57378` — bench: report registration costs and isolate fresh Craftax compilation timing
- `0cad0108527175ce8bf592922b19882a7ac15b4c` — bench: select OGBench profile and report audited native integration baselines
- `9d83fe9758041313f01d47c2fc9a3211d5d2e0d2` — bench: describe actual requested batches in deferred B64 metadata
- `f3acc01370689dc0059c55ff8a794c7658c64a3c` — bench: record cached-goal and renderer dependency versions
- `0d3cf34f149c62407b7356e87d19493aa4f4fd90` — bench: preserve 39-registration CPU and GPU State performance evidence
- Containing commit — `docs: close State performance and native feasibility checkpoint`;
  includes this final handoff, verification and retained raw native logs.
