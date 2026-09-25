# Rendered State performance, clone and replay checkpoint

The acceptance matrix remains **39 VERIFIED / 3 INCOMPLETE**. Production
adapters, shared State/RenderState contracts, the implementation map and all
previous acceptance/native-feasibility evidence are unchanged. No registration
is promoted, no production dependency is added, and no push was performed.

Whole-environment copies provide no independent clone. Seeded Rocket replay
works in the tested scope, but future unseeded reset entropy blocks the requested
complete guarantee. Both PushT registrations have unbounded public physics
history. **Production `REPLAY_RESTORE` is not recommended at this checkpoint.**
These source/runtime failures triggered the requested stop before replay
performance or production implementation.

## Starting checkpoint and repository growth

Starting HEAD and the exact commit containing the preceding
`state_performance_checkpoint.md` are both
`31153d38c800f44e5c4c0ae1acdd0edca4c5f656`. Work continues on
`repair/environment-semantics` in `/private/tmp/stable-envs-semantic-recovery`.
The worktree was clean before this research. The separate main worktree was
not changed.

Relative to `13e51037cdb4afb9ce31aba3a4837ccf5c95adaf`, that preceding
checkpoint added **1,377 files / 50,293,781 bytes**; modified files added another
419 net bytes. Read-only pack streams estimate **3,692,696 incremental packed
bytes**, compared with a 95,104,642-byte baseline pack. The shared `.git` held
273,910,081 logical bytes; `git count-objects -vH` reported 30.19 MiB loose and
185.49 MiB packed objects. These are measurements of the starting checkpoint,
not a fresh-clone guarantee or this sequence's additional size.

The [growth report](branch_replay_repository_growth.md) gives each new evidence
directory, the twenty largest added files, complete object statistics and
methodology. Pre-v0 consolidation is advisable for checkout size and file count;
history cleanup requires a separate decision. Nothing was deleted or rewritten.

## Rendered end-to-end benchmark

Oscar job **6703024** completed on `gpu2109` in **02:16:34**, with all **37 renderable registrations** successful. Both pixel Craftax registrations also completed on CPU. This supplies **624 composed timing distributions**: 39 device/profile rows × four workloads × four render cadences, each with ten warmups and fifty samples at B1/H16.

Cadences are every step, every fourth step, final frame only and no rendering. Workloads are baseline, extra get after each step, one initial restore and checkpoint/eight divergent steps/restore/continuation. The GPU label denotes the CUDA device request on Oscar; each adapter retains its own dynamics backend. CPU/GPU absolute times span different hosts and are not a hardware speedup claim.

Selected every-step-rendered results (all times are sixteen-step medians):

| Registration / device | Baseline ms | Restore + rollout ms | Branch ms | Baseline / additive | Next step after set / control | Next project+render after set / warm |
|---|---:|---:|---:|---:|---:|---:|
| swm/CartPoleControl-v1 / gpu | 195.150 | 195.895 | 206.277 | 0.999 | 1.014 | 1.001 |
| swm/ReacherDMControl-v0 / gpu | 147.693 | 148.893 | 159.735 | 1.001 | 1.022 | 1.001 |
| swm/FetchReach-v3 / gpu | 75.129 | 76.765 | 95.074 | 1.005 | 1.011 | 1.006 |
| swm/OGBCube-v0 / gpu | 690.617 | 1289.673 | 1384.158 | 1.035 | 1.063 | 1.052 |
| swm/OGBScene-v0 / gpu | 711.757 | 1458.060 | 1546.506 | 1.016 | 1.037 | 1.035 |
| swm/SimplePointMaze-v0 / gpu | 118.084 | 139.055 | 153.717 | 0.962 | 1.167 | 1.050 |
| swm/CraftaxClassicPixels-v1 / cpu | 201.315 | 206.954 | 251.039 | 1.011 | 1.037 | 1.000 |
| swm/CraftaxClassicPixels-v1 / gpu | 1175.747 | 1228.343 | 1471.084 | 1.025 | 0.995 | 1.008 |
| swm/CraftaxPixels-v1 / cpu | 266.296 | 270.425 | 324.929 | 0.897 | 1.001 | 0.994 |
| swm/CraftaxPixels-v1 / gpu | 1630.276 | 1724.956 | 2098.110 | 1.033 | 0.996 | 1.009 |

Every-step-rendered baseline measured/additive ratios range from **0.897** (swm/CraftaxPixels-v1, cpu) to **1.065** (swm/BallInCupDMControl-v0, gpu). These are comparisons within each matched profile. Sums of isolated medians do not necessarily predict composed latency: synchronization, trajectory-dependent work, transfers and cache reuse can differ. They are measured effects, not a causal decomposition.

No next-step or next-project/render paired component ratio reaches the descriptive 1.25 attention flag in this run. Smaller effects remain visible in the full table; this is not a guarantee that restoration has no overhead.

SimplePointMaze's next step is **1.167×** its reset-prefix control, a visible
16.7% increase even though it is below that attention flag. Its no-render
sixteen-step rollout measures **0.851×** the additive estimate. The final report
retains these differences; no threshold hides them or establishes a causal claim.

The [complete result tables and plot](semantic_recovery_evidence/rendered_state_report/report.md) retain all component ratios, measured/additive comparisons and all four cadences. Raw artifacts include medians, p95, individual samples, first-call timings and preparation time. Restore itself remains separately timed even where the following operation does not slow down.

The [methodology](rendered_state_benchmarks.md) defines component boundaries,
reproduction commands and CPU source-pin metadata. The
[derived tables](semantic_recovery_evidence/rendered_state_derived/derived.md)
cover 37 registrations across 60 renderer/device rows. Twenty-five rows use
matched prior component measurements; thirty-five are explicitly cross-host
arithmetic. The new composed results use matched host/profile components and
count projection once. The old literal sum is preserved separately where it
double-counts projection already included in public render.

## Whole-environment clones

| Method | Rocket | Continuous PushT |
|---|---|---|
| Shallow copy | ALIASED_NATIVE_ENGINE | ALIASED_NATIVE_ENGINE |
| Deep copy | ALIASED_NATIVE_ENGINE | COPY_REJECTED |
| Pickle | SERIALIZATION_REJECTED | SERIALIZATION_REJECTED |
| Cloudpickle | ALIASED_NATIVE_ENGINE | SERIALIZATION_REJECTED |
| Native copy | COPY_REJECTED | COPY_REJECTED |

Forty isolated cases cover four checkpoint classes and five methods per engine.
Corrected ownership probes check native queries and rendering in both close
directions. Rocket's aliased copies lose their native world when either owner
closes it; four fresh-process cloudpickle loads have no connected Bullet client.
PushT shallow copies share native Space/body addresses. Rejected or aliased
methods do not qualify for a positive continuation gate. No public clone API or
numeric-State claim follows. See the [clone report](clone_diagnostics.md),
including the explicitly superseded step-only ownership diagnostic.

## Replay classifications and exactness

| Registration | Classification | Preserved decisive evidence |
|---|---|---|
| `swm/PFRocketLanding-v0` | RESET_ANCHOR_NOT_REPRODUCIBLE | Future unseeded Variation reset draws new external entropy |
| `swm/PushT-v1` | UNBOUNDED_PUBLIC_HISTORY | No TimeLimit; 10,240 further physics steps after success |
| `swm/PushT-Discrete-v1` | UNBOUNDED_PUBLIC_HISTORY | Independently tested discrete public path has the same lifecycle |

On official PyFlyt 0.29.0/PyBullet 3.2.7, twelve Rocket seeded cases pass immediate
and eight-action suffix equality after replacing a different-history target.
Coverage includes seeds 0/17/42, near-contact, crash, public post-terminal
behavior, a source-supported successful fixture, shortened truncation and
rendered Variation. Independent clients/storage/RNG, 24 explicitly seeded
future resets and six atomic-invalid-candidate checks pass. The rendered
Variation case has exact pixels and render-between-step numeric nonmutation;
this does not establish every rendered contact/lifecycle configuration.

After an exact one-action seed-17 restore, an unseeded future reset changes five
numeric fields and nine Variation RNG streams. The compact archive retains
these fourteen unequal source/destination pairs. Captured past entropy cannot
predict fresh external entropy consumed by a later reset. This failure concerns
the requested future-reset guarantee, not seeded finite-history determinism.

Both official Pymunk 7.2.0 PushT registrations replay a finite 1,025-action
prefix and final pixels exactly for seeds 0/17/42. Source inspection plus native
timestamps establish continued physics after success and no registered horizon.
An arbitrary fixed history capacity would change the supported lifecycle; none
was introduced. Broader contact and reset tests stopped at this decisive bound.

## Candidate State and performance limits

Rocket's research candidate holds named numeric configuration/reset flags,
explicit seed, Variation values/RNG, reset-anchor and future RNG, a padded
`[H,7]` action history with dtype codes/mask/length, bounded post-terminal counters
and caches, and reconstructed current numeric/render evidence. It contains no
handles, pointers, live objects or opaque native snapshots. Staged construction
and real public replay must compare exactly before replacing the destination.

The authoritative default physical prefix bound is `max_steps + 2 = 1,202`.
Post-terminal public calls change action/counters but skip physics; the candidate
stores their count and final caches. Default numeric size is **767,077 bytes**,
with **807,407 bytes** in named uncompressed NPZ. The shortened H=12 candidate
uses 698,057/738,387 bytes; rendered 64×64 Variation uses 88,172/128,502 bytes.
The [full replay report](replay_feasibility.md) and committed candidate schema
record all encodings and source-derived bounds.

This is not a finalized fixed-dtype schema: upstream rendered ignition state can
change from bool to float, and the prototype still preserves that dtype. No
unseeded-anchor import is implemented. The earlier native patches are unused.

| Requested replay performance gate | Result |
|---|---|
| Reset / 25% / 50% / 75% / maximum / contact-heavy restore | Not run after the exactness/lifecycle blockers |
| B1/B16 replay throughput and cold/warm construction | No complete qualifying candidate; no cost verdict |
| 1/10/100-step, multiple-branch and render-cadence costs | Stopped under the requested failure condition |

No timing threshold or guessed replay cost substitutes for those gates. No
production replay design is approved. A later review would first need to decide
future-reset entropy ownership and PushT's allowable history domain.

## Validation, artifacts and commits

Five benchmark accounting tests pass, including both explicit and internally
projected render paths. Ruff and formatting checks pass. The strict documentation
build and final read-only research/evidence verification pass; exact command
records and outputs are retained in the final evidence directory.

The compact audit rechecks 40 clone records, 12 corrected Rocket ownership
records, four fresh-process deserializations, 12 seeded Rocket cases, six invalid
candidate rejections, six PushT registration/seed records, all fourteen unequal
Rocket witness pairs and both exact PushT witness tails. Passing seeded traces
were checked during execution; the compact archive does not duplicate all their
arrays. The report generator independently recomputes every stored median/p95
and verifies the warmup/sample counts for all 624 composed distributions.

Preservation checks cover 208 accepted-artifact hashes, 112 production hashes,
the prior tracked-file set, matrix/map identity and the allowed research-only
change scope. The 39/3 count is recomputed from the matrix rows. No unrelated
semantic acceptance gate was rerun. GPU exit status is `COMPLETED / 0:0`; all
37 subprocesses succeeded and their raw result hashes match the remote manifest.

Separate commits created in this sequence:

1. `f66d8d2f844441ea9ddc9ae15c7a4460ea7a398a` — rendered benchmark collector and starting growth audit.
2. `e461d9c215679184b94c66e70bc3b4d23439b3d2` — clone diagnostics and ownership witnesses.
3. `41df946ab478f93e02e8dc111ec4bc64be106481` — bounded seeded replay and decisive reset/horizon blockers.
4. Final report/evidence seal — the exact containing SHA is provided in the chat handoff, avoiding a self-referential commit hash.

All final changed files are listed in
`semantic_recovery_evidence/branch_replay_final/changed-files.json`.
Per-directory seals cover compact raw results, counterexamples, source identities,
terminal evidence and generated reports. The final evidence directory has its
repository-relative `REPOSITORY-SHA256SUMS` seal covering every changed file
except that seal itself. Development attempts are retained and labelled; prior accepted
artifacts were never overwritten. Final branch/worktree/clean status and ending
SHA are checked after the containing commit and reported in the chat handoff.
