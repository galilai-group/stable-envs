# Replay-backed restoration feasibility

This checkpoint **does not recommend production `REPLAY_RESTORE` yet**. No engine
fork is needed for the tested seeded trajectories, but the requested complete
registration gates remain blocked. The matrix stays **39 VERIFIED / 3 INCOMPLETE**.

| Registration | Research classification | Decisive limitation |
|---|---|---|
| `swm/PFRocketLanding-v0` | RESET_ANCHOR_NOT_REPRODUCIBLE | Future public `reset(seed=None)` draws fresh external Variation entropy |
| `swm/PushT-v1` | UNBOUNDED_PUBLIC_HISTORY | No registered horizon; physics continues after success |
| `swm/PushT-Discrete-v1` | UNBOUNDED_PUBLIC_HISTORY | Same unbounded engine lifecycle, independently exercised discrete actions |

These are specific reset/lifecycle failures, **not** claims that replay diverges
when given the same complete deterministic input history. Per the requested stop
condition, further replay expansion and horizon/branch performance measurements
stopped after the decisive counterexamples. No arbitrary performance threshold
was applied. No production adapter, shared contract, registration or dependency
changed, and the earlier native patches/counterexamples were not modified.

## Rocket: bounded seeded replay works in the tested scope

The prototype in `scripts/research_replay/rocket_replay.py` uses the installed
**PyFlyt 0.29.0 / PyBullet 3.2.7**, not either research-patched Bullet build.
All reconstruction goes through the authoritative constructor, reset and public
step methods. No pose injection, internal-substep skipping, opaque native snapshot
or original-instance ownership is involved.

Twelve cases pass exact immediate numeric comparison and eight-action future
suffixes: resets and mixed thrust for seeds 0, 17 and 42; naturally reached
near-contact; crash/contact; public post-terminal contact; a successful landing
fixture; a shortened time-limit boundary; and rendered/randomized Variation.
The near-contact checkpoint has 301 actions, crash has 303, and the post-terminal
case adds 32 public calls after crash. Landing uses the preserved source-supported
ceiling 3.2 and `accelerate_drop=False`, reaching success in seven calls. This is
not a default-policy success claim. The shortened 0.25-second horizon exercises
its off-by-one truncation in twelve calls.

Every target is independently constructed, receives sixteen different valid
public actions, and is replaced only after staged reconstruction compares exactly.
Reference suffixes interleave the two independently owned Bullet clients.
Rewards, state, success, termination and truncation use exact array comparison.
Main-stream future RNG draws and independent RNG consumption pass; two future
**explicitly seeded** resets per case pass. Rendered Variation compares exact
pixels and checks render calls for numeric nonmutation between steps. This is
bounded evidence, not a blanket renderer/nonmutation claim across all contacts.

Six invalid candidate changes—length, mask, nonfinite action, dtype, configuration,
and a contradictory current reward—are rejected without changing the destination.
The last is rejected after staging and comparison but before publishing; structural
validation occurs before construction. Existing source-owned storage is never
installed into the destination.

## History bound and candidate schema

The source computes `max_steps = int(agent_hz * max_duration_seconds)` and checks
`step_count > max_steps` inside physics, before incrementing `step_count`. Therefore
at most **max_steps + 2** public calls can advance physics: **1,202** at registered
defaults. Termination can shorten that prefix. After termination/truncation, public
step still copies the last action, sets the drone setpoint, zeros reward and
increments the counter, but skips all physics. The prototype retains the physical
prefix plus the post-terminal count and last action. It executes one representative
final post-terminal call and restores the total counter; this is the explicitly
source-derived bounded post-terminal exception, not suppressed active physics.
The earlier audit's extra raw-engine settling steps are outside this public path
and are not silently included as replayable public histories.

The **research candidate**, not a finalized production schema, contains named
numeric leaves only:

| Group | Encoding |
|---|---|
| Constructor/compatibility | version, ceiling/displacement/duration/frequency, reward/angle/render flags, resolution |
| Reset recipe | int64 explicit seed; boolean reset flags and source-named Variation mask; reset-anchor PCG64 words |
| Action prefix | float64 `[H,7]`, uint8 `[H]` original float32/float64 codes, boolean `[H]` mask, int64 length |
| Post-terminal lifecycle | int64 count and uint8 final action dtype; current action/setpoint/counters |
| Current numeric evidence/caches | preserved task/actuator/physics/RNG inventory and immediate goal image, reconstructed and compared rather than injected |
| Variation | named numeric current/initial values and PCG64 state words for every source space |

PCG64 uses four uint32-range limbs each for state and increment plus presence and
cached-word flags in fixed uint64 arrays. Action padding and masks are validated.
Current floating values are promoted losslessly to float64; original action dtype
codes preserve the authoritative arithmetic when replaying mixed input dtypes.
Only float32/float64 finite, bounded public actions and explicit nonnegative int64
seeds are implemented. Other constructor/reset/configuration domains are not
claimed accepted.

The full default reset schema is in the first record of
`semantic_recovery_evidence/replay_rocket_v3/result.json`:

| Tested compatible configuration | Numeric bytes | Uncompressed named NPZ bytes |
|---|---:|---:|
| Default H=1,202, render disabled, 480×480 immediate goal cache | 767,077 | 807,407 |
| Shortened H=12, render disabled | 698,057 | 738,387 |
| Rendered 64×64 Variation checkpoint, H=1,202 | 88,172 | 128,502 |

A further schema limitation remains: the upstream `previous_render_ignition_state`
starts as a bool and can become a floating scalar after ignition in render mode.
The prototype preserves that current field's dtype, so a full fixed-dtype proof
across rendered reset/ignition is unfinished. Its seven-byte difference is not
hidden as serialization noise. A future authorized design would need a canonical
numeric flag encoding. The reset-entropy failure already stops this checkpoint;
no additional implementation was made to work around it.

## Decisive unseeded reset counterexample

After a one-action seed-17 checkpoint is restored exactly into a different-history
target, both call the real public `reset(seed=None, options={"variation":
["environment"], "randomize_drop": True})`. Their reset observations and physical
positions differ, along with nine Variation leaf RNG streams. The preceding
ReplayState and its captured future main RNG are equal.

Pinned Gymnasium `Dict.seed(None)` independently seeds each child with None;
Gymnasium's seeding function obtains new `SeedSequence` entropy. Rocket's reset
calls that Variation seeding path afresh. Capturing the **past** reset entropy or
current generator states cannot predict independent **future** external entropy.
A reset-event entropy input or a changed ownership policy could make that input
explicit, but it would be additional semantics requiring review. The prototype
supports explicit seeded anchors; it does not claim unseeded-anchor import.

The small `unseeded-reset-counterexample.npz` contains the differing numeric fields,
RNG words and the one-action prior prefix. The source code excerpts, package paths
and hashes are in the adjacent `provenance.json`. Failure classification concerns
the requested future-reset guarantee, not the twelve passing seeded continuations.
The failed prototype is retained for review; no acceptance promotion follows.

## PushT: registered physics history is unbounded

Both pinned SWM registrations omit `max_episode_steps`. Gymnasium constructs
`OrderEnforcing → PassiveEnvChecker → PushT/PushTDiscrete`, with no TimeLimit.
PushT's public step always executes ten Munk2D steps, returns `truncated=False`,
and does not guard previous success. Discrete PushT dequantizes its 16×16 polar
bins and then calls the same physical method.

For each registration and seeds **0, 17 and 42**, a source-supported explicit goal
matches the initial state and produces success on the first action. A further
1,024 public actions advance **10,240 native physics steps**. Truncation never
occurs. Reconstructing that finite 1,025-action prefix from reset produces exact
output arrays and final rendering independently for all six cases. This establishes
useful finite-prefix replay evidence but supplies no finite public history bound.

The decisive witness records native timestamps and exact final output tails in
one small archive. No arbitrary H=1,000 or H=1,024 capacity is introduced, and no
fixed-size ReplayState is claimed for either registration. A finite registered
wrapper horizon or explicitly bounded branching API would be a scope/semantics
decision; it was not imposed here. Broader contact-class, Variation, unseeded reset
and performance gates remain unperformed after this source/runtime blocker.

## Performance and next decision

| Registration | Horizon / branch performance |
|---|---|
| Rocket | Not run after future-reset failure; no complete exact candidate qualifies |
| PushT | No finite authoritative physics horizon or fixed-shape candidate |
| PushT-Discrete | Same bound failure, independently checked |

Consequently no B16, restore/step, 1/10/100-step branch or multi-branch throughput
is fabricated from these failed gates. Exact-but-slow and inexact restoration are
kept separate; this checkpoint establishes no replay speed verdict. Existing
accepted-environment State/render benchmarks are an independent deliverable.

The next review would need to decide reset entropy ownership and PushT's allowable
history domain before a production replay class can be recommended. This sequence
requests no shared-contract change and authorizes no production implementation.

Reproduction uses fresh output directories:

```sh
PYTHONPATH=.:/Users/samib/stable-worldmodel SDL_VIDEODRIVER=dummy NUMBA_CACHE_DIR=/private/tmp/branch-replay-numba /Users/samib/se7-rocket-venv/bin/python scripts/research_replay/replay_feasibility.py --engine rocket --output /private/tmp/new-rocket-replay
PYTHONPATH=.:/Users/samib/stable-worldmodel SDL_VIDEODRIVER=dummy /private/tmp/pusht-authority-swm-lock/bin/python scripts/research_replay/replay_feasibility.py --engine pusht --output /private/tmp/new-pusht-replay
```

Early collector errors (an invalid divergent ignition action and a Variation
float32/default-float64 schema mismatch) remain recorded separately. Final evidence
uses valid actions and canonical Variation numeric values. Those development
errors are not engine counterexamples; the final reset/lifecycle failures are.
