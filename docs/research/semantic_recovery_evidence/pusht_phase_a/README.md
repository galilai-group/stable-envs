# PushT Phase A — pinned-profile contact blocker

This is diagnostic evidence, **not** a promotion manifest. No production backend
or public exhaustive TensorDict State was implemented. All comparisons are exact;
no tolerances were relaxed. `results.json` preserves every candidate field before
restoration and after each of four actions, reference/target outputs, action dtype,
contact keys and inspected numeric cache fields for all 12 trials.

## Authority and profile

- Clean integration start: `repair/environment-semantics`, Fetch promotion
  `b45500ca84d8c6b3216d90196acac747e8fb3d5b`.
- SWM: `/Users/samib/stable-worldmodel`,
  `176d54532b3c0a251a4f509b74f2fdacc143a63d`, tracked diff empty.
- Isolated interpreter: `/private/tmp/pusht-authority-swm-lock/bin/python`.
  Python 3.12.11, macOS arm64; exact platform string in `results.json`.
- Pymunk **7.2.0**, NumPy **2.2.6**, Gymnasium **1.2.3**, pygame **2.6.1**,
  opencv-python **4.13.0.92**, CFFI **2.0.0**, Shapely **2.1.2**.
  These versions follow the pinned SWM lock for the exercised dependencies.
  This is not a claim that every optional SWM dependency was installed.
- Pymunk release checkout: `/private/tmp/pusht-pymunk-source-7.2.0`, tag `7.2.0`,
  commit `3ef8f6e304fdc09e16c8ca2cfb1a3aa88a9399bc`.
- Munk2D submodule: `ade7ed72849e60289eefb7a41e79ae6322fefaf3`;
  runtime `pymunk.chipmunk_version` is
  `2.0.1-ade7ed72849e60289eefb7a41e79ae6322fefaf3`.
- `audit.json` hashes all required SWM files, its lock and spaces module, relevant
  Pymunk/Munk2D source, and installed Pymunk Python files and binary extension.
- The isolated profile layers the previously created Fetch authority profile via
  `authority-base.pth`; it does not modify that accepted profile. New packages
  reside in the PushT profile. SWM is imported from the pinned checkout.

The older checkpoint-C probes used Pymunk 7.3.0 / NumPy 2.5.3. They remain
unchanged. This checkpoint freshly executes the pinned 7.2.0 profile. The probe
loads only function definitions from the preserved `07_agent_block_component_ablation.py`
using AST extraction; its hash is recorded in `results.json`.

## Candidate and comparison scope

The diagnostic candidate contains both bodies' position/linear velocity/angle/
angular velocity/force/torque/center of gravity (float64 quantities), sleeping
booleans, reward goal (float64[7]), rendered goal pose (float64[3]), latest action
with its source numeric values, contact-point counter, space timestamp/current
step/shape-ID counter, Gym and local NumPy generators' complete bit-generator
state, and recursive Variation generator state/current values/init values.
Every recorded candidate field is compared, alongside historical seven-number
state, reward, terminated, truncated and `info['n_contacts']`. Success is exactly
terminated in this source. The historical observation is only a diagnostic output,
not a proposed backend State or learner exposure change.

There is **no final exhaustive State schema/dtype contract**. For a future tensor
encoding, physical/task quantities require float64; counters require integer
storage, sleeping flags bool, Variation values their source leaf dtype, and RNG
state a lossless encoding (PCG64's 128-bit state cannot be cast to one int64).
This probe keeps RNG dictionaries rather than claiming such an encoding is built.
Persistent geometry/configuration is fixed and identical in these trials; body
mass/inertia/types, damping, constructor/reset anchors across arbitrary Variation,
RNG reset replay and rendering were not certified. Default sleeping is disabled
(`sleep_time_threshold=inf`); observed body sleeping flags are false. No sleeping
recovery proof is claimed.

Restoration sets angle before position to respect the nonzero block COG, then
restores velocity/force fields and reindexes shapes. Task/RNG/Variation state and
contact counter are restored too. All 12 recorded pre-action candidate roundtrips
are exact. Copying the source-owned cached goal image into the diagnostic target
allows source outputs to be examined; it is not proposed as a backend State field.
No reference source instance is restored, rebuilt or canonicalized.

The enhanced diagnostic performs one real target-only detection step, restores
bodies again, then restores space scalars and every inspected numeric arbiter/
contact field: restitution, friction, surface velocity, normal, swapped/stamp/
state/processCollision/count; each contact's r1/r2, normal/tangent mass, bounce,
normal/tangent/bias accumulated impulses, bias and hash. Exact numeric cache
agreement is asserted. This private-CFFI experiment is not a safe public serializer.
It does not restore opaque linked structures or body bias velocities.

## Results

Each row continues four public actions: float32 zero; float64
`[0.173456789, -0.081234567]`; float32 `[-0.07, 0.11]`; float32 `[0.03, -0.02]`.
Different-history targets first take two other actions in a free-space environment.
Same-history targets are independently constructed and replay the source history
before candidate restoration. Those controls retain matching hidden state and
therefore cannot prove candidate sufficiency on their own.

| Source case | Independent same history | Different history, candidate | Target detection + numeric cache |
|---|---|---|---|
| Free space | Exact 4/4 | Exact 4/4 | Exact 4/4 |
| Agent/block only | Exact 4/4 | Diverges 4/4 | Exact 4/4 |
| Block/wall only | Exact 4/4 | Diverges 4/4 | Exact 4/4 |
| Agent/block + wall | Exact 4/4 | Diverges 4/4 | Diverges 4/4 |

Smallest body-only witness: seed 0, fixed variations, start
`[100,100,400,400,0,0,0]`, reward goal `[200,200,300,300,0,0,0]`.
The default T extends into the bottom wall. After candidate restoration from a
different history and one float32 zero action:

- Untouched source block y: `392.5274849755302`.
- Restored target block y: `393.07986400008406`.
- Absolute error: `0.5523790245538862`.
- Source reward: `-196.40486966116794`; target: `-196.66421736441777`.
- Both terminated/truncated false; both public contact counts 1.

Stronger witness: seed-0 default reset, four chase actions until first public
contact. Source has agent/block and block/right-wall contact. Even after the
numeric-cache restoration described above, the first zero action yields source
block x `386.24343608340274`, target `386.2542073872864` (error
`0.010771303883643668`). Rewards are `-263.08071727822227` and
`-263.0901525558236`. Subsequent maximum output errors are
`0.04810707691274274`, `0.01752311199572887`, `0.015701740370786865`.
Block position, velocity, angle, angular velocity and internal contact counter
also differ. Initial numeric cache fields match exactly; both have two cached
arbiters and two block-active contacts. Cached traversal order differs in this
fresh probe, so order is not excluded as a contributor. The older 7.3.0
order experiment is not substituted as proof of equal order on 7.2.0.

## Bounded source diagnosis

`cpBodyUpdatePosition` integrates `(v + v_bias)` and `(w + w_bias)` then clears
bias velocities. `cpArbiterApplyImpulse` computes and applies bias impulses later
in the step. Consequently these are distinct persistent quantities at a step
boundary, not recoverable by merely assigning ordinary linear/angular velocity.
Body's pickle attributes omit them. The shipped CFFI schema declares `cpBody` and
`cpSpace` opaque. `cpSpaceStep` also maintains active arbiters, cached arbiters,
contact buffers and traversal order. `cpSpaceAddCachedArbiter` inserts contacts
and cached hash entries; it is not a complete live-space restoration API.

These source facts identify missing mechanisms. This experiment does **not**
isolate the exact causal contribution of bias velocity versus ordering/other
space bookkeeping, or prove that no future engine-supported solution is possible.
No opaque struct casts, pointer snapshots, unsafe ABI serializer, engine patch,
approximate solver substitution or shared-contract redesign was attempted.

The supported `Space.copy()` diagnostic was also rerun unchanged on 7.2.0. Here
it raises `TypeError: cannot pickle 'pygame.surface.Surface' object` while copying
the SWM-bound callback graph, before any physics comparison. Its raw traceback
is retained; this is a copy-path blocker, **not** fresh evidence of numerical
copy divergence. The earlier 7.3.0 numerical result is historical only.

## Commands and raw records

Run from `/private/tmp/stable-envs-semantic-recovery`:

```sh
SDL_VIDEODRIVER=dummy PYTHONPATH=/Users/samib/stable-worldmodel /private/tmp/pusht-authority-swm-lock/bin/python docs/research/semantic_recovery_evidence/pusht_phase_a/probe.py
SDL_VIDEODRIVER=dummy PYTHONPATH=/Users/samib/stable-worldmodel /private/tmp/pusht-authority-swm-lock/bin/python docs/research/semantic_recovery_evidence/checkpoint_c/pusht/03_supported_space_copy.py
python3 docs/research/semantic_recovery_evidence/pusht_phase_a/audit.py
.venv/bin/python -m pytest -q tests/unit/test_replacement_readiness.py tests/unit/test_swm_matrix.py tests/unit/test_full_state_contract.py tests/unit/test_semantic_harness.py
.venv/bin/ruff check docs/research/semantic_recovery_evidence/pusht_phase_a/probe.py docs/research/semantic_recovery_evidence/pusht_phase_a/audit.py
.venv/bin/ruff format --check docs/research/semantic_recovery_evidence/pusht_phase_a/probe.py docs/research/semantic_recovery_evidence/pusht_phase_a/audit.py
```

`probe.stdout.txt` / `probe.stderr.txt`: completed investigation, exit 0 with
`PUSHT_PHASE_A_INVESTIGATION_COMPLETE_CONTACT_GATE_FAILED` (not acceptance).
`results.json`: complete numerical record. `space_copy.*`: failed supported-copy
attempt, exit 1. `setup_attempt.*`: initial diagnostic reporter failed on a 0-D
NumPy Variation value after nine cases; superseded by completed results, not
acceptance evidence. Raw stderr retains duplicate SDL Objective-C warnings and
optional ALE/Box warnings. No platform warning was silently suppressed.

`audit.*`: all 33 accepted rows, 170 distinct artifact hashes, 11 terminal logs
including full literal success lines, and all 75 files of the accepted Fetch
SHA256 manifest. `regression.out`: 42 passed, one warning. Matrix/map, production,
shared interfaces, tests and frozen fixtures remain byte-identical to the start.
`SHA256SUMS` hashes this entire diagnostic directory except itself.
