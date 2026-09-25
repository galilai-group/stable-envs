# Rocket source and exact-state feasibility audit

`swm/PFRocketLanding-v0` remains **INCOMPLETE**. Stage B fails on the
authoritative CPU engine, so Stages C–G were not entered. The matrix remains
**39 VERIFIED / 3 INCOMPLETE**: Rocket and the two PushT registrations. No
production adapter, promotion manifest, shared-contract change, engine fork,
PushT work, or GPU submission was made. No push occurred.

At a naturally reached near-contact checkpoint, restoring all captured numeric
fields except read-only cached link diagnostics gives a different continuation:
maximum reward difference **25.228588632559653**, with different termination
timing. PyBullet's official memory and file snapshots fail the same comparison,
even after explicitly restoring dynamic mass/inertia and Python-side state.
This is the user's required stop condition, not a relaxed acceptance result.

## Starting checkpoint and evidence preservation

The starting HEAD and final Craftax containing/promotion-report commit are both
`badc193af3ee450ac486a75270bc88f47d9a1816`. The branch is
`repair/environment-semantics`, worktree
`/private/tmp/stable-envs-semantic-recovery`; it was clean at entry. The prior
Craftax evidence commit is `59ba52300cc2e9c05da2eea0a58b64c57008036f` and the
Cube/Scene containing commit is `bcf8699d9983f95e97b3c71607521a61c354247b`.

[Starting checks](semantic_recovery_evidence/rocket_source_audit/start.json)
resolve all four Craftax acceptance manifests, the 102-entry final and 87-entry
download manifests, and job **6679082**, `COMPLETED`, `0:0`, elapsed `58:35`,
node `gpu2110`, including all four exact acceptance markers. Its downloaded
archive SHA256 is
`8f26f97ca5509606a32ca88f65224836ee70dbe67f759a80a2c1d4add6e2e932`.
The final verification checks both Cube/Scene acceptance manifests as well.
All pre-existing files other than the research index and Rocket entries in the
two live bookkeeping tables remain byte-identical to the starting commit.
Prior evidence collectors were not rerun or their outputs overwritten.

## Authoritative implementation and ownership

The registration resolves to
`stable_worldmodel.envs.rocket_landing.pyflyt_rocketlanding:RocketLandingEnv`
at pinned SWM `176d54532b3c0a251a4f509b74f2fdacc143a63d`. Its MRO is SWM
`RocketLandingEnv` → PyFlyt `RocketBaseEnv` → Gymnasium `Env` → `Generic` →
`object`. It is a SWM-local modified landing task over unmodified PyFlyt and
external Bullet, not a direct construction of PyFlyt's landing task.
[`compute_state` is AST-identical to upstream; reset, step augmentation,
render augmentation and task thresholds are not.](semantic_recovery_evidence/rocket_source_audit/provenance.json)

| Component | Exact provenance and local source |
| --- | --- |
| SWM oracle | [Pinned source](https://github.com/galilai-group/stable-worldmodel/blob/176d54532b3c0a251a4f509b74f2fdacc143a63d/stable_worldmodel/envs/rocket_landing/pyflyt_rocketlanding.py); `/Users/samib/stable-worldmodel`; file SHA256 `5ede3640ce1e2add32b6b0a6c36e190bb1130e38bb258ab182af12a6fbedee23` |
| PyFlyt | [Official repository](https://github.com/jjshoots/PyFlyt/tree/0.29.0), package/tag **0.29.0**, commit `0ac436c5eb9537dafe8f78b8556fa9996b5ffdc9`; `/Users/samib/se7-rocket-venv/lib/python3.12/site-packages/PyFlyt` |
| PyBullet | [Official repository](https://github.com/bulletphysics/bullet3), installed **3.2.7**, binary SHA256 `fdd9de8bf96c4f3d9bdb23c7d38b8fd360c983cf9a9def7ec02489701c0efeba`; `/Users/samib/se7-rocket-venv/lib/python3.12/site-packages/pybullet.cpython-312-darwin.so` |
| Bullet source examined | [3.2.7 source distribution](https://files.pythonhosted.org/packages/de/95/b9a98cd4ed948d4f25a7d0b13cb0c1f87e58dc6d113be9e18940641eb25f/pybullet-3.2.7.tar.gz), SHA256 `042879db8d101ac7590dee475fc6aded508b85fe1273fdbbfde1d88bd200e14f`; extracted under `/private/tmp/rocket-upstream-audit/bullet-source/pybullet-3.2.7` |

The package metadata does not establish an exact Bullet Git commit; none is
asserted. Source-distribution and installed-binary identities are recorded
separately. All installed PyFlyt Python, YAML and URDF files match the official
tag byte-for-byte. The source manifest contains **41 individually hashed
files**, including wrappers, assets, engine sources and licenses. PyFlyt is
MIT, Bullet is zlib, and pinned SWM declares MIT in `pyproject.toml`. No source
migration or substantial third-party fork was introduced.

| Methods/assets | Current owner and prospective boundary |
| --- | --- |
| Registration; constructor defaults; Variation | SWM registration and local `RocketLandingEnv`; a future bounded task migration must preserve these semantics |
| `_modify_rocket_urdf`, `_modify_pad_urdf`, reset, goal generation | SWM; rocket URDF rewrite exists but is not called by reset; pad rewrite and live rocket recoloring are used |
| `step`, `render` augmentation | SWM adds goal info and removes alpha; delegates ordinary dynamics and camera rendering |
| `compute_state`, `compute_term_trunc_reward` | SWM observation/cache and modified reward/success implementation; upstream landing task is not a substitute |
| `begin_reset`, `end_reset`, ordinary `step`, `compute_attitude`, `compute_auxiliary`, base termination, camera rendering, `close` | Reuse PyFlyt `RocketBaseEnv` |
| World construction, substeps, contact array, wind | Reuse PyFlyt `Aviary` and its `BulletClient` |
| Control mixing, body/fin aerodynamics, booster fuel/throttle, gimbal, camera | Reuse PyFlyt `Rocket`, `BoringBodies`, `LiftingSurfaces`, `Boosters`, `Gimbals`, `Camera` |
| Rigid-body integration, manifolds, solver/cache internals, snapshots, TinyRenderer | External PyBullet/Bullet; do not reimplement |
| Rocket URDF/YAML and pad URDF | Official PyFlyt assets; source-supported SWM color changes only |
| Plane URDF, mesh, material, checker image | Official `pybullet_data` assets |
| Gym checking/order wrappers | Gymnasium; registration itself has no `TimeLimit` |
| Learner image resizing/packaging, optional episode cap and episode reset | SWM `World`, `EnvPool`, `MegaWrapper`; outside exhaustive simulator State |

If feasible later, stable-envs may own the bounded SWM-local task layer while
delegating to real PyFlyt/Bullet; production must not import SWM. The research
collectors intentionally import the immutable SWM oracle. Production files
were unchanged in this audit.

## Source call paths and semantics

Constructor defaults are dense reward, ceiling 500, horizontal displacement
200, 30 seconds, quaternion observation, 40 Hz agent, `render_mode=None`, and
480×480 rendering. The registration supplies no kwargs or maximum steps.
`gym.make` adds order/checker wrappers. SWM World normally requests `rgb_array`
and can impose a separate `max_episode_steps=100` outer limit; its episode
reset is not a hidden reset in the registered environment.

Reset seeds the Variation tree, restores its initial values, samples selected
leaves, constructs start position/orientation, and calls PyFlyt `begin_reset`.
That seeds the per-environment NumPy generator, disconnects/rebuilds the DIRECT
world and resets action, counters, reward and flags. Default options are
`randomize_drop=False`, `accelerate_drop=True`; start height ratio is 0.9,
offset/tilt zero. PyFlyt randomized drop can sample XY ±20, height 400–450 and
rotation ±0.3. Its misspelled `randoimize_drop` velocity branch is a source
fact, not silently corrected. SWM clamps lateral spawn to ±2 and replaces
accelerated vertical velocity with **−30**, rather than upstream −100.
Starting fuel ratio is **0.05**. The fixed pad is placed at `(0,0,0.1)`;
`end_reset` runs ten Aviary steps (twenty Bullet steps), registering contacts
and recomputing observation. Default observed post-reset height is
447.46685770679596.

With rendering disabled, cached goal is a zero RGB image. With rendering
enabled, reset saves an official engine snapshot, temporarily poses the rocket
upright at z=1.5, renders the goal, then restores/removes the snapshot. This
source behavior also crosses the engine's cache-recomputation path; recovery
must preserve the source behavior rather than canonicalize it away.

The seven continuous controls, in order, are finlet x/y/roll, ignition,
throttle, gimbal x/y. The declared space is float64 with lower bounds
`[-1,-1,-1,0,0,-1,-1]` and upper bounds all one. The wrapper copies the action;
there is no global clipping. Four mixed fin commands are clipped to ±1;
booster/gimbal components assert their bounds, and ignition tests `>0.5`.
There are no separate side-jet thrusters. Agent steps repeat three Aviary
steps, each with two 240 Hz Bullet substeps. Zero is a real no-op control,
but still consumes booster noise. Integer/float32/batched public parity and
boundary-neighbour tests have not been accepted.

Booster throttle and gimbal time constants are 0.01; fin actuation is 0.05.
Fuel capacity is 410.9, initially 20.545, with maximum burn rate 1.451 and
thrust range 2966.7–7607. Dynamic fuel mass/inertia are updated every physics
substep. The exact noise call is `np_random.normal(*self.throttle.shape)`:
for the one-booster shape `(1,)` this passes a scalar mean of 1, not a
zero-mean array shape. Default wind is `None`. The body/control calculations
use retained component velocities and engine link transforms.

`compute_state` shifts previous reward caches before updating attitude and
auxiliary state. The default learner observation is 30 float64 values:
body angular velocity (3), quaternion converted from cached Euler angles (4),
body linear velocity (3), position (3), action (7), auxiliary values (9),
previous pad-contact indicator (1). It is not exhaustive State. Contact is
updated after observation, so this indicator lags an internal iteration.

For each internal iteration, dense shaping is

```text
-0.3 + 0.3/(norm(xy)+0.1)
+10*(norm(previous_xy)-norm(xy)) +0.2*(previous_z-z)
+4*(ground_vz-previous_ground_vz+1)/exp(z)*(1 if ground_vz<0 else -1)
-abs(body_angular_velocity_z)-norm(roll_pitch)
```

Pad contact adds `5 - 0.3*abs(ground_vz)`. Contact is fatal if previous
angular-speed norm is >10 or previous body-linear-speed norm is >5. Success
requires pad contact, previous angular norm <5, previous linear norm <4,
tilt norm <0.6, XY norm <12, and z<3. It sets **truncation**, `env_complete`
and adds 3; these thresholds differ materially from upstream landing.
Non-ignored collision, z<0, radius>200 or z>500 terminates. The base excludes
the rocket/pad collision pair from generic collision failure. The duration
check is `step_count > max_steps` before increment, with max_steps=1200.

After terminal/truncated state, a raw public `step` still copies action, sets
the setpoint, resets reward and increments its counter, but breaks before
physics. It does not auto-reset. Post-terminal engine settling below is
explicitly diagnostic, not an alternative public terminal policy. Base close
disconnects Bullet; final cleanup of SWM's temporary modified-pad file would
need attention in future production lifecycle work.

## Candidate State, RNG and restore boundary

[Candidate schema](semantic_recovery_evidence/rocket_source_audit/feasibility-final/candidate-schema.json)
enumerates **91 named numeric probe leaves** with dtypes and prospective
leading batch dimension 1. The private probe stores independent NumPy arrays,
not a validated public TensorDict. It is intentionally **not claimed
exhaustive or accepted**: complete editable engine state is unavailable through
the inspected binding. [Field classifications](semantic_recovery_evidence/rocket_source_audit/state-inventory.json)
identify each captured leaf and remaining source-derived categories.

| Category | Treatment required for recovery |
| --- | --- |
| Body position/quaternion/velocities, link masses/inertias, nonfixed joint state | Required future State; captured numeric getters; all rocket joints are fixed |
| Fuel, throttle/ignition, gimbal and fin actuation, controller setpoint/command | Required future State; explicitly captured and restored |
| Current/previous kinematics and contact/reward/lifecycle caches, info flags | Required future State or immediate-output cache; do not recompute by stepping |
| PCG64 state/increment/cache | Required future State, owned by each environment and shared by its PyFlyt components |
| Current goal, ignition color/render bookkeeping | Required immediate-output cache; goal image has independent storage |
| Engine cached/interpolated link and collider transforms, velocities, contact/solver internals | Required complete engine state; numeric roundtrip is blocked; read-only cached-link table is diagnostic evidence, not a public schema |
| Constructor settings, assets/topology, solver configuration, visual materials, Variation initial values | Persistent configuration requiring future compatibility validation; mutable settings must be represented if exposed |
| Variation values and recursively owned RNGs; reset options/history | Required for complete reset replay; not implemented by this dynamics-only feasibility collector |
| Quaternion/rotation transforms and assembled learner vector | Derived or learner-only in principle, but preserve current cached outputs where source timing or floating-point recomputation matters |
| Drone camera RGBA/depth/segmentation capture buffers | Diagnostic/derived for this registration; raw task rendering uses the camera matrices, not these image buffers |
| Bullet client/body handles and temporary filenames | Resource identities, never public State |

PCG64's 128-bit `state` and `inc` are each encoded as four little-endian
32-bit words stored in int64 leaves; `[has_uint32, uinteger]` is a separate
int64 pair. Future validation must check ranges, odd increment and cache
constraints. The wrapper, Aviary, Rocket and booster share the same generator;
the collector asserts that identity and restores it last. Action-space
sampling is not used by the dynamics path. Default wind adds no separate RNG.
Variation reseeds its space tree on every reset; `seed=None` introduces OS
entropy. Isolated ownership of that reset entropy, complete Variation RNG
capture, interleaved-environment/global-RNG tests and unseeded replay remain
unproved. No claim of complete RNG acceptance is made.

The private diagnostic restore performs official memory/file import or numeric
base/joint restoration, restores dynamic masses/inertias (including for
official snapshots), copies Python actuator/task/cache fields, restores info
and finally restores PCG64. It cannot restore cached links or complete solver
internals. It has no public batch validation, atomic mutation guarantee,
configuration rejection or RenderState restoration API. A future production
order must first validate all lanes/configuration, restore task and complete
engine state without changing its caches, restore immediate outputs and RNG,
then compare exact State before atomic swap. This audit does not approve that
order as implementable with the current engine binding.

## Feasibility evidence and controls

[Final collector result](semantic_recovery_evidence/rocket_source_audit/feasibility-final/result.json)
retains all original snapshots, immediate restores, full continuation arrays,
actions and contact counts. Every comparison uses `numpy.array_equal`.
Each case saves the source checkpoint, runs a reference continuation, then
runs 35 substantially different Aviary steps with controls
`[0.8,-0.6,0.3,1,0.8,0.2,-0.2]`, restores and repeats the original continuation.
Three routes are tested: official `saveState`/`restoreState`, official
`saveBullet`/file restore, and named numeric reconstruction. Opaque `.bullet`
files are retained solely as diagnostic controls, never proposed State.

| Naturally reached checkpoint | Official memory/file continuation | Numeric continuation |
| --- | --- | --- |
| Free flight, seeds 0/17/42 | Exact captured future arrays; immediate cached links differ by up to 0.128143310546875 | Same result |
| Mixed thrust and rotation, 12 agent steps | 22 differing future leaves; max reward delta 0.00020266653983025407 | Same result |
| Default no-op descent, z=4.791145304711372 | 29 differing future leaves; reward delta 25.228588632559653; termination timing differs | Same result; every immediate captured non-diagnostic leaf is exact |
| Crash, z=2.4458051246633 | 25 differing future leaves; reward delta 4.312914344525655 | 23 differing leaves; reward delta 4.549935390544111 |
| Sustained contact after 300 raw Aviary steps | Captured future arrays exact, although immediate link cache differs | 22 differing leaves; reward delta 0.7085699805161774 |

Crash and resting continuations deliberately advance the raw engine after
public terminal state. They diagnose contact state; the near-contact failure
uses ordinary public stepping and independently establishes the blocker.
The final evidence includes actual contacts during the different histories.

[Additional controls](semantic_recovery_evidence/rocket_source_audit/controls-final/result.json):

- Two independently constructed source instances reset with seed 17 and given
  the same twelve mixed actions produce exactly matching captured trajectories.
- Restoring an official file snapshot into the independent instance still
  produces the thrust/rotation mismatch above.
- An immediate official restore, with no intervening history, changes cached
  link data. This isolates a restore-side change rather than requiring a
  different-history explanation alone.
- A source-supported alternate constructor fixture (`ceiling=3.2`,
  `accelerate_drop=False`) reaches successful landing through seven real
  gravity steps, z=2.576009909572719, `env_complete=True`, truncation true and
  termination false. Its post-success raw contact continuation differs by
  0.6176515523011528 reward for official restore and 0.9881692619647602 for
  numeric restore. This is **not registered-default landing-policy acceptance**.
- With `render_mode="rgb_array"`, seed 17 and twelve mixed actions, rendering
  does not mutate captured State. An immediate official restore changes
  **34,725 RGB channel values**, maximum difference 255, in a 480×480 uint8
  frame. Both frames are retained.

The initial diagnostic run exposed omitted dynamic fuel mass/inertia in
official restoration. The collector was corrected to restore these for all
three routes. `feasibility-initial`, `feasibility-mass-complete` and `controls`
are preserved as developmental evidence; `feasibility-final` and
`controls-final` are authoritative for this blocker. No conclusion depends
on the initial mass omission.

### Engine source explanation and limits

In the pinned 3.2.7 source distribution,
`PhysicsServerCommandProcessor.cpp:14772` (`processSaveStateCommand`) and
`:15051` (`processSaveBulletCommand`) set `BT_SERIALIZE_CONTACT_MANIFOLDS`.
It would be incorrect to say official snapshots simply omit all contacts.
`processRestoreStateCommand` at :14837 imports into existing objects through
`btMultiBodyWorldImporter`. Its `syncMultiBody` at :105–185 restores velocities
but runs `forwardKinematics` and `updateCollisionObjectWorldTransforms`,
rebuilding link transforms from current body pose. Contact synchronization
dispatches collision pairs and matches/restores manifolds, clearing unmatched
ones. The observed immediate link-cache changes agree with that code path.

PyFlyt reads cached link state for aerodynamic component velocities and camera
placement. At near contact, the immediate cached-link difference is
0.3359065055847168 while every other captured numeric field is identical.
Numeric pose reset also recomputes link frames, and some rotated cases acquire
quaternion ULP differences. `getContactPoints` and `getLinkStates` diagnostics
do not provide an exhaustive typed setter for the engine's mutable state.

These results establish that the tested numeric and official restore paths
are insufficient. They do **not** establish that one cached-link setter alone
would solve every solver/contact issue, or that no future Bullet binding can
support exact restoration. No engine fork or binary-blob schema workaround
was attempted.

## Rendering, Variation and performance status

Authoritative rendering uses CPU PyBullet DIRECT/TinyRenderer:
`RocketBaseEnv.render` calls `getCameraImage` with the camera view from cached
link 0, offset `[-1,0,3]`, pitch 30, tracking enabled; projection uses FOV 90,
near 0.1 and far 255. SWM drops alpha. Camera pose may lag the live body by
a physics substep. Rocket's separate 128×128 camera captures at 40 Hz are not
the raw task render output. Flame link 9 is recolored through ignition-change
bookkeeping, so live visual material and its history also matter.

The existing stable-envs Rocket renderer remains the frozen CUDA/nvdiffrast
visual-tier renderer, with its 13-value observation-prefix representation;
it is not exhaustive dynamics State. It was neither changed nor recertified.
No minimal sufficient public RenderState, full/projected frame equality,
batch/chunk invariance, Variation image effects or backend acceptance was
established. CPU source rendering does not automatically authorize CPU-only
promotion under the existing renderer policy. That decision remains deferred
until the physics gate passes; no invented GPU backend or GPU gate was run.

The nine source Variation leaves are rocket body/fin/leg/booster RGB colors,
pad RGB, sky RGB, start height ratio (float32, 0.7–0.95), start XY offset
(float32, ±2) and start tilt (float32, ±0.2). Initial values are persistent;
selected draws and active values are reset-scoped. Sky color is unused in the
source path. Reset recolors rocket visual shapes; pad color rewrites a temp
URDF with three-decimal channels. Without selected color variation the
original white pad is used, despite the Variation default grey value. The
unused rocket-URDF helper is not evidence of live reset geometry changes.
No Variation leaf is newly exposed or accepted.

Public step/reset/get_state/set_state/render throughput, batching,
same/changed-configuration restore and warm/cold performance are **NOT RUN**:
there is no feasible public adapter to benchmark. The collector's elapsed
3.457 seconds is diagnostic runtime, not a throughput result.

## Verification, inventory and resumption

[Commands and results](semantic_recovery_evidence/rocket_source_audit/commands.json)
record the CPU collectors, independent evidence verification, shared tests,
format/lint, whitespace and strict documentation checks. The offline verifier
recomputes **49 array comparisons**, hashes **41 source files**, checks all
**41 non-Rocket rows** unchanged, resolves six Craftax/Cube/Scene acceptance
manifests and preserves 170 earlier artifact hashes and 11 terminal logs.
Preservation is byte/hash verification and shared regression testing, not a
claim that all 39 expensive family CPU/GPU gates were rerun.

[Complete changed-file inventory](semantic_recovery_evidence/rocket_source_audit/changed-files.json)
and [SHA256 manifest](semantic_recovery_evidence/rocket_source_audit/SHA256SUMS)
cover this audit. Only one Rocket commit is needed: source/ownership/feasibility
audit. There are no implementation, acceptance or promotion commits. The
commit containing this report is the ending checkpoint (exact SHA is supplied
in the final handoff; embedding its own hash would be circular).

Resume only after an engine-supported complete **typed numeric export/import**
is available for the same world/configuration, including cached/interpolated
link/collider transforms and velocities, dynamic mass/inertia, and all
future-relevant contact/solver state. It must restore across independent
instances without recomputing away the checkpoint, replaying action history,
or exposing opaque handles/blobs as public State. A corrected official
snapshot API or reviewed binding could be investigated, but is not requested
or implemented here. Rerun the bounded Stage B probes into a **new output
directory** first, including immediate State/pixels and public near-contact
continuation. Only after that passes should production State validation,
complete reset RNG, rendering, Variation, performance and acceptance proceed.

All 39 prior accepted rows and their committed evidence remain intact. No
shared code changed and no shared-contract redesign is requested. Production
gained no SWM import and no external physics engine was reimplemented. Rocket
remains incomplete; this sequence stops for review at the explicit Stage B
gate.
