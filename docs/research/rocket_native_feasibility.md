# Rocket native restoration feasibility

**Outcome: EXACT_RESTORE_NOT_YET_FEASIBLE.** Rocket remains INCOMPLETE.
This extends the accepted audit at `13e51037cdb4afb9ce31aba3a4837ccf5c95adaf`;
it does not repeat or replace its 91-leaf Python/task/RNG inventory. No production
code, ordinary engine step, dependency, State contract or matrix changed.

## Controlled evidence

The research build is isolated under `/private/tmp/rocket-native-research/`.
Its unmodified-restore control reproduces **5,096 numeric array-leaf comparisons**
against the original committed feasibility arrays exactly. The existing seven
checkpoint collectors and actions, landing fixture, independent-instance and
pixel controls run unchanged, through an in-memory diagnostic wrapper.

| Controlled addition | Official memory/file continuation | Interpretation |
|---|---|---|
| Rebuilt baseline, hook disabled | Thrust 22, near-contact 29, crash 25 differing leaves | Reproduces installed-engine failure |
| Skip importer forward kinematics/collider reconstruction | Still fails, including free flight | Preserving target caches is not restoring source caches |
| Restore cached link world transforms | Thrust and independent file continuation exact; immediate pixels exact | These source caches are future-relevant |
| Add collider/interpolation transforms and velocities | Contact failures remain | Insufficient subset |
| Add absolute/local link velocities | Contact failures remain | Insufficient subset |
| Add interpolated base/joint and cached parent transforms | Contact failures remain | Insufficient subset |
| Add activation/deactivation values | Near-contact 29, crash 25 differing leaves | Immediate exposed equality is insufficient |
| Import these caches before contact matching | Same contact failures | Ordering this import before matching is insufficient |
| Also synchronize an empty serialized manifold set | Near-contact drops to 18 differing leaves; crash remains 25 | Clears stale points, still not exhaustive restoration |

Counts are differing trajectory leaves, not failed test counts. Numeric-only
reconstruction remains worse than official memory/file plus caches. Free flight
seeds 0/17/42 and sustained resting contact pass the official snapshot routes;
this does not cover the failing transition into contact. Successful landing uses
the preserved source-supported ceiling fixture, not a default-policy acceptance
claim. All raw per-route results and numeric continuations are retained.
Even the empty-manifold synchronization variant has near-contact reward error
0.1697551170328957 and crash reward error 4.312914337371227 on the memory route;
the crash continuation also differs in truncation on two recorded steps.

## Native path and fields

Pinned source: [Bullet/PyBullet](https://github.com/bulletphysics/bullet3), PyBullet
**3.2.7 source distribution**; its exact archive SHA and patched source/binary
hashes are in `build-identity.json`. Do not substitute a moving repository HEAD.
The bundled `btScalar.h` reports `BT_BULLET_VERSION 326`; its separate header
identity is recorded in the final checkpoint's `native-engine-version.json`.
PyFlyt 0.29.0 provenance and SWM `176d54532b3c0a251a4f509b74f2fdacc143a63d`
remain those in the original provenance artifact.

`syncMultiBody` loads serialized base/joint state, then calls `forwardKinematics`
and `updateCollisionObjectWorldTransforms`. PyFlyt body/fin updates consume cached
link state, so recomputing those caches on restore changes its next force update.
The hook exports/imports named doubles/integers for:

- topology multibody/link/collider counts;
- each link's cached world basis/origin;
- absolute-frame total/local angular and linear velocities;
- cached relative vector and parent quaternion, including interpolation copies;
- interpolated joint positions (only `m_posVarCount` used entries);
- interpolated base position/world-to-base quaternion;
- collider world/interpolation basis/origin and interpolation angular/linear velocities;
- collider activation state and deactivation time.

Groups are enabled cumulatively by masks 1/3/7/15/31. Only cached world transforms
have isolated positive sufficiency evidence for the free-flight/thrust gap. Other
fields are source-identified inputs/caches whose addition did not solve contacts;
this report does not claim every one is necessary.

The serializer **does request `BT_SERIALIZE_CONTACT_MANIFOLDS`**. The importer
updates AABBs, computes overlapping pairs and dispatches collision algorithms,
then matches serialized manifolds by ordered broadphase IDs. It deserializes a
match and empties unmatched existing points. Originally this path is skipped for
an empty serialized manifold set. The v2 export records manifold point geometry,
normal/friction directions, restitution/friction, IDs/flags, normal/lateral
impulses, previous RHS, contact motions, CFM/ERP and lifetime. These are diagnostic
numeric records; this prototype does **not** install a full typed contact graph.

One retained source has zero manifolds while the diverged target initially has
seven. The empty-set synchronization leaves three empty manifolds, not the
source's empty collision-algorithm graph. Pair membership/order, broadphase and
algorithm persistence, island/solver inputs and cache reconstruction remain
unresolved. Activation values and point warm starts alone do not prove closure.
The remaining failure is concrete; the responsible remaining fields are not yet
isolated. No claim of engine impossibility or proven requirement for a large fork
is made. A safe complete fixed bound for all algorithm/manifold storage is **not
established**; no public State schema is finalized from this subset.

## Fresh-link migration diagnostic

A separate run uses the original installed PyBullet and requests
`computeForwardKinematics=True` only when PyFlyt body/fin components consume link
state. Thrust restoration becomes exact, but near-contact and crash still fail.
Ordinary source trajectories change: maximum reward differences are
0.013229327159354654 for thrust, 0.7085699804791776 for resting contact and
0.0000022779302879527563 for crash. Even free-flight cached-link arrays change.
This would be a semantic migration, not faithful recovery, and was not adopted.

## Patch, packaging and cost

Exact diffs are `cache-extension-v1.patch` and `cache-extension-v2.patch` in the
[new evidence directory](semantic_recovery_evidence/rocket_native_feasibility/).
The latter adds read-only contact diagnostics and controlled importer variants.
`prepare_rocket_build.py` applies it to an untouched source distribution using
`rocket_cache_hook.hpp`. Ordinary `stepSimulation` is unmodified. Environment
variables and text-file transport are research plumbing: they are neither
thread-safe nor validated/atomic enough for a production API. All imported values
would need named numeric public leaves, validation and independent storage;
opaque `.bullet` files are diagnostic transfer only and are excluded from the
new committed payloads (their hashes are recorded).

Only **macOS arm64 / CPython 3.12** was built and exercised. The native binary is
not committed. A production extension would need reproducible Linux x86_64,
macOS arm64 and other explicitly supported wheel builds, Python ABI coverage,
precision/build-option locking, compiler/architecture determinism gates and
packaged source/license notices. This build emitted deployment-target warnings;
it is not a distributable wheel validation. Bullet's zlib-style license is
retained as `Bullet-LICENSE.txt`; third-party bundled components need their own
notices if distributed. PyFlyt/task sources were not changed.

A narrow upstream cache/snapshot fix is plausible, but this experiment is not a
complete upstreamable restoration API. Maintaining an importer patch against
Bullet internals is a continuing source/ABI/testing burden. A fork is not justified
by the current evidence, and no patched production dependency is proposed.

At a free-flight reset checkpoint, 10 warmups + 200 samples on Apple M5 yield:

| Diagnostic operation | Median µs |
|---|---:|
| Numeric cache export including text transport/contact diagnostic | 310.500 |
| Numeric cache import including text transport | 309.896 |
| Opaque `saveState` | 383.188 |
| Opaque `restoreState` | 1.458 |

Raw samples, p95, throughput and subset size are in `performance/result.json`.
These operations are **not exhaustive State**. Official snapshots are
`OPAQUE_SAME_PROCESS_NON_EQUIVALENT`; the cheap restore time is not a correct
State baseline. Timing includes the Python/native call boundary and file IO,
excludes construction/compilation, and has batch size 1.

## Reproduction

Use the original pinned Rocket profile and an isolated extraction of the hashed
3.2.7 archive. From this worktree:

```sh
.venv/bin/python scripts/research_native/prepare_rocket_build.py --source /private/tmp/rocket-native-research/pybullet-3.2.7 --patch /private/tmp/new-cache-extension.patch
```

From that pristine, newly patched source directory:

```sh
/Users/samib/se7-rocket-venv/bin/python -c 'import multiprocessing,runpy,sys; multiprocessing.cpu_count=lambda:4; sys.argv=["setup.py","build_ext","--inplace"]; runpy.run_path("setup.py",run_name="__main__")'
```

Use the resulting **build/lib** module directory first on PYTHONPATH (the source
root alone silently loads the installed unpatched module). For each controlled
mode use a new output directory:

```sh
PYTHONPATH=/private/tmp/rocket-native-research/pybullet-3.2.7/build/lib.macosx-15.0-arm64-cpython-312:.:/Users/samib/stable-worldmodel NUMBA_CACHE_DIR=/private/tmp/rocket-native-research/numba SDL_VIDEODRIVER=dummy /Users/samib/se7-rocket-venv/bin/python scripts/research_native/rocket_native_probe.py --mode empty_contacts --output /private/tmp/rocket-new-empty-contacts
PYTHONPATH=.:/Users/samib/stable-worldmodel NUMBA_CACHE_DIR=/private/tmp/rocket-native-research/numba /Users/samib/se7-rocket-venv/bin/python scripts/research_native/rocket_fresh_links.py --output /private/tmp/rocket-new-fresh-links
PYTHONPATH=/private/tmp/rocket-native-research/pybullet-3.2.7/build/lib.macosx-15.0-arm64-cpython-312:.:/Users/samib/stable-worldmodel NUMBA_CACHE_DIR=/private/tmp/rocket-native-research/numba /Users/samib/se7-rocket-venv/bin/python scripts/research_native/native_cost.py --engine rocket --output /private/tmp/rocket-new-cost
```

Other modes: baseline, skip_reconstruction, links, colliders, velocities,
interpolation, activation, pre_contacts. v1 uses its preserved exact diff/module;
v2 results identify their separate build. Initial wrong-module development failure
and unused-slot prototype correction are not semantic evidence; final retained
baseline and tests use the recorded build identities.

**Stop before any patched production dependency.** Contact feasibility failed,
so atomic State, public validation, production rendering and promotion gates did
not begin. No shared-contract change is requested.
