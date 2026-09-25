# Whole-environment clone diagnostics

Neither pinned environment provides an **EXACT_INDEPENDENT_CLONE** through the
examined routes. These are research object-copy diagnostics, not numeric public
State candidates. No cloning dependency or API was added to production.

| Method | Rocket | Continuous PushT |
|---|---|---|
| `copy.copy` | ALIASED_NATIVE_ENGINE | ALIASED_NATIVE_ENGINE |
| `copy.deepcopy` | ALIASED_NATIVE_ENGINE | COPY_REJECTED: pygame Surface |
| standard pickle | SERIALIZATION_REJECTED: local lambda | SERIALIZATION_REJECTED: local lambda |
| cloudpickle 3.1.2 (already installed) | ALIASED_NATIVE_ENGINE; fresh process has no Bullet connection | SERIALIZATION_REJECTED: pygame Surface |
| source-supported engine copy | COPY_REJECTED: no whole-world clone API | COPY_REJECTED: `Space.copy()` traverses callback/Surface ownership |

Four checkpoints per method cover free motion, first contact, combined/contact
history and a success/terminal fixture. All forty isolated child processes
completed. Rocket's `combined` checkpoint is the crashed world followed by public
post-terminal calls: those calls do not advance physics. It is not a new claim of
sustained raw-engine contact coverage. PushT reuses the preserved naturally
reached multiple-contact fixture. These decisive alias/rejection results stop
before an independent-clone continuation gate can be claimed.

Rocket's deep copy creates different Python objects but retains the same live
Bullet client ID and owner process. Stepping a clone changes the source's native
world, and RNG ownership is tested separately. Closing either copy makes the
other's native-state queries and rendering unusable. Four fresh-process
cloudpickle loads retain a client number but fail with `Not connected to physics
server`. A post-terminal `step()` alone can return cached Python output even with
a dead client; the final ownership check therefore also queries native fields
and renders. Earlier step-only ownership diagnostics are preserved and explicitly
superseded by the `*_ownership_verified` records.

PushT's shallow copy shares native Space/body addresses and changes source
physics when stepped. Its `close()` in rgb-array mode does not release the shared
Space, so both directions remain usable after close; that does not undo aliasing.
Deep copy and `Space.copy()` cannot serialize the callback graph containing the
pygame Surface. Cloudpickle handles the lambdas but still rejects the Surface.
No callbacks or native caches were removed to force a successful serialization.

The initial 40 records and final 16 ownership records are compact JSON files in
`semantic_recovery_evidence/clone_rocket*` and `clone_pusht*`. The latter include
four fresh-process Rocket deserializations. Diagnostic pickle payloads stay in
local temporary directories with recorded hashes/byte counts; they are neither
portable State nor proposed committed dependencies. Constructor failure/aliasing
prevents stronger identical-future, independent-RNG and cross-process claims;
no unperformed positive-clone gate is marked passed.

Reproduction, using fresh output paths and the existing official-engine profiles:

```sh
PYTHONPATH=.:/Users/samib/stable-worldmodel SDL_VIDEODRIVER=dummy NUMBA_CACHE_DIR=/private/tmp/branch-replay-numba /Users/samib/se7-rocket-venv/bin/python scripts/research_replay/clone_diagnostic.py --engine rocket --output /private/tmp/new-clone-rocket
PYTHONPATH=.:/Users/samib/stable-worldmodel SDL_VIDEODRIVER=dummy /private/tmp/pusht-authority-swm-lock/bin/python scripts/research_replay/clone_diagnostic.py --engine pusht --output /private/tmp/new-clone-pusht
PYTHONPATH=.:/Users/samib/stable-worldmodel SDL_VIDEODRIVER=dummy NUMBA_CACHE_DIR=/private/tmp/branch-replay-numba /Users/samib/se7-rocket-venv/bin/python scripts/research_replay/clone_ownership.py --engine rocket --output /private/tmp/new-clone-rocket-ownership
PYTHONPATH=.:/Users/samib/stable-worldmodel SDL_VIDEODRIVER=dummy /private/tmp/pusht-authority-swm-lock/bin/python scripts/research_replay/clone_ownership.py --engine pusht --output /private/tmp/new-clone-pusht-ownership
```

Rocket uses PyFlyt 0.29.0/PyBullet 3.2.7; PushT uses Pymunk 7.2.0. Both are the
unpatched installed profiles from the preceding audits. Native-address values
appear only as research ownership diagnostics. No production fast path is
recommended from these failed clone mechanisms.
