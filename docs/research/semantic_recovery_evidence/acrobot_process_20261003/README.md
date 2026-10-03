# Private Acrobot process candidate

This candidate starts from 5ca37dac and does not change public factories,
State/RenderState, the step API, shared request handling, or other families.
It is restricted to `swm/AcrobotControl-v1`; no process default is enabled.

The accepted post-export profile under `../classic_postexport_profile_20261003/`
puts Acrobot's source task work at 66.54 ms per B1024 step, compared with
90.58 ms for the complete adapter before the newer direct-column optimization.
Gymnasium's Python RK4 and `_dsdt` dominate task execution. That makes Acrobot
a stronger process candidate than classic tasks dominated by State packing.
The comparison here uses the newer accepted direct-column exporter, not the
slower profile checkpoint. Profile components are independently measured and
must not be added or interpreted as new throughput results.

The source path remains Gymnasium `Acrobot-v1` → TimeLimit/other Gym wrappers →
AcrobotEnv.step → RK4. Each persistent process owns independent complete source
environments; no physics is copied or approximated. Workers retain sequential
actions within each lane and source reset/termination semantics. Typed shared
columns carry task results and exhaustive State. Parent snapshots copy their
storage before another request can overwrite buffers. The existing
ProcessRequests mixin handles one in-flight request, propagation and cleanup.
Rendering stays in the parent. The private pipeline uses the existing common
dispatcher, with its public `[B,1]`, `[B,T,1]` and discrete `[B]` spellings and
independent State/frame slices.

Restore validates the complete schema and every lane's PCG64 assignment before
dispatching mutation. Source Acrobot restore then only assigns state arrays,
wrapper counters and the prevalidated RNG. Unexpected worker failure closes all
owners. Action validation retains upstream Python-list indexing, including
negative indices and rejection of floating-point indices. Physical variation
is routed to each owner's accepted adapter. Workers still call the accepted
step/export implementation even when the pipeline does not request a snapshot;
conditional worker export is a separate potential optimization, not claimed here.

Local tests: eight pass; two CUDA public gates are explicitly reserved for
Oscar. They cover owned State/task/RNG continuation, per-lane physical variation,
inactive lanes, invalid last-lane restore, pending-request exclusion, worker
death, public one/eight-action slices, mixed time limits/termination and exact
native 500x500 frames. Both serial dispatch and pipeline dispatch are checked,
including `render()` without an explicit State after stepping. A local B8/P2
benchmark smoke passes four exact-output timing cells; its slower process path
does not establish cluster performance or justify a small-batch default.

The frozen Oscar experiment runs all ten tests without skips, then compares
local/P4/P8 at B64/256/1024 and T1/8 on one RTX 3090 allocation with eight
physical CPUs. Both output devices retain CPU physics. Six samples alternate
strategy order; construction, restore, warmup and equality checks are outside
timing. Every timed slot must be valid. All States are retained, frames are
disabled in timings. These measure adapter steps/s, not render FPS, upstream
speedups or combined pipeline throughput. Independent CPU/CUDA native-frame
checks must pass before accepting any timing result.

Reproduce the paired CSV with `summarize_acrobot_process.py REPORT TESTS_XML
NEW_CSV --source FROZEN_CHECKOUT`. It refuses missing tests, skips, incomplete
cells, non-3090 hardware or mismatched source hashes. Source manifest, four
overrides, launcher and local JUnit are preserved here. No semantic promotion,
default activation or performance acceptance follows from submission alone.
