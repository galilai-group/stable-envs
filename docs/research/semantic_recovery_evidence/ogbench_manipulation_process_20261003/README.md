# Cube/Scene private process candidate

Parent b14e60b978859d8625ab79693a5142f64c618524. This candidate preserves the
public slice-only step API, complete State schema, source physics and renderers.
Defaults remain local. No speedup or GPU acceptance is claimed before the jobs pass.

Persistent spawn processes own contiguous source-world shards. Shared typed
columns hold full State and task outputs; commands/actions use pipes. Received
outputs copy into owned storage before lookahead may overwrite buffers. The
common ProcessRequests protocol enforces one request in flight and reaps every
worker after an unexpected failure. No family-specific threading is introduced.

Restore now exposes the existing authoritative-world construction as private
prepare/discard/commit hooks. Every worker stages and verifies all replacement
worlds before any commit. A compilation/restore failure discards every staged
world and refreshes shared columns from the unchanged live worlds. Unexpected
worker death fails closed. Reset/configuration validate their arguments before
sending mutating requests.

The public semantics layer accesses nq, variation metadata, current style and
anchors through adapter hooks rather than reading every simulator directly.
The process parent retains exactly one non-advancing source metadata/validation
instance, independent of batch size. Validation of an owned State remains usable
while workers advance the next transition and the parent renders the prior one.
Rendering stays in the parent; no CUDA contexts are created in physics workers.

CPU evidence: eight new tests passed independently across Cube and Scene, plus
44 existing manipulation regression tests (42 controlled-reset matrix cases
not rerun). New coverage includes float32/64 actions, uneven three-lane/two-worker
shards, RNG caches, active masks, State ownership, variation anchors, seeded and
unseeded reset, one/eight-action public calls and reverse/empty/disabled slices.
A spawn-safe fault injection fails after real world reconstruction on the final
shard; all live owners remain unchanged and accept a later valid restore. The
two failure tests passed again after adding shared-column cleanup on discard.
Worker-death tests verify all processes and the metadata instance close.

Oscar array 6966296 runs independent Cube/Scene tests, then separate CPU/software-GL
and CUDA/EGL public frame gates and paired throughput on RTX 3090. Each task has
8 physical CPUs, 64 GB RAM and three hours. Source archive: 175 SHA-256-verified
files; all eight tests collect from the isolated bundle. Existing job/evidence
paths are not overwritten.

Each device gate covers 22 cases, exact State/task/frame equality, both action
dtypes, one/eight actions, five slice selections, held outputs, variation/reset
continuation and rejected restore. It verifies that frames render while a future
transition is submitted and outputs remain exact. Timing compares local/P4/P8 at
B16/64 and T1/8, with full State history and frames=None, six samples in two reverse
order blocks. Restore, construction and warmup are outside the timers; CUDA is
synchronized and rates count valid transitions. These are public stepping rates,
not render FPS, native GPU physics or upstream source speedups.

After both tasks finish, summarize_ogbench_manipulation_process.py requires all
four independent gates, matching source hashes and the complete 32 paired cells.
Full step/render pipeline, raw component/source comparisons, larger batches,
measured defaults and integration with the other family candidates remain open.
