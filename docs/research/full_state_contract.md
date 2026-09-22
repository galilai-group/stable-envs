# Shared semantic recovery contract (Phase 0.5)

`get_state()` returns an independent TensorDict snapshot. `state` is its alias.
`set_state(state)` restores every native field and returns None. Persistent
Variation must be the same when resuming. Native correctness here does not certify
historical SWM task/reset/reward/lifecycle equivalence: the recovery matrix tracks
those separately. Cells without simulators explicitly refuse reset/get/set State.

Compact float32 rows are private RenderState. `_render`, `_get_render_state`,
`_set_render_state`, `_pack_render_state`, and `_unpack_render_state` exist for
renderer implementations and frozen corpus tests. The decoder omits hidden
fields; it never fabricates zero velocities. Public restoration and transition
reject raw rows and incomplete dictionaries. `render(full_state)` projects
internally through the existing renderer. Frozen row layouts and fixtures remain
unchanged. Renderer datasets retain their internal compact storage boundary.

`stable_envs._adapters.SimulatorAdapter` is a structural protocol for reset,
get/set State, step, close, and num_envs. It has no rendering or transition method.
Family adapters own upstream simulators; the Env owns Variation and rendering.
The first delegated proof is DMC Reacher, before broader family work.

The immutable Phase-0 audit remains source evidence. The maintained
`semantic_recovery_matrix.json` holds current status for all 42 historical IDs.
No baseline INCOMPLETE or WRONG_ABSTRACTION cell becomes VERIFIED from this API
change alone. Benchmarks remain unrun until semantic gates pass.

Local validation before Reacher implementation: 156 native/contract/harness
checks passed; 180 shared input/dataset/API checks passed, four skipped. Seven
frozen renderer corpora (four rows each) produced byte-identical CPU frames
against commit 6d80b68, including Reacher and Hopper. See
`semantic_recovery_evidence/phase05_cpu_renderer.json`. GPU verification is
pending the Oscar job. This is scoped evidence, not a 42-cell REGRESSION=0 claim.
