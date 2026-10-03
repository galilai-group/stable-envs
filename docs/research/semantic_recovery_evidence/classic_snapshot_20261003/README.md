# Classic snapshot-selection candidate

Base: 405216af3548f99c1ad02934cba560ac5e83f822. Production is unchanged at
submission. Array 6970326 independently gates all five registrations on CPU
and CUDA, then measures B1/64/256/1024, T1/8, no/all intermediate States and
no/final frames. Six samples reverse candidate/control ordering. Each report
requires 128 timing cells, 64 paired CSV rows and 14 public cases per device.
This compares the existing library adapter, not upstream or parallel source.

The two production candidate files reuse _advance_without_snapshot and pass
capture_state=False when no snapshot is needed. The shared dispatcher, public
slice API, renderer, schemas and routing policy are unchanged. The mandatory
final State is always owned and complete. Acrobot workers skip exports for
advance requests while retaining the existing task/shared-buffer protocol.

Frozen baseline implementations derive exactly from the base commit with only
the declared import rewrites in reference_provenance.json. Acrobot uses real
frozen spawned workers, preserving the old export-without-State-transport cost.
Candidate and control use identical worker counts and renderer settings.

Local full suite: 147 collected, 135 passed, 12 CUDA skips. Each registration
also passed a B8 CPU driver smoke. Those five smoke reports predate an import-
order-only correction in the frozen Acrobot reference; local_pre_sort_override
records the original contents for exact hash reconstruction. The full suite
and cluster source manifest use the corrected imports. These local checks do
not establish CUDA acceptance or a speedup.

To reconstruct the frozen candidate, export the base commit into a fresh
checkout, overlay candidate/ from this directory, and verify every file listed
in source_manifest.json. Run run.sh on its documented Oscar allocation; use a
fresh output directory. Existing committed reports must not be overwritten.
The archived upload hash and exact job ID are recorded in submission.json.
