# Rejected required-State scheduling experiment

This isolated variant tried using `_step` at selected State/frame/final
transitions and `_advance_without_snapshot` elsewhere. No production file was
changed and no cluster job was submitted for this variant.

Classic-only tests pass 135 with 12 CUDA skips. Broader CPU slice/masking
checks expose a real compatibility problem: the existing CheapAdvance fixture
returns fewer task fields from its advance hook than its full-step hook. Mixing
the hooks makes task-history stacking fail for final-only/sparse selections.
The shared protocol did not previously require those two schemas to match.
The existing test is preserved unchanged; no acceptance rule is weakened.

The broader run has 222 passes, 77 CUDA skips and three failures. One is the
compatibility failure above; two Navigation failures are missing local
matplotlib, not evidence of a Navigation behavioral regression. This variant
is rejected for shared production use. Its source, tests and failed XML are
retained. The family-only snapshot candidate remains separately unaccepted
because of the original continuous MountainCar CPU regression. Further work
should profile that path without assuming a shared dispatch rewrite is safe.

Reconstruct from the prior isolated-cells manifest/overlays, then this candidate
overlay; source_manifest.json records the resulting hashes.

The unchanged production scheduler passes the same slice/masking suite: 218
passed, 77 CUDA skips. This isolates the task-history failure to the experiment.
