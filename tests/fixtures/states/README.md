# State corpora

Reached-state arrays that unit and GPU tests render or restore. Files are read
only; regenerate into a new directory and compare before replacing one.

## `fidelity/`

One `<Configuration>.npz` per renderer configuration, holding `states` (the
rows tests render) and `meta` (JSON selection metadata: seed, pool size, kept
count). They were frozen once by the Step-4 fidelity sampler
(`docs/research/artifacts/fidelity_step4.py`, `corpus_for`) and first committed
under `tests/fixtures/regression/join/step4/states/`; the sampler and its
original location remain in Git history at tag `pre-repository-cleanup`.
`SHA256SUMS` lists the file hashes, unchanged by the move.
