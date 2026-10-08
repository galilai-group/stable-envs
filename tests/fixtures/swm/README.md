# Frozen stable-worldmodel correspondence

`frozen_cells.json` records, for each of the 42 stable-worldmodel ids, the
`stable_envs` classes that implement it. The test-side crosswalk
(`tests/helpers/reference/crosswalk.py`) reads it, and `test_registry.py` and
`test_worldmodel.py` check the registry and bridge against it.

Extracted unchanged from the `swm_id` and `stable_envs_cells` fields of
`tests/fixtures/regression/gate_f_matrix.json`
(sha256 `1d266c7268278b0564297f486d95a7fa83a96075a3f11db30917969378908271`),
which remains in Git history at tag `pre-repository-cleanup`. The accepted
Step-8 support matrix (`tests/fixtures/regression/support_matrix.json` at that
tag) records the identical mapping, in the same order.
