# Testing scripts

Generators and verifiers for the sealed fixtures under `tests/fixtures/`. Each
generator refuses to overwrite an existing corpus; write to a new directory and
compare before replacing a fixture.

| Script | Produces or checks |
|---|---|
| `freeze_native.py` | Shared native trajectory corpus (`tests/fixtures/semantics/shared_native`) |
| `freeze_*_oracle.py` | Per-family semantic oracles (`tests/fixtures/semantics/<family>`) |
| `capture_ogbench_golden.py` | OGBench Cube/Scene golden frames |
| `verify_saved.py` | Saved-corpus and evidence verification |
| `inventory.py` | Required-gate coverage inventory |

## Moved generators

The semantic oracle generators previously lived in `scripts/recovery/`. Existing
fixture manifests keep their original `generator` field, which records how each
corpus was created. Corpora generated from now on record the new path.

| Recorded `generator` | Current location |
|---|---|
| `scripts/recovery/freeze_dmc_aligned_oracle.py` | `scripts/testing/freeze_dmc_aligned_oracle.py` |
| `scripts/recovery/freeze_dmc_continuation_oracle.py` | `scripts/testing/freeze_dmc_continuation_oracle.py` |
| `scripts/recovery/freeze_dmc_first_family_oracle.py` | `scripts/testing/freeze_dmc_first_family_oracle.py` |
| `scripts/recovery/freeze_dmc_humanoid_oracle.py` | `scripts/testing/freeze_dmc_humanoid_oracle.py` |
| `scripts/recovery/freeze_native_classic_oracle.py` | `scripts/testing/freeze_native_classic_oracle.py` |
| `scripts/recovery/freeze_native_classic_remaining_oracle.py` | `scripts/testing/freeze_native_classic_remaining_oracle.py` |
| `scripts/recovery/freeze_native_navigation_oracle.py` | `scripts/testing/freeze_native_navigation_oracle.py` |
| `scripts/recovery/freeze_ogbench_maze_oracle.py` | `scripts/testing/freeze_ogbench_maze_oracle.py` |
| `scripts/recovery/freeze_reacher_oracle.py` | `scripts/testing/freeze_reacher_oracle.py` |

Run a generator from the repository root, in the interpreter pinned for its
family in `tests/helpers/reference/profiles.py`, for example
`python -m scripts.testing.freeze_reacher_oracle`. Move the existing fixture
directory aside first; the generator will not overwrite it.
