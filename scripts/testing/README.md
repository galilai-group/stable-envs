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

Run a generator from the repository root, in the interpreter pinned for its
family in `tests/helpers/reference/profiles.py`, for example
`python -m scripts.testing.freeze_reacher_oracle`. Move the existing fixture
directory aside first; the generator will not overwrite it.
