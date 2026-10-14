# State corpora

Reached-state arrays that tests render or restore. Write regenerated corpora to a new directory and compare first.

- `fidelity/<Configuration>.npz`: `states` and JSON `meta` per renderer configuration; `SHA256SUMS` lists the hashes. Read by `tests/unit/test_cpu3d.py`, `tests/gpu/test_fast_paths.py` and others.
- `rocket_landing/`: `states.npz`, `states.meta.json` and `meta.json` (state/style layout, render validation). Read by `tests/unit/test_rocket_landing.py`.
- `ogb_manipulation/`: OGBCube/OGBScene corpora and the pools they were selected from. Read by `tests/unit/test_ogb_manip.py`.
