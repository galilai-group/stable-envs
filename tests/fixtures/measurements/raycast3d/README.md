# Raycast primitive cost measurements

`render/raycast3d.py` bounds a primitive kind only when its measured worst-case
cost ratio exceeds 1 in every run. `test_raycast3d_primitive_bounds.py` checks
`_BOUNDED_KINDS` against these committed measurements.

- `r6_kind_cost.json`: Step-6 per-kind cost table (authoritative; supersedes the
  earlier r5 table). Previously `tests/fixtures/regression/join/step6/raycast3d/experiments/`.
- `kind_cost_ellipsoid.json`: Step-7 ellipsoid aggregate from the quadruped
  workstream. Previously
  `tests/fixtures/regression/join/step7/dmc-quadruped/phase1/cpu_oscar/`.

Both files are byte-identical to their originals, which remain in Git history at
tag `pre-repository-cleanup` together with the other raw runs.
