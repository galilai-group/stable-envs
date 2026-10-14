# Raycast primitive cost measurements

Per-kind CPU cost ratios of the bounded and unbounded intersection paths in `render/raycast3d.py`.
`r6_kind_cost.json` covers PLANE, SPHERE, BOX, CYLINDER and CAPSULE; `kind_cost_ellipsoid.json` covers ELLIPSOID.
`tests/unit/test_raycast3d_primitive_bounds.py` checks that `_BOUNDED_KINDS` matches the kinds whose worst ratio
exceeds 1 in every run.
