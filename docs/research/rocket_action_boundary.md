# Rocket action boundary

The supported domain is a NumPy float32 or float64 vector of shape `(7,)`,
with finite components inside the declared bounds, including both endpoints:
three fin controls in `[-1, 1]`, ignition and throttle in `[0, 1]`, and two
gimbal controls in `[-1, 1]`. Validation does not cast, clip, or otherwise
change valid controls. The official engine retains its fin mixing and clipping.

The local task validates before invoking PyFlyt. Unsupported types/dtypes raise
`TypeError`; incorrect shapes and nonfinite values raise `ValueError`;
out-of-bounds controls raise `AssertionError` (matching the authoritative
booster/gimbal exception category, with local messages). Validation applies
also after termination. A rejected call is not a transition and must not enter
replay history. It must leave physics, pending forces, controllers, lifecycle,
RNG, task caches, and pixels unchanged. The next valid transition must equal
one without the rejected call.

This deliberately normalizes invalid calls. PyFlyt 0.29.0 applies body and fin
forces before its booster/gimbal assertions. Its partial corruption is preserved
as provenance in the [preceding checkpoint](rocket_recovery_checkpoint.md) and
its sealed rejected-control evidence; it is not part of the supported public
transition domain. Neither PyFlyt nor PyBullet is modified.

`scripts/research_rocket_production/action_boundary_gate.py` checks seeds
0/17/42, both float dtypes, every bound and adjacent representable values, all
128 simultaneous endpoint combinations, invalid shapes/dtypes/NaN/infinities,
repeated rejections, and the prior cancellation controls. It compares all 91
numeric diagnostics, every Variation RNG, rendered pixels, and three subsequent
valid steps. These task-level checks precede public State and adapter acceptance.
