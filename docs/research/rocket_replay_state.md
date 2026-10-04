# Rocket replay State candidate

The current 480×480 CPU/CUDA public gates and 96×96 online source gates pass.
The low-resolution coverage blocker was fixed without changing the 0.001 limit.
Rocket is accepted, bringing non-Craftax State/simulation acceptance to **38/38**.
See the [final acceptance and limits](rocket_acceptance.md). Broad performance
certification remains separate.
The earlier [renderer blocker checkpoint](rocket_public_renderer_blocker.md)
is preserved as historical evidence, not the current renderer status.

The public factory class `PyFlytRocketLanding` delegates CPU dynamics to the
migrated SWM-local task on official PyFlyt 0.29.0 / PyBullet 3.2.7. Its optional
`rocket` extra pins the audited NumPy and Gymnasium profile. Production imports
no stable-worldmodel code. CPU rendering uses TinyRenderer with explicit body
and cached camera poses. CUDA rendering retains its visual tier and frozen
acceptance limits; the final 204-checkpoint gate has maximum normalized MSE
0.0000970718, while the CPU gate is pixel-exact. These are bounded measured
claims, not universal fidelity or throughput guarantees.

Each batch lane stores numeric fixed-shape leaves:

- `recipe`: version, constructor configuration, reset seed/presence, reset flags,
  Variation selection and main reset RNG anchor.
- `anchor.variation`: the PCG64 states immediately before reset sampling.
- `history`: float64 actions with per-row original float32/float64 codes,
  lengths and validated prefix masks. Capacity is `max_steps + 2`, or 1,202
  for the registered 30-second/40-Hz configuration.
- `lifecycle`: post-terminal call count and final action dtype. Post-terminal
  calls update source counters/control caches without advancing physics.
- `current`: exhaustive named physics, actuator, task, reward, lifecycle and RNG
  diagnostics. Replay verifies these rather than installing partial native state.
- `variation`: current values, persistent future-reset values, episode reset
  values, their numeric representation codes, and owned RNGs.
- `render.rgba`: actual engine visual colors, including the retained flame state.
- `rng.action_space`: optional action-space generator state.

The goal image remains the named `current.wrapper.current_goal` immediate-output
cache. At 480×480 it occupies 691,200 bytes. No complete proof justifies removing
it. Python float default height 0.9 remains distinguishable from sampled float32
height. Numeric leaves have fixed dtypes even where the source changes a scalar
between bool and float on its first step.

Restoration validates all batch lanes, constructs independent temporary engines,
installs reset anchors, and replays valid physical actions through public `step`.
It reconstructs post-terminal control/counter effects and owned future RNGs,
then requires exact equality of all supplied numeric leaves. Only after every
lane passes are the engines swapped. Failure closes all staged engines.
Rejected actions never enter history. Snapshots own their storage.

Reset stages a fresh engine with persistent Variation and the current owned RNG;
it does not replay discarded physics history. Seeded resets are exactly
replayable. Unseeded resets preserve the authoritative fresh external entropy
of Variation seeding; snapshots of the resulting episodes are replayable.

Restore costs O(H) for the recorded physical prefix, with H at most 1,202 for
the registered configuration; ordinary stepping does not replay history.
Accepted replay requires a matching source/runtime math profile; cross-BLAS-profile
bitwise portability is not established. Remaining descriptive performance work is
tracked separately, without replacing prior sealed evidence.
