# Source-aligned historical DMC defaults

This bounded cohort recovers four registrations from stable-worldmodel commit
`176d54532b3c0a251a4f509b74f2fdacc143a63d`: Ball-in-Cup Catch, Finger Turn
Hard, Manipulator Bring Ball, and Quadruped Walk. Their existing public names
match the pinned default constructors; no sibling task is certified. Humanoid
remains blocked because the historical default is Walk while the existing cell
is Stand.

The production adapter reconstructs the wrappers' first-reset MJCF writes and
imports dm_control/MuJoCo only. All 585 accessible numeric model arrays, model
shapes, dimensions, observations, and control/lifecycle settings match exactly.
This includes Finger's float32-encoded hinge friction and the wrappers'
float32-encoded geometry sizes.

Common State is qpos, qvel, time, qacc warmstart, DMC lifecycle counters,
cumulative reward, and MT19937 state. Finger additionally stores the target
site x/z coordinates. Manipulator stores the target body's x/z and quaternion
qw/qy. Quadruped stores its 12 activation values. Candidate-only omission tests
against the same uninterrupted reference prove every extra field: warmstart
errors range from `3.41e-15` to `3.60e-13`; Finger target omission reaches 15,
Manipulator target position/orientation omissions reach 0.199/1.08, and omitted
Quadruped activation reaches 2456.58. Restoring the complete schema gives exact
zero error.

The stored model slices are exactly those written by upstream reset: Finger
assigns target site x/z, while Manipulator assigns target body x/z and quaternion
qw/qy. Site y, body y, and quaternion qx/qz remain fixed model coordinates and
are neither reset draws nor task caches.

The CPU gate covers seeds 0/17/42, batches 1/3, exact reset and 32-step output
parity, independent snapshots, full roundtrip, different histories, the
502-public-step implicit-reset boundary, RNG replay, action validation and
historical finite out-of-range pass-through, and full-State rendering through
the unchanged private compact layouts without simulator mutation. Physical
Variation is proven with nondefault gravity for Ball-in-Cup and Quadruped,
Finger proximal density 750, and Manipulator upper-arm density 850. Visual-only
variation preserves exact future physics for all four.
Each physical route also produces nonzero observable divergence from a same-seed
default rollout, ruling out a shared no-op. Constructed in-target states give
positive sparse rewards for Ball-in-Cup and Finger; Ball-in-Cup reports success 1.

`scripts/recovery/dmc_aligned_gate.py` accepts `--device cpu|cuda --output`.
The immutable reference-only seed-17, batch-3, 32-step corpus under
`tests/fixtures/semantics/dmc_aligned/` includes every numeric output leaf,
dependency/source provenance, and hashes; its generator refuses overwrite.
