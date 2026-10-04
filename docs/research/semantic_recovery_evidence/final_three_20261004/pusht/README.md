# Independent public PushT acceptance, 2026-10-04

Both `swm/PushT-v1` and `swm/PushT-Discrete-v1` pass their own public gates.
The matrix is **41 VERIFIED / 1 INCOMPLETE**, or **37/38 excluding Craftax**.
Rocket remains incomplete. This acceptance covers source-aligned simulation,
numeric get/set State, action chunks, CPU rendering and CUDA State/output
storage. It does not certify accelerated GPU rendering or online throughput.

Production is unchanged from `8451e7accc2925456d5573c3d8fabd8d1c361b85`.
`acceptance.json` records independent per-registration decisions and hashes.
The CPU and CUDA raw-verification reports recompute all 74 source cases and
2,424 pixel-exact frame/continuation pairs per registration per device profile.
Oscar job 6993756 passed 473 tests without skips and completed with exit 0.
The preceding public CPU report also supplies 12 mixed-terminal cases per ID.

`edges/` closes the specifically enumerated acceptance edges:

- 48 direct source reward/success predicate cases per ID: position 20 and its
  adjacent doubles, angle pi/9 and its neighbors, angular wraparound and
  velocity contributions. Testing the predicate directly prevents a physics
  step from moving the exact boundary before it is evaluated.
- 18 continuous float32/64 boundary controls and eight discrete int32/64
  boundary controls; ordinary source and public task/physics/frame outputs agree.
- All 256 polar bins compared to source, followed by source discrete, public
  discrete and public continuous transitions using the source-dequantized
  action. Observations, physics/RNG diagnostics, reward, done and frames agree.
- All 17 exposed persistent Variation leaves, every rendering flag combination,
  full/private frame equality, atomic compact-State refusal, and body-state
  non-mutation by configuration changes.

The 2,094 numeric comparisons are retained in `edges/witnesses.npz` and
independently recomputed. Prior source task/reset-anchor evidence is reused only
after confirming unchanged task, quantizer, space and drawing code. Public
continuation and shape evidence covers the current replay schema rather than
assuming old private-schema acceptance transfers automatically.

The State contains a typed reset recipe, complete owned RNG/configuration,
finite typed action history and current diagnostics, including visible contact
segments. Restoration reconstructs official Pymunk 7.2.0 caches by replay into
new engines, verifies every lane and then swaps them atomically. Ordinary steps
do not replay. Export and restoration costs grow with history; there is no new
fixed history horizon. Source action bounds are validated at the public boundary;
source behavior outside the declared action domain is not claimed. Nondefault
source-only `damping`, `block_cog`, and `relative=False` constructor settings
are not exposed. The source's raw truncation is false; SWM World's external
horizon remains outside this simulator contract.

The pinned source is stable-worldmodel `176d54532b3c0a251a4f509b74f2fdacc143a63d`.
Versions and source/production hashes are in the public reports. The full CPU
Pygame renderer is the public route on both storage devices. Earlier compact
GPU renderer fixtures are unchanged and do not confer speedups on this route.

Preservation: 1,121 artifact references across all 35 previously accepted
non-Craftax registrations resolve with matching SHA256. No production/shared
code or frozen fixtures changed in this closure. The older readiness table's
PushT execution classification is corrected to DELEGATED_CPU; its historical
VariationMap capability classification remains distinct from this acceptance.

The broad local map test attempt is retained: 111 passed, 182 skipped, 12 failed
after macOS CoreGraphics context creation failed and its cleanup guard cascaded.
The platform-compatible rerun passed 104 with 54 optional skips. These are not
fresh all-family rendering results. Readiness and semantic harness tests are
included; the unchanged accepted-family evidence provides preservation coverage.
