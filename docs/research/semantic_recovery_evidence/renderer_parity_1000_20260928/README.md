# Fresh 1,000-frame rendering comparison

38 measurable RGB registrations pass **normalized MSE <= 0.001 on every
measured frame on both candidate devices**, using the backends listed in the
CSV. Of these, 24 also pass <= 0.0005 on every frame. 36 pass <= 0.0005 on the
per-registration means; Cheetah and Hopper are slightly above that stretch
mean, at 0.0005066 and 0.0005405 respectively.

This is not 42/42 default-renderer completion. The two PushT registrations are
unestablished because source-compatible full-State correspondence is incomplete.
The two symbolic Craftax registrations have no RGB output. PickAndPlace's three
variants have **reset-only** coverage because source rendering after actions
mutates continuation caches. See each row's scope.

- `render_differences_42.csv`: readable requested columns, mean ± sample SD,
  worst-frame errors, backends and scope.
- `renderer_comparison_42.csv`: numeric mean/SD/max, six pairwise comparisons,
  timing samples, unique State counts and explicit availability statuses.
- `images.html`: worst measured GPU/source comparison for each RGB registration;
  source is on the left. For source environments without GPU rendering, the
  reference is source CPU. These are original exported pixels, not AI images.
- `per_frame_errors.csv.gz`: all independently recomputed frame errors.
- `report.json`: profiles, source paths, immutable image/campaign hashes,
  manifest and detailed comparison outcomes.
- `renderer_throughput_tradeoffs.csv`: same-device, batched RenderState rendering
  costs for Cube, Scene and Rocket. Raw timing samples are in the companion JSONs.

MSE is mean squared RGB error after dividing uint8 channel values by 255.
SD is the sample SD across 1,000 frame observations (ddof=1), not the standard
error of their mean. Some observed States repeat; unique State counts are in the
numeric table. Cold initialization and warmups are excluded from steady timings
but retained in the original job records. Image/state file I/O is excluded.

The main timing columns include public full-State projection/validation when the
environment exposes it. They exclude get_state/set_state/reset/step. This differs
from the separate throughput probe, which supplies a prebuilt RenderState.
Consequently its small latency numbers must not be substituted into the main
CSV. For example, Rocket's main CPU/GPU means are 140.40/37.19 ms with full-State
projection; its GPU RenderState-only batch-1 measurement is 5.49 ms. Source
Craftax render returns an already cached observation while candidate rendering
recomputes it; those are different workloads. CPU/GPU machines and library
profiles differ across roles and are recorded in `report.json`; these timings
are not controlled cross-hardware speedup ratios.

Performance determines the default selection: Cube's optional two-pass
`mjwarp_fidelity` path costs 66% more at batch 256 in the production probe;
Scene's optional OpenGL path costs about 89% more at that batch. Their fast
defaults remain unchanged. The corrected Rocket CUDA path costs 1.75% more at
batch 256 and is faster at small batches. CPU OpenGL additions are explicit.

All matched-State joins require identical full-State/case/seed/step hashes.
Cube/Scene and Quadruped source GPU images were rendered after installing the
recorded CPU numeric State in the actual source implementation, then recapturing
it exactly and verifying rendering nonmutation. Their State archives remain
in the original CPU corpus; `manifest-v6.json` and `corpus_sha256` identify it.
Craftax CPU/GPU profiles have incompatible full-State witnesses, so their
cross-device gaps remain **NONCOMPARABLE_STATE**. Their same-device candidate
versus source comparisons pass independently; no cross-device equality is claimed.

The exact source pin is 176d54532b3c0a251a4f509b74f2fdacc143a63d. The code base is
df80c9c54590afa9fc0482a4dcb241d21f6b63a3 plus the local changes whose production
file hashes are recorded here. All 13 changed production files match the final
Oscar-tested source-v12. Accuracy runs used the named immutable earlier
revisions; v12 only bounds Cube's reference cache relative to v11 and corrects a
test fixture. Its batching/nonmutation regression passed. No push occurred.
The semantic matrix remains 39 VERIFIED / 3 INCOMPLETE.

Raw images, State archives, launchers, logs and source revisions remain under:
`/oscar/scratch/sboughan/se-semantic-recovery/render-parity-df80c9c5-20260928`.
The prior accepted evidence was not overwritten. CUDA initialization failures
on gpu1503 were retried successfully on other nodes. The abandoned Rocket
array task used an obsolete renderer and was canceled after its corrected
replacement had completed. Neither is counted as successful evidence.
