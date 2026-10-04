# Public PushT CUDA-storage source gate, 2026-10-04

Production checkpoint: `b0fee4068f09ed1c42a4c2aced0c1e66e17f5a50`, `sami-dev`.
Job **6993756** completed with exit 0 on RTX 3090 node `gpu2108` in **8:28**.
The immutable launcher and candidate manifest are retained in
`../pusht_public_state_20261003/`; all production hashes in the completed
report were independently checked against this checkout.

| Registration | Source cases | Continuation transitions | Exact frame pairs | Normalized MSE |
| --- | ---: | ---: | ---: | ---: |
| swm/PushT-v1 | 74 | 2,424 | 2,424 | 0 |
| swm/PushT-Discrete-v1 | 74 | 2,424 | 2,424 | 0 |

`unit.xml` records **473 passed, zero failed, zero skipped**. The source gate
uses independently advancing pinned upstream simulators, disk checkpoints
restored into different public instances, three seeds, B1/B2, histories of
0/100/1025 steps, eight-action chunks and all 56 supported shape combinations
per registration. Source simulators are never restored.

`verification.json` recomputes the raw NPZ witnesses, checks state schema,
padding, file hashes, production hashes and complete case coverage. Frame,
observation and task outputs match exactly. `scheduler.txt` records terminal
success. `public.log` retains the individual case results.

CUDA refers to **State/output tensor storage**. Rendering executes on CPU
through Pygame; these results do not establish GPU rasterization or speedup.
CPU mixed-terminal evidence remains in the preceding checkpoint; it was not
part of this frozen CUDA source script.

No registration promotion is made here. The combined matrix remains
**35/38 VERIFIED excluding Craftax**, pending final per-registration review of
the complete acceptance requirements, including boundary/configuration and
lifecycle coverage. The separate Rocket 96-pixel CUDA rendering failure remains
open. Public replay State export/restore has history-dependent cost.

The lower-priority Rocket CPU cost run was stopped to free the GPU quota slot;
its completed samples and decision are retained in
`../acceptance_priority_20261004/`. The identical pending condo duplicate
`6994833` was canceled only after successful independent verification.
`3090-gcondo` remains the preferred partition for future GPU work.
