# Public PushT checkpoint integration, 2026-10-03

Base: `sami-dev`, `4b60f01ec17152f13bcc0bf92ba8cba097d1c56a`.
No registration promotion: the combined semantic matrix remains 39/42,
or **35/38 excluding Craftax**. This checkpoint closes public CPU integration;
it does not establish accelerated GPU rendering or universal online speedups.

Public `PushT` and `PushTDiscrete` now use source-aligned official Pymunk 7.2.0
physics and complete numeric replay State. The earlier approximate solver and
compact renderer remain in `stable_envs.envs.pusht` for historical evidence.
They are not the classes selected by public construction.

## Evidence

`cpu/result.json` records 74 independently checked cases per registration:
three seeds, B1/B2, prefixes 0/100/1025, plus all 8 agent × 7 block shapes.
Source engines advance normally and are never restored. Checkpoints pass
through numeric NPZ files (`allow_pickle=False`) into a different public
instance. Suffixes compare observations, rewards, terminal flags, internal
physics/task/RNG diagnostics and frames. Eight-action chunks compare selected
states, validity, task outputs and frames against source lane advancement.

`cpu-verification.json` independently recomputes the retained raw witnesses:
**2,424 transitions and frame pairs per registration, 4,848 total, all exact**.
These are new bounded public gates; prior private candidate counts must not be
added as if they were the same candidate or independent public acceptance.
`pusht-public-v3-unit.xml` records **356 passes / 117 skips** on the Mac.
CUDA skips are not device passes.

`cpu-terminal/` adds 24 forced mixed-terminal cases (12 per registration), with
all/reverse/sparse/empty selectors. The finished lane advances exactly once,
the other advances eight times; every selected checkpoint restores and renders
correctly, and a subsequent public call matches source post-terminal stepping.
`cpu-terminal-verification.json` recomputes validity/task/frame witnesses.

The integration run passed 80 tests with one historical-profile skip and three
deselections. The first broad attempt preserved here failed only on two absent
MiniGrid dependency cases. The scoped rerun excludes those and the unrelated
Craftax-only case; it is not a claim that all optional-family tests ran.

`candidate_manifest.json` freezes Oscar's exact submitted files. Job **6993756**
uses `run.sh`, the pinned upstream checkout and isolated Pymunk 7.2.0 dependencies.
It runs public/history/shared API tests and repeats the source gate with CUDA
State/frame storage. It is pending; submission is not acceptance.
After this archive was frozen, package metadata separated the historical
`gym-pusht` reference extra from the actual migrated-task runtime extra.
Production simulator/render files remain byte-identical to the submitted files.

`prior-history-cuda-6993082/` records the earlier candidate's 344 test passes,
zero skips, on gpu2114. Its frozen manifest is in the previous campaign's
`pusht_history/candidate_manifest.json`. It does not certify the new public code.

## State and renderer details

Schema version 2 adds a finite numeric cache for visible Pymunk contact segments.
Up to two contacts per shape pair, at most ten supported shapes, gives 90 slots;
overflow fails explicitly. Restoration recomputes and compares this cache as
part of the complete State. Version 1 private candidates are not silently loaded.

The CPU renderer builds a separate scene from reset geometry, then installs
current poses, current appearance, action marker and contact segments. Ordinary
render calls do not replay the history. A unit test forbids stepping both live
and warmed render engines while rendering saved projections. The live simulator
is not mutated. State export/restore still has O(history) cost.

The public backend is explicitly `pygame`. `device='cuda'` means CUDA tensor
storage and frame transfer; it does **not** mean GPU rasterization. The old
renderer throughput table does not describe this new public path.

Continuous actions are relative controls in [-1,1]. Discrete actions are two
polar indices in [0,16). Both accept [B,A] and [B,T,A], with shared slice-only
state/frame selection. Persistent configuration validates the whole batch
before installation. The sampled flat Variation path respects integer colors
and the block-shape space's nonzero start.

## Remaining before full registration acceptance / v0 closure

- Fetch and verify the new CUDA State/output gate.
- Retain distinct lifecycle/configuration and source-profile coverage; mixed
  terminal CPU closure is recorded above and is not a CUDA pass.
- Preserve the separate accelerated renderer and online performance backlog.
- Rocket's public CPU state checks already pass; its 96-pixel CUDA rendering
  failure remains a separate blocker. No claim that all renderers are complete.
- A wheel build, isolated wheel import/branch smoke and dependency-resolution
  dry run pass. These are not a release or a supported-platform CI pass.

Run the source gate with `scripts/research_replay/pusht_public_gate.py`, then
`scripts/research_replay/verify_pusht_public.py`. Both require fresh output paths.
The source pin and exact dependency/source/production hashes are in the reports.
