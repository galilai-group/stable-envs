# Rocket production recovery: approval-blocked preflight

The matrix remains **39 VERIFIED / 3 INCOMPLETE**. There is no production
implementation or promotion in this checkpoint. No correctness or renderer
failure has been established. GPU acceptance is pending.

Starting and current HEAD: `3d0a799d329bb00fbecbf708e970be18a14ce611`, also the
exact commit containing `branch_replay_checkpoint.md`. Branch:
`repair/environment-semantics`; worktree:
`/private/tmp/stable-envs-semantic-recovery`. The starting worktree was clean.
Current documentation, scripts and evidence are **uncommitted**, because the
approval service blocked the requested local commit. No new commits; no push.

## Starting checkpoint and preservation

[Growth and starting record](semantic_recovery_evidence/rocket_production_start/checkpoint-and-growth.json):
179 added files, 2,591,455 added bytes; one modified index adds 601 net bytes;
2,592,056 net tree bytes. Read-only single-branch pack streams measured
98,773,284 bytes before and 99,380,093 after: **606,809 incremental packed bytes**.
Pack streams can differ from prior measurements as Git's existing packing changes;
this comparison measured both tips in the same run. Nothing was repacked or removed.

[Preservation audit](semantic_recovery_evidence/rocket_production_pending/preservation.json)
checks all 208 accepted artifact hashes, 112 production hashes, byte-identical
matrix/map, all 179 prior repository-seal entries (the index is append-only),
and unchanged existing reset/RNG claims in the additive 42-row metadata.
All previously tracked files except that index are unchanged. PushT is untouched.

## Reset clarification

[Reset semantics](reset_replay_semantics.md) and
[per-registration metadata](reset_replay_classification.json) preserve the
accepted scopes. Rocket's source requirement is
`SEEDED_REPLAYABLE_FRESH_UNSEEDED_ENTROPY`; this is not production acceptance.
No shared State/RenderState structure, protocol, or production code changed.

## Renderer preflight prepared

The immutable SWM task is from commit
`176d54532b3c0a251a4f509b74f2fdacc143a63d`, file SHA256
`5ede3640ce1e2add32b6b0a6c36e190bb1130e38bb258ab182af12a6fbedee23`.
The local installed copy matches that hash. Official PyFlyt 0.29.0 / PyBullet
3.2.7 provide the physics and source rendering. This script imports SWM solely
as a research oracle; no production SWM import was added.

[Source capture](semantic_recovery_evidence/rocket_renderer_preflight_source_v2/source.json)
contains **102 frames**, generated only through public reset/step/render:
seeds 0/17/42, float32/float64 controls, reset, early descent, near contact,
crash and post-terminal calls, successful landing and post-success calls,
and all-selected Variation with mixed controls. A consolidated numeric NPZ
holds the frames, observation prefix, actual material colors and cached link pose.
Repeated source renders are byte-identical. No native teleport, snapshot restore,
or raw physics calls were added by the collector.

A possible renderer discrepancy remains unclassified: at the default reset,
its velocity-derived prior link position differs from the actual cached link
position by about 0.12814 m. Ordinary tested flight/contact steps agree near
floating-point precision. SWM's reset goal-image save/restore changes the cache
boundary. This position difference alone is **not a visual-policy failure**.
The frozen renderer must be measured on Oscar against the captured real frames.
The existing arbitrary-pose reference bridge excludes some contact-band rows;
this preflight uses directly reached source frames and makes no such exclusions.

The prepared GPU script compares the frozen renderer at its registered 480-pixel
resolution against the existing mean-absolute/within-eight/SSIM limits, records
per-case groups, and checks singleton/batch equality and input nonmutation.
It is an early feasibility gate, not public full-State adapter acceptance.
Frozen renderer source and all historical baselines are unchanged. No GPU result,
job ID, terminal acceptance marker, or production rendering claim exists yet.

[Prepared bundle manifest](semantic_recovery_evidence/rocket_renderer_preflight_submission/prepared.json)
and [SLURM script](semantic_recovery_evidence/rocket_renderer_preflight_submission/preflight.sbatch)
are ready. Local bundle: `/private/tmp/rocket-render-preflight-3d0a799d`.
Remote scratch directory exists at
`/oscar/scratch/sboughan/se-semantic-recovery/rocket-render-preflight-3d0a799d`.
**The upload did not execute and no job was submitted.** The six required Oscar
skills were read before preparing commands. The sole existing job seen in the
queue was unrelated and was left alone.

## Validation and remaining work

- Core shared-State/harness/readiness/repository/registry/Rocket checks:
  **95 passed, 2 skipped**.
- Wider legacy Craftax/Variation renderer suite: **253 passed, 61 skipped,
  112 failed**. Running the identical command on an extracted starting-commit
  archive produces exactly the same 112 failing test IDs, with no new failures.
  Causes include legacy renderer-only State expectations, preexisting Variation
  map differences, unavailable classes and headless macOS CGL failures.
- Ruff, formatting, whitespace and strict documentation build pass.
- Preservation verifies; compact validation logs and comparison results are
  in the pending evidence directory.

Production files changed: **none**. A complete production Rocket State schema,
fixed ignition dtype, atomic replay adapter, production CPU gates and performance
measurements are **not implemented/run**. The earlier research schema and its
767,077-byte default numeric payload remain historical evidence, not a production
claim. No new State size, per-horizon restore timings, B16 measurements, rendered
end-to-end costs, or nine-leaf production Variation coverage is claimed.
The source capture's all-selected Variation is only a renderer preflight fixture.
All previous State-performance/rendered reports remain unchanged.

The next action is to restore approval-service access, commit the prepared
reset/start bookkeeping, upload the sealed bundle, recheck for active Rocket jobs,
and submit exactly one preflight job. Inspect both terminal status and the
script's result markers. If the renderer fails the required semantics, preserve
the smallest decisive witness and stop without promotion. If it passes, continue
the authorized task migration and production replay/acceptance/performance gates.
Do not begin PushT or change the frozen renderer/shared structures.

The automatic approval reviewer rejected two scoped local commit attempts, a
proposed research-script edit, and the GPU upload because its authentication
request returned **401 Unauthorized**. These actions did not execute. The script
edit was not retried through another mechanism. This is an infrastructure block,
not a request for new task authorization or evidence that Rocket cannot pass.
