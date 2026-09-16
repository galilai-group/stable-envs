# Step 8 takeover WIP — stopped at user request

Implementation stopped immediately on request. Agents were interrupted; no further implementation
or tests were started. This checkpoint is for review, not approval or Step-8 completion.

## Checkpoints

- Coordinator: `codex/step8-takeover-wip`, created from current HEAD
  `bd074d5ff48cdfe36886233289070456d73e5efa`. This document is part of its WIP commit.
- A corrections: `step8/reviewfix-a` at `0cbb136ed0a6ec728d78a9b1e69430fef2c64d94`,
  worktree `/private/tmp/stable-envs-step8-reviewfix-a`.
- C unfinished colors: `step8/variations-newcells-followup` at
  `69e74c48a020e524534025bade6871850375a6f3`, worktree `/private/tmp/stable-envs-step8-c-followup`.
- E certification: `step8/certification-followup` at `550cfc6b3993692cac02fea10bd3850ecfb394ab`,
  worktree `/private/tmp/stable-envs-step8-e-followup`.

All new commits use sami-bg <samibghanem@gmail.com>. Nothing was merged, rebased, reset, restored,
stashed, deleted, cleaned, or pushed. No Oscar job was submitted or cancelled by this coordinator.

## Changes and reasons

Coordinator updated `docs/research/step8_handoff.md` and `examples/results/step8/dashboard.yaml`
to resolve the stale integration reference: `bd074d5` is exactly `e915466` plus the handoff commit.
Independent A/B/C reviews are under `examples/results/step8/reviews/`; E review was interrupted
before completion. Exact initial C diff is preserved under `examples/results/step8/takeover/`.
The four `stop-*.txt` files there record pwd, status, branch, HEAD, worktree list, diff stat, and
full tracked diff for the coordinator and each new follow-up worktree before checkpointing.

A follow-up changes only narrative text/docstrings in
`docs/research/step8_variations_dmc_control.md`,
`examples/results/step8/variations-dmc-control/findings.md`, and
`stable_envs/compat/variation_maps/dmc_control.py`: correct stale taxonomy/counts and record
proposed merge-gate carry-forward reasoning. That reasoning has not received final review.

C follow-up copied the preserved authorized patch into an isolated worktree. Files:
`stable_envs/envs/ogb_manip.py`, `stable_envs/compat/variation_maps/newcells.py`,
`tests/unit/test_variation_maps_newcells.py`. It adds cube/floor color mapping, attempts to fix
custom-to-default cube-color leakage, and preserves original floor representatives during grouping
to avoid quantization round trips. It is unfinished and unverified; do not merge it.

E follow-up restores historical Gate-A rejudgment including family_specific checks without editing
the source measurements. Files: `scratch/step8/rejudge_gate_a.py`, `scratch/step8/report.py`,
`scratch/step8/refresh_census_diff.py`, `stable_envs/testing/conformance.py`,
`tests/unit/test_step8_gate_record.py`, `examples/results/step8/certification/gate_a_rejudged.json`,
`examples/results/step8/certification/census_class_diff.json`,
`examples/results/step8/support_matrix.json`, `docs/research/step8_support_matrix.md`.
Also fixes exclusion-only applicability when all sampled states lie outside the exclusion, and
refreshes a stale MiniHack classifier result. Generated records report historical Classic
resolution and Rocket color failures rather than retaining mandatory-only PASS claims.

## Verification performed

- A original delivered branch: independent 138/138 targeted CPU tests passed; generated classification
  JSON and documentation tables reproduced byte-for-byte. Narrative corrections not re-reviewed.
- B original delivered branch: 8 passed / 48 skipped in both available tested interpreters;
  live wrapper construction unavailable. Review BLOCK. Remote job 6452376 evidence found with
  Git SHA 4b3d09f; remote unit log is 8 passed / 48 skipped. No recovery file was committed locally.
- C original committed branch: 58 passed / 5 skipped, generator check passed; reviewer reproduced
  dirty patch's custom-to-default color leak with a CPU stub. Live local probe aborted 134.
  New C follow-up has no completed GPU/replay/performance verification.
- E new generators executed successfully: census changed=0, unknown resolved=2, remaining unknown=0;
  support matrix remains 42 IDs (40 pixel, 2 symbolic), now 34 PASS / 5 FAIL / 1 UNMEASURED
  pixel rows. This is historical rejudgment, not fresh certification. Targeted pytest invocation
  for test_step8_gate_record.py, test_step4.py, test_certification.py aborted exit 134 with no result.
- E reviewer independently reproduced the original matrix except HEAD stamp; remaining review
  and tests were interrupted. No final APPROVE exists for any follow-up.

## Boundaries, assumptions, and incomplete work

Original A/B/C/E branch tips were not changed. Original C dirty file is unchanged: binary Git diff
22,578 bytes, SHA256 b44808ed56e48bb2cfde5000959629eeab95b6437076a6c8cde59b22ace7ab60.
Coordinator documentation was initially edited in the integration worktree, then moved with the
worktree to the requested WIP branch. The `step8/integration` branch remains at bd074d5 with no
new commit or merge. Pre-existing untracked STEP_08_AGENT_BRIEFS/ and STEP_08_HANDOFF.md are
user material and remain untracked, unchanged, and outside this checkpoint commit.

Assumed prior cube/floor authorization permitted finishing those two leaves in an isolated copy;
broader model-array changes, contract changes, and re-freezes were not authorized or performed.
Assumed historical measurements should be rejudged conservatively without inventing missing
applicability/provenance evidence. Every follow-up still needs independent review.

Unresolved: Fetch wrapper-chain bug and partial skybox semantics; C tests/docs/artifacts and GPU
gates; OGBMaze checker-versus-flat mismatch (do not merely apply affine inversion); E join and
applicability review/tests; lit-color contract decision, feasible unsupported leaves, Craftax
provenance; Wave 2 D/F/G/H; all final certification. No merge approval was granted.

Oscar queue was inspected using the six required remote Oscar workflow instructions. Allocation
6443480 was still running gpu2264 at elapsed 8:04/12h; this is a timestamped observation, not a
promise it remains alive. Existing unrelated jobs and all prior allocations were left untouched.
