# Independent review (resubmission): B — Fetch VariationMaps

Reviewed SHA: `84c40e195efc6341d0272a0d4a76e2f7290c7af3` (branch `step8/variations-fetch`, worktree
`/Users/samib/stable-envs-step8-variations-fetch`), a merge of `step8/integration` at `cde1d9c`
(Workstream A, already approved/merged). Baseline `43ac95238f7472137619f3e69e7636e11dd088b1`.
Review scope: `git diff cde1d9c 84c40e1` only (17 files, +7304/-1). A's files are not under review.
Reviewer: fresh independent agent, no subagents used. Author worktree was clean and was not edited,
committed, stashed, reset, or pushed. No Oscar/SLURM access used.

**Verdict: APPROVE**

## The five corrections, verified against artifacts

1. **Provenance — VERIFIED.** `gpu_evidence.json` and `pixel_parity.json` both carry
   `git_sha`/`branch_sha` `5940eb6866e2cb0a91c7589f2f0aca1d58997a43` (non-null), plus job id
   `6443480`, host `gpu2264.oscar.ccv.brown.edu`, GPU `NVIDIA GeForce RTX 3090`, driver, CUDA, and
   package versions. Independently confirmed `5940eb6` is an ancestor of `84c40e1` and
   `git diff --stat 5940eb6 84c40e1 -- stable_envs tests` is empty — the tested tree is the reviewed
   tree. `unit_tests.txt` reads `10 passed, 49 skipped in 6.76s`, matching every place that count is
   claimed (findings.md, merge-gate table). Grepped all committed JSON/txt under
   `examples/results/step8/variations-fetch/` for `/oscar` — none found (the raw kept stdout log
   `gpu_job_6452376.out.txt` contains `sboughan` in an `ls -l` transcript, which is an audit-trail
   artifact explicitly kept for correction-1 history, not a path claim in a machine-readable
   artifact).
2. **Frozen-corpus replay — VERIFIED.** `frozen_corpus_replay.json` references
   `examples/results/step4/states/Fetch{Reach,Push,Slide,PickAndPlace}.npz` (64 states each). I
   independently recomputed the corpus digest with Step 4's own recipe
   (`shape` + `dtype` + bytes, i.e. `conformance.frames_digest`) against the actual committed `.npz`
   files and got exact matches to both the artifact's recorded `states_sha256` and to Step-4's own
   `examples/results/step4/conformance/fetch.json` record, for all four scenes (`f5a51968…`,
   `93d8ca80…`, `17375ce1…`, `ec78e35a…`). `frozen_corpus_replay.py`'s `_sha()` is a verified
   line-for-line reimplementation of `conformance.frames_digest`. Branch-vs-baseline is
   byte-identical (maxdiff 0) and equal to the frozen `digest_ours` on all four scenes per the
   artifact. I independently verified the five `git ls-tree` identity claims at baseline `43ac952`
   vs the actual tip `84c40e1` myself (not trusting the artifact's own stamp): `stable_envs/envs`,
   `stable_envs/render`, `stable_envs/testing`, `examples/results/step4`, `examples/results/step7`
   are all byte-identical tree objects between baseline and tip. One provenance nit: the artifact's
   own stamped `branch_tip_sha` (`13ef3789…`) predates the current tip by one commit range that
   touches `stable_envs/compat/variation_maps/fetch.py` and `worldmodel.py` (97 and 190 lines,
   respectively — metadata/classification changes from correction 3/4, not render-path code); this
   does not undermine the replay's conclusion because the five ls-tree ids I checked directly against
   the true tip (not the artifact's stamp) are the ones that can reach a rendered Fetch frame, and
   they are identical. Not a defect, but the coordinator should note the artifact's self-reported SHA
   trails the tip by a few metadata-only commits.
3. **`check_mapping` wrapper discovery — VERIFIED.** `git diff cde1d9c 84c40e1 -- stable_envs/compat/worldmodel.py`
   is empty (byte-identical to integration). `git diff cde1d9c 84c40e1 -- stable_envs/compat/variation_maps/__init__.py`
   is a one-line change: `_MODULES = ("dmc_control",)` → `("dmc_control", "fetch")`. B's own
   `variation_maps.fetch.swm_variation_space` delegates to `worldmodel.swm_variation_space` (no
   second wrapper-walk copy) and closes the constructed env in a `try/finally` on every path,
   including failure — read directly at `stable_envs/compat/variation_maps/fetch.py:261-271`.
   `test_check_mapping_reads_a_fetch_variation_space_through_the_public_function`
   (`tests/unit/test_variation_maps_fetch.py:298`) calls `WM.check_mapping(env)` — the public
   function — and asserts `background.color == "partial"`, `camera.angle_delta == "dead_upstream"`,
   `rendering.transparent_arm == "unsupported"`, `block.mass == "not_applicable"`, and
   `"background.color" in got["covered"]`. Caveat, disclosed by the author and confirmed by me: this
   specific test is skipped on every interpreter available to either of us — `se6-venv` lacks
   `stable_worldmodel`, `se8-fetch-venv` fails Fetch construction with the documented upstream
   numpy 2.5.3 / mujoco enum defect (finding 5), and Oscar's own unit run (`unit_tests.txt`) also
   shows `gymnasium_robotics is not importable`. So the regression test for correction 3 has not
   actually executed to a pass anywhere in this review chain. This is the same tier-2 gap the task
   brief says is a documented upstream limitation, not a B defect — and it is transparently disclosed
   in findings.md ("still open" item 3), not hidden behind a green run. I checked whether it could
   hide a real regression: the logic it exercises (`worldmodel.swm_variation_space`'s `.env` walk) is
   simple, is A's already-approved code, and is exercised statically by the same walk succeeding in
   `_live_space` calls in other tier-1/2 tests where dependencies are present; I did not find a
   pathway by which this specific skip conceals a live defect in B's delta.
4. **`background.color` partial — VERIFIED.** `stable_envs/compat/variation_maps/fetch.py` declares
   `_PARTIAL = {"background.color": "...the floor half only..."}` and every one of the twelve
   `VariationMap`s built by `_map_for()` passes `partial=dict(_PARTIAL)`. I loaded
   `leaf_classification.json` and confirmed `classification == "partial"` for `background.color` on
   all twelve ids (`FetchReach[Dense|Dict]`, `FetchPush[Dense|Dict]`, `FetchSlide[Dense|Dict]`,
   `FetchPickAndPlace[Dense|Dict]`), each carrying the measured residual (floor ~9.2-9.4%, skybox
   ~61.7-73.0%). Totals: `33 mapped / 12 partial / 12 unsupported / 15 dead_upstream / 51 not_applicable
   = 123`, `undocumented: 0` — I summed these myself from the artifact and they match. Docs table
   (`docs/research/step8_variations_fetch.md:31`) says `partial`. I ran
   `test_the_committed_classification_artifact_agrees_with_the_maps` as part of the full suite run
   below — it passed, and it explicitly asserts leaf-by-leaf agreement between the artifact and
   `VariationMap.classification()`, plus `per_classification["partial"] == 12`.
5. **Terminology / Gate-F MISMATCH — VERIFIED.** `findings.md` and
   `docs/research/step8_variations_fetch.md:227-228` both carry the explicit
   `Gate-F status: SWM registered-default appearance — MISMATCH (upstream)` line, distinguishing
   "stable-envs default vs empty swm config: identical" from "stable-envs default vs the SWM
   wrapper's reset appearance: NOT EQUAL (upstream)". `STABLE_ENVS_REQUIRE_LIVE_SWM` is documented in
   the test module and I ran it myself both ways in both venvs (`MUJOCO_GL=glfw`,
   `-p no:cacheprovider`):
   - `se6-venv`, no switch: `10 passed, 49 skipped`. With switch: `25 failed, 10 passed, 24 errors`.
   - `se8-fetch-venv`, no switch: `10 passed, 49 skipped`. With switch: `25 failed, 10 passed, 24 errors`.
   The switch does convert every tier-2/3 skip into a failure/error, so an all-skipped tier cannot be
   reported green under it — the core claim holds and is reproducible. One inaccuracy: findings.md
   states "25 failed, **9** passed, 24 errors" for `se6-venv` with the switch; I measured and
   reproduced **10 passed** twice on both venvs. Minor documentation slip (off by one on a passed
   count), not substantive — flagging for the record, not a blocker.

## `gpu_evidence.py` fix and non-empty unmapped sections

Confirmed at `scratch/step8/variations-fetch/gpu_evidence.py:106-116`: the harness iterates
`sorted(m.unmapped())` (the three typed buckets: `unsupported`/`dead_upstream`/`not_applicable`),
not the empty `swm_only` bucket the earlier defect measured. A zero-collection guard at
lines 136-152 requires `len(unmapped_leaves_move_nothing) >= (5 if Reach else 7)` and
`len(mapped_leaves_move_pixels) >= (3 if Reach else 4)` before the run can report PASS — an empty
section now fails rather than vacuously passing. The committed `gpu_evidence.json` shows non-empty
`unmapped_leaves_move_nothing` per scene: 5 leaves on Reach, 7 on each of Push/Slide/PickAndPlace,
matching the declared unmapped sets exactly.

## Ten coordinator merge-gate items (plan section 18)

| # | item | status |
|---|---|---|
| 1 | Scope | PASS — twelve `swm/Fetch*-v3` ids, one family map + package entry, no renderer/perf change (`git diff --stat` shows no file under `stable_envs/render`). |
| 2 | Before/after diagnosis | PASS — 123/123 leaves classified, 0 undocumented, both open issues (skybox residual, SWM default mismatch) labelled `partial` / `MISMATCH (upstream)` rather than left in prose. |
| 3 | Targeted unit tests | PASS — full required suite run below is clean; `test_variation_maps_fetch.py` is 10 passed/49 skipped on both local venvs, matching the committed `unit_tests.txt`. |
| 4 | Affected Gate A/F | PASS — `check_mapping` works for Fetch through the public function and A's file is untouched by B; Gate-F carries the explicit `MISMATCH (upstream)` row; frozen Step-4 digests replay identically (item 5). |
| 5 | Frozen-corpus replay | PASS — independently re-verified corpus sha256s against the committed `.npz` files and Step-4's own record, and the five `git ls-tree` ids against the actual tip myself. |
| 6 | Pixel movement | PASS — no renderer file touched (diff stat confirms); frozen-corpus and sampled-state replays both show 0/255 default-path movement on all four scenes; new mapped/partial leaves intentionally move pixels, measured per-leaf. |
| 7 | Performance regression | NOT APPLICABLE — no hot-path code changed. |
| 8 | Independent review | This report: **APPROVE**. |
| 9 | Semantic follow-up | NONE REQUIRED — default render path unchanged and proved so by replay; no claim extended to SWM default/background behavior, which is recorded as upstream mismatch. |
| 10 | Dashboard | COORDINATOR — record this APPROVE, the `partial` background.color classification and its post-Wave-1 aggregation, the Gate-F `MISMATCH (upstream)` row, and the five upstream findings. |

## Required test command

```
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q tests/unit -p no:cacheprovider -x \
    --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"
```

Tail:
```
------------------------------ stable-envs safety ------------------------------
resources: 10 EGL contexts created, 0 released, 10 live (peak 10); renderers 9 created / 0 closed
node: cis1b1zs2j3.ad.brown.edu slurm=? reason=none reference-renderer={}
948 passed, 186 skipped, 112 deselected, 25918 warnings in 88.00s (0:01:28)
```

No failures, no errors. `test_every_cited_scratch_path_resolves` was independently run standalone and
confirmed to fail for the documented pre-existing reason (other workstreams' `scratch/` paths not
present in this worktree, e.g. `scratch/step8/perf/*`, `scratch/step8/semantic/*`) — not a B defect,
correctly excluded from the required run.

## Remaining defects / notes (none blocking)

1. `examples/results/step8/variations-fetch/findings.md:190` states "25 failed, 9 passed, 24 errors"
   for the `STABLE_ENVS_REQUIRE_LIVE_SWM=1` run on `se6-venv`; the actual, reproducible count is
   **10 passed** (verified twice). Cosmetic — the substantive claim (switch defeats all-skip green
   runs) is correct and reproduced.
2. `frozen_corpus_replay.json`'s self-reported `provenance.branch_sha` (`13ef3789…`) is a few
   metadata-only commits behind the reviewed tip `84c40e1` (diff touches
   `stable_envs/compat/variation_maps/{__init__,fetch}.py` and `worldmodel.py`, not the render path).
   The replay's conclusion is unaffected because I independently verified the five renderer-relevant
   `git ls-tree` ids against the true tip directly, but the coordinator may want a re-stamp for
   cleanliness in a future round; not required for this merge.
3. `test_check_mapping_reads_a_fetch_variation_space_through_the_public_function` (correction 3's
   regression test) has not executed to a pass on any interpreter reachable in this review chain
   (skips on `se6-venv`, `se8-fetch-venv`, and Oscar's own unit tier per `unit_tests.txt`). This is
   the documented tier-2 gap (upstream numpy/mujoco defect), disclosed openly in findings.md, and per
   the review brief is not treated as a B defect. Flagged for visibility only.

The escalation ruling on the skybox residual (background.color partial) is a coordinator-level
capability decision already recorded (Sami, 2026-09-16) and does not block this workstream. The
tier-2 live-wrapper gap is a documented upstream limitation, not fixed and not required to be fixed
here.

## Verdict

**APPROVE.** All five required corrections from the prior BLOCK are verified against artifacts, not
prose: provenance is non-null and matches the reviewed tree; the frozen-corpus replay is real,
independently re-hashed, and matches both baseline and Step-4's own record; `check_mapping`'s public
entry point is fixed centrally (A) and exercised through the public API by a regression test (even
though that specific test cannot execute green on any interpreter available here — a disclosed,
non-B, upstream gap); `background.color` is truthfully `partial` everywhere generated and by-hand
counts agree; and the Gate-F `MISMATCH (upstream)` line and strictness switch are documented and
behave exactly as claimed, reproduced independently in both venvs. The required full unit suite run
is clean (948 passed, 0 failed, 0 errored). Two cosmetic issues noted above (an off-by-one passed
count in prose, and a slightly stale self-reported SHA in one artifact) do not rise to blocking
defects.
