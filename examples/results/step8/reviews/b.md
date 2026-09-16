# Independent review: B — Fetch VariationMaps

Verdict: **BLOCK**. Reviewed `step8/variations-fetch` at
`ce1fcb3d77af55812e55aadf78ae4a0d87922978` against baseline
`43ac95238f7472137619f3e69e7636e11dd088b1`, 2026-09-16.
Reviewer: fresh independent `review_b` agent. Author worktree was clean and was not edited.

The family table is sensibly shared across twelve IDs, and the changes do not touch renderer hot paths.
The exact 1.25 light conversion and the distinctions among appearance, reset/state, dynamics,
and dead camera leaves are supported by code and the supplied measurements. This is useful delivered
work to finish and review again, not a request to reimplement it.

## Required corrections

1. Recover the actual clean GPU evidence or rerun it on Oscar. `findings.md` says job 6452376 fixed
   both provenance and fixture errors, but the committed `gpu_evidence.json` still records job
   6451948 with `git_sha: null` and `git_branch: null`. Committed `unit_tests.txt` still ends with
   `8 passed, 24 skipped, 24 errors in 10.69s`. The corrected evidence script exists, but the claimed
   corrected results do not exist in this commit. Record baseline and tested source SHA, and prove
   the tested implementation matches the candidate. Do not simply edit the old artifact's SHA.
2. Run the required affected frozen-corpus replay and attach its results. `pixel_parity.json`
   reports identical pixels for four scenes, but `pixel_parity.py` samples eight random states per
   scene using `sample_states`; it never loads the Step-4 frozen Fetch corpora. The parity artifact
   names branch SHA `a835d9a6ad7511eecf0a00b1377e0562e272adcd` and only a baseline directory, not its
   verified SHA. It supports a limited unchanged-default smoke check, not merge-gate item 5.
3. Resolve wrapper discovery centrally in `worldmodel.check_mapping`, with a regression test
   through the public function. Its current `.unwrapped.variation_space` access (line 302 on B)
   fails for FetchWrapper, as the author correctly reports. The family helper walks wrappers but
   the public API never calls it. This is a stable-envs integration defect, not an upstream defect.
   Cover resource closing on failure as well. Coordinate ownership with A/D; do not independently
   overwrite the shared file.
4. Resolve the explicitly escalated partial `background.color` support. `VariationMap.pairs()` and
   `check_mapping` classify the floor-only mapping as covered, while the artifact classifies it as
   `mapped`; the unimplemented skybox portion changes an additional 61.7–73.0% of reference pixels.
   Documentation acknowledges this, but automatic completeness claims cannot treat it as full
   support. Make the limitation visible in generated classification/Gate-F/support output, and
   record the coordinator/user decision required by plan section 4 if feasible required behavior
   is to remain unsupported. No renderer rewrite or contract weakening is authorized by this review.
5. Make registered-default terminology precise in downstream reports. The measured equality is
   stable-envs `default_style()` versus `style_from_variation(env, {})`, and an inverse mapping
   round-trip. It is not equality to the registered SWM wrapper's default reset appearance:
   the author's own measurements show different leaf defaults and material unbinding. Record an
   explicit Gate-F status for that behavior rather than counting these tests as SWM default parity.

## All ten coordinator merge-gate items

| Item | Status and evidence |
|---|---|
| 1. Scope | PASS: twelve Fetch IDs; one declarative family map, registration, tests/docs/evidence. No renderer/performance implementation change. |
| 2. Before/after diagnosis | PASS with reporting corrections above: previously no maps, now all 123 offered leaves classified. The partial background mapping and SWM default mismatch remain explicit open issues. |
| 3. Targeted unit tests | PARTIAL: independently reran static tests, 8 pass. Live-wrapper/GPU coverage is not independently reproduced; committed GPU unit log contains 24 errors. |
| 4. Affected Gate A/F | BLOCK: public `check_mapping` still fails for Fetch; GPU variation isolation evidence lacks tested SHA; no complete affected Gate-A result provided. Default/partial behavior needs truthful Gate-F reporting. |
| 5. Frozen-corpus replay | MISSING: random-state baseline comparison is not frozen-corpus replay. Existing Step-4 Fetch state corpora are present and can be used. |
| 6. Pixel movement | PARTIAL: no renderer code changes; four-scene random-state artifact says IDENTICAL. Frozen/default-path evidence needs complete provenance. New nondefault mapping intentionally changes appearance and background is incomplete. |
| 7. Performance regression | NOT APPLICABLE: no renderer hot-path changes; no new optimization work requested. |
| 8. Independent review | BLOCK: this review. Only a later explicit APPROVE permits merge. |
| 9. Semantic follow-up | No broad downstream rerun needed for unchanged default render implementation. Preserve prior claims only after replay; do not extend them to SWM background/default behavior that is not reproduced. |
| 10. Dashboard | Coordinator must record this BLOCK, artifact discrepancies, wrapper defect, and background/default decisions before merge. |

## Independent local checks and limits

- `/Users/samib/se6-venv/bin/python -m pytest tests/unit/test_variation_maps_fetch.py -q -p no:cacheprovider`:
  **8 passed, 48 skipped**.
- `MUJOCO_GL=glfw /Users/samib/se8-fetch-venv/bin/python -m pytest tests/unit/test_variation_maps_fetch.py -q -rs -p no:cacheprovider --tb=short`:
  **8 passed, 48 skipped**. All 24 live-wrapper checks skip due to construction `AssertionError`;
  all 24 GPU checks skip because `mujoco_warp` is absent. Thus the claimed local 32-pass run is
  not reproducible from the committed tests/current environment without the reported external shim
  or an appropriate supported dependency environment. No shim was installed or upstream file edited.
- Initial unconfigured run hit `MUJOCO_GL=egl` inherited on macOS and failed 24 imports; rerunning
  with the documented GLFW setting removed that environment issue. It is not a new renderer defect.
- The broad `except Exception: pytest.skip(...)` around wrapper/environment construction can hide
  an actual regression on a supposedly supported certification environment. The certification run
  must explicitly assert expected non-skipped coverage; a green all-skipped capability tier is not
  evidence for Gate F.
- No GPU computation was attempted on the Mac, no cluster jobs were submitted/cancelled, no merge,
  push, oracle changes, or edits to the author's branch were performed.
