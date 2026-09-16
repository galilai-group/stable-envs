# Independent Step-8 A review

Verdict: **APPROVE-WITH-CHANGES**. This is not permission to merge under coordinator plan §18.

Reviewed commit: `063f2168ebcc563cd7f433c99dc3d2140d25530c` on `step8/variations-dmc-control`.
Baseline: `43ac95238f7472137619f3e69e7636e11dd088b1`.
Reviewer: fresh independent `/root/review_a`, 2026-09-16.
Author worktree was clean before and after review; no author files were modified.

## Reproduction and findings

The final classification artifact is current. Running `build_classification.py` with the committed
`measure/` directory and live SWM into `/private/tmp/step8-review-a.7LYY0z/leaf_classification.json`
produced a byte-identical file (`diff -u` exit 0). Counts are 17 IDs, 18 cells, 161 distinct per-ID
leaves: 46 mapped, 94 unsupported, 21 dead upstream, 0 not applicable. Running `build_doc.py` into
the same temporary directory reproduced the committed document's generated table section
(line 334 onward) byte-for-byte. The handoff's concern about incomplete artifact regeneration is
therefore resolved for the JSON and generated tables.

The hand-written document still contradicts that taxonomy:

- `docs/research/step8_variations_dmc_control.md:186`: remaining density/mass/friction leaves
  are called `not_applicable`, but the final maps classify them `unsupported`.
- The same document at line 192 calls other DMC gravity leaves `not_applicable`; final maps and
  tables classify them `unsupported`.
- The module's narrative and findings say twelve leaves never reach the model, but their list
  contains eleven never-read leaves plus five explicit-mass-overrides-density leaves, i.e. sixteen.
  With five hidden target leaves, the actual dead count is twenty-one. Correct the stale prose
  counts (including “two more leaves” when referring to two groups containing five leaves).

Independent test command, run from A's worktree:

```text
MUJOCO_GL=glfw PYTHONPATH=. /Users/samib/se8-vm-venv/bin/python -m pytest -q tests/unit/test_variation_maps_dmc_control.py
138 passed, 3343 warnings in 26.04s
```

Both the first sandboxed test attempt and generator attempt exited 134 before output. Required
sandbox escalation resolved this; these were local CPU/GL checks, not CUDA checks. The successful
suite includes live leaf completeness, defaults, classic-control official pixel equality,
isolation, round trips, model light behavior, dead-leaf probes, and pre-Step-8 compatibility.
No Oscar jobs were launched and no GPU test result is claimed.

## Ten-part merge gate

| Gate | Review result |
|---|---|
| 1. Explicit scope | PASS: 5 classic-control IDs + 12 DMC IDs, additive mapping support and shared conversion helpers. No renderer changes. |
| 2. Before/after diagnosis | PASS with documentation corrections above: missing maps become explicit measured classifications, including physics/camera coupling and documented range gaps. |
| 3. Targeted unit tests | PASS: independently reproduced all 138 targeted tests with live SWM. |
| 4. Affected Gate-A/Gate-F checks | PARTIAL: targeted default/variation/isolation/round-trip checks pass. No separately recorded Gate-A/Gate-F run is in A's packet. Coordinator must record the affected-check evidence and applicability, including whether unchanged GPU renderer gates carry forward. Full 42-ID Gate F remains Workstream D. |
| 5. Affected frozen-corpus replay | PENDING applicability record: no renderer, cell, testing harness, or frozen corpus is changed by A. The same states/styles reach byte-identical implementation by inspection, but A contains no fresh frozen-corpus replay result. Record explicit carry-forward reasoning or run the affected replay; do not imply a replay occurred. |
| 6. Pixel movement classification | PASS within stated scope: no renderer/default implementation changed; existing default styles retain baseline rendering. New mappings deliberately select different existing styles for five SWM registered appearances. Classic recolor non-default differences and light-range gaps are documented. |
| 7. Performance regression | NOT APPLICABLE to renderer hot paths: only mapping/setup functions changed. No renderer throughput claim is added. |
| 8. Independent review | This report: APPROVE-WITH-CHANGES until listed corrections and gate records are complete. |
| 9. Semantic follow-up | PASS: no renderer information changed, so no downstream semantic rerun required by this patch. |
| 10. Dashboard update | PENDING coordinator: current row still says unreviewed/delivered-unreported and lists dmcontrol.py/classic_control.py as affected although neither changed. Record this review and actual evidence. |

The `floor.color` limitation is an explicit coordinator decision candidate in A's findings §6.
This review does not authorize accepting a feasible semantically required leaf as unsupported;
record existing authorization or resolve that decision before claiming Step-8 closure. No
contract weakening or oracle re-freeze is requested by this review.

Recommended disposition: retain implementation, correct stale prose, complete explicit merge-gate
applicability/evidence and dashboard records, then request re-review. Do not reimplement A.
