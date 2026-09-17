# Step 8 Workstream F — reference versions and provenance — independent review

Reviewed SHA: `f4164d30f1c994ca55f2686ac5bad3a5704524d0` (branch `step8/reference-provenance`,
worktree `/Users/samib/stable-envs-step8-reference-provenance`, parent `4210793` on
`step8/integration`).

**APPROVE**

---

## Merge gate (plan section 18), ten items

1. **Explicit workstream scope** — CONFIRMED. `git diff 4210793 f4164d3 --stat`: 32 files, all
   under `docs/research/step8_reference_provenance.md` (new), `examples/results/step8/provenance/*`
   (new), `scratch/step8/provenance/*` (new), `tests/unit/test_step8_provenance.py` (new), one
   `mkdocs.yml` nav line, `docs/research/step8_support_matrix.md` and
   `examples/results/step8/support_matrix.json` (regenerated join outputs), and
   `scratch/step8/report.py` (+64/-0-ish reference-arm branch). `stable_envs/`,
   `scratch/step4/certify.py`, `pyproject.toml`, and `examples/results/step4/` all show **zero**
   diff lines against the parent.
2. **Before/after diagnosis** — CONFIRMED. Reran `certify.py`'s `_richness` by inspection
   (`scratch/step4/certify.py:104`): tiebreaks on `(status==measured, n_measured, shard_or_"")`,
   i.e. lexicographic shard tag, and `"adhoc" < "dmc"` — matches the claimed mechanism verbatim.
3. **Targeted unit tests** — CONFIRMED. `tests/unit/test_step8_provenance.py` collects exactly 18
   cases as claimed; all pass (see command/tail below).
4. **Affected Gate-A/Gate-F checks** — CONFIRMED for Gate A (four trackcom cells, frozen judge,
   mandatory + family_specific, rejudged from unchanged measurements). Gate F untouched, correctly
   out of scope for this workstream.
5. **Affected frozen-corpus replay** — CONFIRMED. No Step-4 replay re-run (both arms are committed
   Step-7 shards read by sha256, verified below). The 3.13 evaluation independently replayed
   `OGBCube`/`OGBScene`/`DMCCheetahRun`/`FetchReach` under three interpreters, artifacts present
   under `examples/results/step8/provenance/mjwarp313/`.
6. **Pixel-movement classification** — CONFIRMED. No file under `stable_envs/` changed; the 3.13
   pixel differences are a measurement finding, not an adopted change (bump rejected).
7. **Performance regression check** — N/A, correctly noted (no hot-path code touched).
8. **Independent reviewer verdict** — this document.
9. **Semantic follow-up decision** — "none" is reasonable: no renderer pixel moved under the
   accepted change, so there is nothing for a DINO tripwire or Gate H to detect.
10. **Dashboard update** — left to the coordinator per the report; the numbers it hands off
    (`gate_a` 40 PASS / 2 UNSUPPORTED_BY_DESIGN, `pixels_changed: no`) reproduce independently
    (see §3 below).

---

## Verification detail

### 1. Contract integrity
```
git diff 4210793 f4164d3 --stat -- stable_envs/            -> empty
git diff 4210793 f4164d3 --stat -- scratch/step4/certify.py -> empty
git diff 4210793 f4164d3 -- pyproject.toml                  -> empty
git diff 4210793 f4164d3 --stat -- examples/results/step4/  -> empty
```
No frozen digest, threshold, or pin touched.

### 2. The trackcom claim
- `stable_envs/envs/dmcontrol.py:2939`: `_AT_VERSIONS = {"mujoco": "3.11.0"}`, published through
  `reference_versions_of`/`describe()` — matches the doc's citation.
- `scratch/step8/provenance/reference_arm.py` judges both `dmcvis.json` (adhoc/3.11.0) and
  `dmc.json` (dmc/3.12.0) with the unmodified `scratch/step4/certify.py:verdict`, and selects the
  authoritative arm via `stable_envs.testing.provenance.reference_version_mismatch(declared, prov)
  == []`. Re-ran a byte-patched copy (output path only) locally:
  ```
  DMCCheetahRun:    authoritative=dmcvis.json gate_a=PASS   (dmc.json: FAIL, check_resolution_consistency)
  DMCHopperHop:     authoritative=dmcvis.json gate_a=PASS   (dmc.json: FAIL, check_resolution_consistency)
  DMCHumanoidStand: authoritative=dmcvis.json gate_a=PASS   (dmc.json: BLOCKED)
  DMCWalkerWalk:    authoritative=dmcvis.json gate_a=PASS   (dmc.json: FAIL, check_resolution_consistency)
  ```
  matches `docs/research/step8_reference_provenance.md` exactly, including the claimed numbers
  (4.992/0.8820/0.6613 etc. — read directly from the committed `reference_arm.json` checks, not
  re-derived here since the judge and shard are both frozen/unmodified and the sha256s below
  confirm the shards are unaltered).
- Regenerated `reference_arm.json` into a temp dir with only the `OUT` path patched: diff against
  the committed file shows **only** `generated_at_head_sha` differs (`c2770c5...` vs `f4164d3...`),
  exactly the claimed "only a head-sha stamp" delta.
- Verified `source_sha256` for all 8 (cell, arm) entries in `reference_arm.json` against the actual
  bytes of `examples/results/step7/closure2/replay/{dmcvis,dmc}.json`: all 8 match.
- Root cause read directly from `scratch/step4/certify.py:104`:
  ```python
  def _richness(row: dict) -> tuple:
      n = sum(1 for g in ("mandatory", "family_specific")
              for v in (row.get(g) or {}).values() if v.get("status") == "measured")
      return (row.get("status") == "measured", n, row.get("shard") or "")
  ```
  confirms `"adhoc" < "dmc"` tiebreak is real, exactly as claimed.

### 3. Support matrix regeneration
Ran `scratch/step8/report.py --json-out <tmp> --md-out <tmp>` at HEAD (`MUJOCO_GL=glfw
/Users/samib/se6-venv/bin/python`). 42 rows; Gate A distribution: `{'PASS': 40, 'UNSUPPORTED_BY_DESIGN': 2}` —
matches the claim exactly. Diffed the regenerated 42-row matrix against the parent
(`git show 4210793:examples/results/step8/support_matrix.json`): **exactly 4 rows changed**
(`swm/CheetahDMControl-v0`, `swm/HopperDMControl-v0`, `swm/HumanoidDMControl-v0`,
`swm/WalkerDMControl-v0`), each moving `gate_a.status` from `FAIL`/`UNMEASURED` (parent) to `PASS`
(head) and each carrying a `DOCUMENTED_LIMITATION` / "reference/upstream drift" entry in
`limitations` plus a `reference_drift` block preserving the 3.12 verdict. No other row moved.

### 4. mujoco_warp 3.13 verdict
`examples/results/step8/provenance/mjwarp313_verdict.json` records all three conditions with
evidence matching the findings/research doc: texcoord guard `false`@3.12 / `true`@3.13; `DMCCheetahRun`
`exact_frac` 0.522233 < floor 0.528850 (frozen bound 0.533850, margin 0.005); digests move for
`OGBCube`/`OGBScene`/`DMCCheetahRun` (all `byte_identical: false`) and `FetchReach` cannot even build
under 3.12 or 3.13 (`gymnasium-robotics` assertion). `recommendation: "DO NOT BUMP"`. Provenance
recorded inline: `"hardware": "SLURM allocation 6443480, gpu2264, ... driver 575.57.08 ..."`,
`"venv": "$SCR/venvs/se8-mjwarp313"` — matches the claim (job 6443480, gpu2264, se8-mjwarp313).

### 5. Antmaze classification
Findings §3 cites `examples/results/step6/raycast3d/findings.md`, `docs/research/step6_optimization.md`,
`examples/results/step7/closure/findings.md`, `docs/research/step7_coverage.md` — all four exist in
the worktree. No new measurement was taken (consistent with "no re-run needed"); `AntmazeTeleportExplore`'s
frozen digest in `examples/results/step4/certification.json` is untouched (confirmed under item 1's
zero-diff on `examples/results/step4/`). Classified `DOCUMENTED_LIMITATION` (oracle hole), not a
re-freeze; re-freeze explicitly deferred per plan §19 (corrected implementation does not yet exist).

### 6. Paths and tests
No genuine absolute `/oscar` or `/Users` paths in committed files — the only string hits are the
literal filename `oscar_paired_tests.json`, a comment in `sanitize.py` describing the pattern it
strips, and the regex literal inside the path-hygiene test itself (all non-matches on inspection).

Commands and tails:
```
$ MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider \
    tests/unit/test_step8_provenance.py tests/unit/test_step8_gate_record.py
................................................                         [100%]
48 passed in 0.30s
```
(18 in `test_step8_provenance.py`, 30 in `test_step8_gate_record.py`.)

```
$ MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_repo_contracts.py
FAILED tests/unit/test_repo_contracts.py::test_every_cited_scratch_path_resolves
1 failed, 10 passed in 0.40s
```
Reproduced the **identical** failure (same missing-path list) by checking out parent `4210793`
(via the sibling `stable-envs-step8-integration` checkout, no edits) and running the same test —
confirms the workstream's claim that this failure pre-exists and is not introduced here.

---

## Defects found

None release-blocking. Two minor observations, neither blocking:

- `examples/results/step8/provenance/mjwarp313_verdict.json` has no top-level `"provenance"` key
  (provenance is inlined per-condition as `"hardware"`/`"venv"` strings inside
  `conditions.3_moves_nothing_unrelated.evidence` rather than a single structured block) —
  cosmetic only, all the required facts (job id, node, GPU, driver, venv) are present and correct.
- `scratch/step8/report.py`'s `main()` calls `a.json_out.relative_to(ROOT)` unconditionally for its
  final print, which throws `ValueError` if `--json-out` points outside the repo root (as it does
  when a reviewer redirects output to a temp dir for verification, `scratch/step8/report.py:615`).
  The files are written correctly before the exception; this only affects the final cosmetic print
  statement, not the artifact contents used in this review. Worth a `try/except` for reviewer/CI
  ergonomics but not a merge blocker.

## Verdict

**APPROVE.** All ten merge-gate items verified independently against the artifacts and against the
worktree's own code (not just re-reading the author's prose): contract integrity holds, the
trackcom join fix is real and reproduces byte-for-byte modulo the head-sha stamp, the support-matrix
regeneration changes exactly the claimed four rows with the drift preserved, the mujoco_warp 3.13
verdict's three conditions and DO NOT BUMP recommendation are evidenced and consistent with the
plan, the antmaze classification is provenance-only with no re-freeze, and the provenance/tests
items are clean.
