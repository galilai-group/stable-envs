# Source-helper and native acceptance cleanup

Candidate: `d13741c4b2963a2f585d73701d4306d760efefdb` on `sami-dev`.
Baseline: `f49a6816e9488e0c94e5d083ea28532842efff02`.
Status: **CPU validated; CUDA validation blocked by Oscar SSH authentication.**
This is a test-infrastructure checkpoint, not a new registration promotion or
renderer/performance measurement. The previously accepted 38-registration scopes
and evidence remain intact.

## Changes

Independent Fetch, PushT, Rocket and Cube/Scene capture lives in
`tests/helpers/reference/`. The 38 shared profiles no longer import research
scripts or execute archived Python. Active benchmark consumers import these
helpers and include the extracted dependencies in their new provenance records.
The historical Cube/Scene Python and outputs remain untouched.

One native oracle replaces two serializers. One acceptance module replaces two
large gate drivers; the CLI paths remain small entry points. The shared native
profile now requires 72 cases, including 16 specialized boundary, action,
Variation and rejected dtype-control cases. The duplicate integration entry
points are removed; their five frozen-corpus tests remain and no longer require
an installed source package just to replay disk data. An architecture check
prevents source helpers from importing research drivers or loading archived code.
Craftax is excluded.

Maintained Python: **1,144 lines added, 1,191 removed; net −47 lines**.
This includes promoting the archived Cube/Scene oracle into maintained test
support. No production file changed. This cohort improves dependency direction
and removes duplicate native orchestration; it is not a large repository-wide
code reduction.

## Validation

Final shared CPU runs verify the exact committed candidate-file manifest:

| Profile | Registrations | Required | Passed |
|---|---:|---:|---:|
| Native | 5 | 72 | 72 |
| Fetch | 12 | 192 | 192 |
| PushT | 2 | 24 | 24 |
| Rocket | 1 | 16 | 16 |
| Total | 20 | 304 | 304 |

Additional CPU validation: 48 Cube/Scene source-oracle tests; 62 infrastructure,
coverage and failure-control tests; 149 affected native/benchmark consumer tests;
and five preserved frozen-corpus tests. Five CUDA variants in the consumer suite
are explicitly skipped locally and included in the pending Oscar plan. There are
no skips in the four required CPU profiles. The documentation build also succeeds,
with the same four pre-existing historical-link warnings.

Migration witnesses:

- All 28 extracted functions, classes and constant definitions have identical
  Python ASTs to their original definitions. The only AST normalization removes
  the old Cube/Scene `previous.` module qualifier.
- Native old/new State serializers match exactly for all five domains and seeds
  0, 17 and 42: reset, 32 source steps, restoration into different histories,
  eight-step continuation and unseeded reset (15 scenarios).
- All 40 native acceptance records, five physical-Variation records and the
  rejected dtype-control result match the old drivers. The old scalar comparisons
  predate `states`/`state_valid`/`valid` response fields. The comparison proof checks
  those captured states and validity before removing only that metadata; the new
  scalar witnesses request `states=None, frames=None`. Source values still compare
  exactly. The original interface failures are preserved. The historical driver's
  package-version label uses the pinned checkout's `pyproject.toml` because the
  local source is loaded directly, without installed distribution metadata.
- Cube/Scene old/new source oracles match exactly across 42 family/seed/task
  scenarios and 12 actions each, including Variation, engine/cache/model fields,
  RNG ownership and reset-settling actions/sequences.

The first exploratory native run failed on obsolete scalar capture assumptions;
the first Rocket attempt was blocked by missing local PyYAML. Both outputs are
retained, followed by passing runs. Missing dependencies are not counted as passes.

## Evidence and pending work

`summary.json` and `profiles.csv` distinguish fresh runs from pending profiles.
`evidence.tar.gz` contains sealed final CPU runs, logs, the extraction and equivalence
proofs, baseline helper copies, failed exploratory outputs, and the prepared Oscar
runner/manifest. `SHA256SUMS` seals this report. The candidate source archive is
reproducible from the commit rather than duplicated here; `submission/prepare.py`
in the evidence archive records its construction. Its archive hash is in the summary.

Oscar authentication was retried and still returned
`Permission denied (keyboard-interactive)`; no new jobs were submitted. Reconnect
with the established control socket, reread the Oscar skills, regenerate/upload
the candidate archive and run all seven profiles on CPU and CUDA. The plan requires
**1,184 profile cases** (592/device), plus all 10 classic-export CPU/CUDA cases.
Prefer `3090-gcondo`; the prepared `gpu`/`norm-gpu` template is an authorized backup,
not an RTX 3090 claim. Use new output directories and record actual hardware and
terminal Slurm status. No new CUDA acceptance is claimed in this checkpoint.

Remaining cleanup: other families' specialized acceptance orchestration; shared
process-adapter plumbing; certification monkeypatching; and installation/CI/docs
alignment. None requires redesigning the public environment API.
