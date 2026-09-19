APPROVE

# Independent review — Step 9 Workstream K (test layout)

Reviewed clean `step9/testing` HEAD `daea98c7e5738db442614d0305a2d50417b812f7` against
integration parent `786993b` (`git diff 786993b..HEAD`). Step-8 behavioural baseline
`21cb30678b7a7d0f3344b59f176f33d3f3b09ffc` is not in this delta.

Nothing was edited, committed, pushed, or reconfigured except this review file.

## Verdict

**APPROVE.** This is a mechanical layout move. The public measurement library stays in
`stable_envs/testing/`. `stable_envs/rollout.py` stays. Certification assertions and golden JSON
now live under `tests/certification/` and `tests/fixtures/`. Directory markers auto-apply.
`Path(__file__).parents[N]` and the two golden-file readers were updated. Assertions, thresholds,
and corpora are unchanged. `git log --format=%B 786993b..HEAD` contains **zero** `Co-authored-by`
strings. Author on all four commits is `sami-bg <samibghanem@gmail.com>`.

## Scope

`git diff --shortstat 786993b..HEAD`: 33 files, +328 / −42. `git diff --check` is clean.

Fifteen `git mv` operations (R100 except two R099 comment/ROOT edits):

| before | after |
|---|---|
| `tests/unit/test_{certification,declarations,provenance,step3,step4,step5,step6_report,step8_*}.py` (11) | `tests/certification/` |
| `tests/swm/test_gate_f_matrix.py` | `tests/certification/test_gate_f_matrix.py` |
| `tests/test_safety_net.py` | `tests/integration/test_safety_net.py` |
| `tests/data/{prng_threefry,craftax_full}_golden.json` | `tests/fixtures/` (byte-identical) |

`tests/swm/` and `tests/data/` are gone. `tests/{unit,integration,gpu,certification,fixtures}` plus
`tests/conftest.py` is the plan §14 tree.

Product files in the delta: one comment path in `stable_envs/testing/gate.py`. `stable_envs/rollout.py`
is not in the diff. No renderer, env, threshold, corpus, or assertion body changed.

The other edits are pytest `testpaths = ["tests"]`, the directory-marker hook, ROOT/fixture path
rewrites, and launcher/docs strings that would otherwise miss the moved files
(`docs/testing.md`, `scripts/run_{unit,smoke}_tests.sbatch`, four `scratch/` launchers,
`scratch/step4/bounds.py`, the two golden writers).

## Checklist

| check | result |
|---|---|
| `stable_envs/testing/` not moved | package still present; only a comment path in `gate.py` |
| `stable_envs/rollout.py` stays | untouched |
| `tests/certification/` and `tests/fixtures/` exist | 12 assertion modules; 2 golden JSON, R100 |
| markers auto-apply | `tests/conftest.py::pytest_collection_modifyitems` stamps `unit` / `integration` / `certification` / `gpu` from the folder after `tests/`; the three GPU oracle modules also get `certification` |
| ROOT / `parents[N]` updated | certification files stay at depth 2, so `parents[2]` is still the repo root; `test_safety_net.py` `parents[1]` → `parents[2]`; unit golden readers `parents[1] / "data"` → `parents[1] / "fixtures"` |
| no assertion / threshold / corpus changes | the two R099 files change one comment and one `parents` index; fixtures are identical |
| no `Co-authored-by` in `786993b..HEAD` | count 0 |

Moved certification modules have no leftover `pytestmark = pytest.mark.unit`, so `-m unit` cannot
select them. Existing per-file marks (`unit`, `gpu`, `integration`, `reference`, `smoke`, `selftest`)
are kept. `slow` remains registered in `pyproject.toml` and is not a directory bucket.

## Marker collect counts

Findings table (after, this host, no CUDA) is internally consistent and independently checked
without re-running the suite:

| bucket | after | how confirmed |
|---|---:|---|
| `tests/certification/` | 285 | AST + literal/named parametrize expansion (240 functions; extras from `REQUIRED_VISUAL_FIDELITY` 7, learned-decoder 7, malformed versions 7, `REFERENCE_JUDGED_CHECKS` 3, Step-3 ladder 9, Gate-E missing keys 5, TRACKCOM × 4 = 16) |
| `tests/integration/` | 18 | 12 safety-net + 6 worldmodel functions, no parametrize |
| `tests/gpu/` | 104 | directory unchanged in the diff |
| `tests/unit/` | 1836 | 2116 − 280 moved certification files |
| `pytest` (testpaths) | 2243 | 1836 + 285 + 18 + 104; the 11-test drop vs 2254 is `scratch/step9/packaging-ci/test_packaging.py` leaving `testpaths` |

Arithmetic the findings state: 2116 − 1836 = 280 left `tests/unit/`; 280 + 5 Gate-F = 285
certification; 6 + 12 = 18 integration. `-m unit` / `-m integration` / `-m certification` match
those directory counts on CPU because the hook is the only mark on the moved files and the three
GPU oracle modules skip at import without CUDA (`-m "certification and gpu"` = 0 here). The
`-m unit` jump 616 → 1836 and `-m gpu` jump 55 → 104 are the hook covering files that forgot
`pytestmark`.

I did not re-run the prescribed 1721-pass suite. No path bug showed up. The one reported failure
(`test_a_c_level_exit_mid_test_is_treated_as_a_crash_not_as_a_test_failure` timeout) is in a file
whose only edit is `parents[1]` → `parents[2]`.

## Live leftovers that are not bugs

- `stable_envs/envs/ogbench/manipulation.py` still names `tests/unit/test_declarations.py` in two
  comments. Comment only; env file is out of scope.
- Frozen Step-8 research pages and earlier Step-9 findings still cite `tests/unit/test_step*`,
  `tests/swm`, `tests/data`. Historical on purpose.
- `docs/research/certification.md` still cites `tests/unit/test_certification.py`. That page is
  frozen Step-2 policy; Workstream M owns living docs. Operational `docs/testing.md` was updated.

## Findings SHA table

The findings commit list (`4f75633`, `249ad75`, `65b28c9`) does not match reachable SHAs
(`c52519e`, `ba6fa05`, `d39c992`, plus this findings commit `daea98c`). Subjects match. Not a
layout defect.

## Coordinator requests (from findings; not blocking)

1. Freeze the two rulings in `WAVE0_CONTRACT.md`: `stable_envs/testing/` stays the public library;
   `rollout.py` stays as `snv.sample` / `BasePolicy`.
2. Dashboard row for K.
3. L's `pytest tests/unit -m unit` now collects every unit file (1836 here).
4–6. Stale comments, the safety-net orphan, and historical path strings: leave them.

## Commits reviewed

| SHA | subject |
|---|---|
| `c52519e` | step9: put certification assertions and golden fixtures in tests/ |
| `ba6fa05` | step9: stamp directory markers so pytest -m buckets work |
| `d39c992` | step9: run the suite through the new marker commands |
| `daea98c` | step9: record the testing-layout rulings and marker counts |

## Coordinator note

Independent review confirmed APPROVE. Already merged into `step9/integration` at `491b1c1`
(dashboard record `2bec3a9`). This file replaces the shorter coordinator write-up with the
independent review text.
