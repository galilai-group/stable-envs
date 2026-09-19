APPROVE

# Step 9 Wave 2 — Workstream K (tests/certification cleanup) — independent review

**Reviewed SHA:** `daea98c7e5738db442614d0305a2d50417b812f7` (`step9/testing`).
That tip is a trailer-free rewrite of author delivery `4bbcc56`; the trees are identical
(`8b181f8083992b081bf630e26b8f3d161c1bd1a7`). Parent is `step9/integration` `786993b`.
**Step-8 behavioural baseline:** `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`.
No product modules under `stable_envs/envs` or `stable_envs/render` were edited.

## Verdict

**APPROVE.** This is a layout and marker change. Certification assertions moved to
`tests/certification/`, golden JSON to `tests/fixtures/`, and the safety net to
`tests/integration/`. `stable_envs/testing/` remains the public measurement library.
`stable_envs/rollout.py` remains `snv.sample` / `BasePolicy`, not `env.rollout(policy)`.
`git log --format=%B 786993b..HEAD | grep -ci co-authored` is 0.

The one failed test in the author's prescribed run
(`test_a_c_level_exit_mid_test_is_treated_as_a_crash_not_as_a_test_failure`) is a
pre-existing supervisor timeout, not a path break from the moves.

## Checks

- `git diff 786993b..HEAD --stat` is moves plus conftest markers, `testpaths = ["tests"]`,
  and path updates in docs/scripts/golden readers. No threshold or corpus edit.
- Directory hook in `tests/conftest.py` stamps `unit` / `integration` / `certification` /
  `gpu`; the three GPU oracle modules also get `certification`.
- `Path(__file__).parents[2]` still reaches the repo root for files that stayed at
  depth 2; `test_safety_net.py` was updated from `parents[1]` to `parents[2]`.
- Marker collect counts in findings.md are consistent with the hook (unit 1836 because
  files that forgot `pytestmark` are now selected).

## Coordinator follow-ups (not merge blockers)

1. Record the two rulings on the dashboard: public `stable_envs.testing`; `rollout.py` stays.
2. L CI that used `pytest tests/unit -m unit` now sees every unit file. Intended.
3. GPU certification still needs a CUDA runner; `-m "certification and gpu"` collects 0
   on this CPU host by design.
