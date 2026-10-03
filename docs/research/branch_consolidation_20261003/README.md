# sami-dev branch consolidation — 2026-10-03

The integration starts at `be13197ddb7424f2f556bbc17df40ab144d626a1`, the tested and pushed `repair/render-config-fixes` tip. `starting_refs.json` records all 76 other local and origin branch tips after fetching origin. Every origin tip was already an ancestor; nine local histories still required joining.

## Resolution

- `repair/dmcontrol`, `repair/fetch-pickandplace-proof`, `repair/fetch-representative`, and `repair/native-navigation`: every unique non-merge commit is patch-equivalent to an existing integration commit (the recorded `git cherry` lines begin with `-`). Preserve the latest implementation and join their ancestry.
- `step9/benchmarks-pre-attribution-cleanup`: both unique non-merge commits are patch-equivalent. Its merge's author attribution is already present in the current `stable_envs/__about__.py`; retain the current summary and identity.
- `step6/fetch-gpu`: preserve the unique historical profiling scripts, results, and final findings under `historical/`, with original paths and content hashes recorded in `historical_files.json`. Their original locations were deliberately removed by the later v0 archaeology cleanup (`e0b2c5e3`). Do not restore obsolete benchmark entry points over current workflows.
- `step6/integration` and `step7/integration`: preserve their final historical closure dashboards under `historical/`. Do not rewrite regression fixtures with historical approval metadata.
- `step8/integration`: incorporate the acceptance and skybox-decision updates in the two existing Step 8 research documents. Preserve its historical dashboard and exact document originals in the archive. These decisions describe that milestone, not current renderer completeness.

With those contents accounted for, ancestry-only merges join the nine histories without replaying obsolete implementations. All existing branches and worktrees are retained. No semantic acceptance, performance, or renderer status is promoted by this consolidation.

## Verification

`historical_files.json` identifies 21 exact archived files by branch, original path, archive path, SHA-256, and Git blob ID. Compare each archived file with `git show <recorded-tip>:<original-path>` using the branch tips recorded in `starting_refs.json`.

After merging, verify each recorded tip using `git merge-base --is-ancestor <tip> HEAD`. The only tree differences from the base must be this consolidation directory and `docs/research/step8_closure.md` / `docs/research/step8_handoff.md`. Production code, tests, benchmark harnesses, regression fixtures, and accepted evidence remain byte-identical to the tested base; this documentation/history consolidation requires no fresh GPU run.

Historical scripts in this archive are provenance records, not supported current benchmark commands.
