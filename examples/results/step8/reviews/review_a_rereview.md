# Independent Step-8 A re-review (follow-up verification)

Verdict: **APPROVE**

Reviewed SHA: `0cbb136` on branch `step8/reviewfix-a` (implementation `063f2168ebcc563cd7f433c99dc3d2140d25530c` on
`step8/variations-dmc-control` plus one prose/docstring-only follow-up commit `WIP Step 8 A review
corrections checkpoint`). Baseline `43ac95238f7472137619f3e69e7636e11dd088b1`. Reviewed in worktree
`/private/tmp/stable-envs-step8-reviewfix-a`, local CPU only, no files in the worktree edited,
committed, stashed, or reset. No Oscar/SLURM activity.

## 1. Count verification (task 1)

`examples/results/step8/variations-dmc-control/leaf_classification.json["totals"]` reads exactly:
`{"mapped": 46, "unsupported": 94, "dead_upstream": 21, "not_applicable": 0, "leaves": 161, "ids": 17,
"cells": 18}`. Dedup of the 170 raw rows by `(swm_id, leaf)` (needed because `CartpoleDMControl-v0`
maps to two stable-envs cells, `DMCCartpoleBalance` and `DMCCartpoleSwingup`, giving 18 cells over 17
ids) reproduces the same 161/46/94/21/0 split independently.

Breaking down the 21 `dead_upstream` rows by `reason` text gives exactly 11 "never reads it" rows + 5
"explicit mass … wins" rows + 5 remaining rows (`Acrobot.target.{color,shape}`,
`BallInCup.{rendering.render_target, target.color, target.shape}`) — i.e. 11 never-read + 5
density-overridden + 5 hidden-target = 21, matching the follow-up's claim in
`stable_envs/compat/variation_maps/dmc_control.py` ("Sixteen SWM DM Control leaves never reach the
compiled model" = 11 + 5, and "Five more leaves, in two groups, reach the model and no pixel") and in
`docs/research/step8_variations_dmc_control.md` §3 ("Twenty-one DM Control leaves are dead") and
`examples/results/step8/variations-dmc-control/findings.md` §3 ("21 swm leaves are dead … Sixteen
leave the compiled model unchanged … Five more reach hidden target elements").

`git diff 063f216 0cbb136` shows only these three files touched: `docs/research/step8_variations_dmc_control.md`,
`examples/results/step8/variations-dmc-control/findings.md`, and the module docstring in
`stable_envs/compat/variation_maps/dmc_control.py` — no code logic, tests, or the JSON/generated
tables changed.

`grep -rniE "twelve (dm|dead|swm)|12 dead|12 swm leaves"` across those three files returns nothing
stale; the only surviving "twelve" is findings.md:190, "Five classic-control IDs and twelve DMC
IDs," which correctly counts the 12 DM-Control gym ids, not dead leaves. All remaining `not_applicable`
mentions are legitimate (the count is 0, and the surrounding prose explains why the taxonomy value is
unused by design) — no leaf is still called `not_applicable` for density/mass/friction/gravity; both
former instances (doc lines 186, 192 in the pre-follow-up version) were corrected to `unsupported`
with the "dynamics request … measured 0 px at a fixed state" framing, consistent with the JSON.

**Result: all counts agree across doc, findings, docstring, and JSON. No stale claims found.**

## 2. Reproduction (task 2)

```
$ cd /private/tmp/stable-envs-step8-reviewfix-a && MUJOCO_GL=glfw PYTHONPATH=. \
  /Users/samib/se8-vm-venv/bin/python -m pytest -q tests/unit/test_variation_maps_dmc_control.py -p no:cacheprovider
...
138 passed, 3344 warnings in 24.95s
```

Regenerated the classification artifact into a temp dir using the committed `measure/` directory:

```
$ MUJOCO_GL=glfw PYTHONPATH=. /Users/samib/se8-vm-venv/bin/python \
  scratch/step8/variations-dmc-control/build_classification.py /tmp/step8-rereview-a/leaf_classification.json \
  --measure examples/results/step8/variations-dmc-control/measure
wrote /tmp/step8-rereview-a/leaf_classification.json {'mapped': 46, 'unsupported': 94, 'dead_upstream': 21, 'not_applicable': 0} 161
$ diff -u examples/results/step8/variations-dmc-control/leaf_classification.json /tmp/step8-rereview-a/leaf_classification.json
(no output, exit 0)
```

Regenerated the doc tables and diffed against the committed slice (`docs/research/step8_variations_dmc_control.md`
lines 335-808, the per-id table section):

```
$ MUJOCO_GL=glfw PYTHONPATH=. /Users/samib/se8-vm-venv/bin/python \
  scratch/step8/variations-dmc-control/build_doc.py /tmp/step8-rereview-a/leaf_classification.json \
  /tmp/step8-rereview-a/generated_tables.md
wrote /tmp/step8-rereview-a/generated_tables.md 474 lines
$ diff -u committed_tables_slice.md generated_tables.md
(no output, exit 0)
```

Both artifacts are byte-identical to the committed versions. 138/138 tests reproduce.

## 3. Renderer-evidence carry-forward argument (§11 of findings.md) (task 3)

```
$ git ls-tree 43ac95238f7472137619f3e69e7636e11dd088b1 -- stable_envs/envs stable_envs/render \
  stable_envs/testing examples/results/step4 examples/results/step7
$ git ls-tree 0cbb136 -- <same paths>
```

All five tree object ids match exactly between baseline and head:
`stable_envs/envs`=`b6ba355d...`, `stable_envs/render`=`6abd350f...`, `stable_envs/testing`=`0a6299ac...`,
`examples/results/step4`=`15a0fc2f...`, `examples/results/step7`=`938581...` — identical at both SHAs,
exactly as the §11 table claims.

```
$ rg -n "style_from_variation|variation_from_style|compat\.worldmodel|variation_maps" stable_envs/testing tests/gpu
(no matches)
```

Nothing under `stable_envs/testing` or `tests/gpu` imports the changed mapping helpers or
`compat.worldmodel`. Combined with the identical tree ids, **the carry-forward argument is valid**:
the renderer, cells, testing harness, and frozen-corpus/step4/step7 results are byte-for-byte
unchanged, and no GPU/testing code path depends on the changed files, so existing Gate-A/frozen-replay
evidence for those unchanged paths legitimately carries forward without a fresh GPU run for this
mapping-only patch. The record correctly scopes this as "with its original device/version
applicability and limitations," not a new pass, and correctly says the coordinator must still run the
final full-universe pass.

## 4. Runtime scope vs baseline (task 4)

```
$ git diff --name-status 43ac95238f7472137619f3e69e7636e11dd088b1 0cbb136
```

Only `M stable_envs/compat/worldmodel.py` and `A stable_envs/compat/variation_maps/{__init__.py,dmc_control.py}`
are runtime code changes/additions. Everything else is `mkdocs.yml` (nav), `tests/unit/test_variation_maps_dmc_control.py`,
docs (`docs/research/step8_variations_dmc_control.md`), and `examples/results/step8/...` / `scratch/step8/...`
artifacts. **Confirmed as claimed.**

## 5. Broader unit suite (task 5)

Ran the full requested command from the worktree (no `timeout` binary needed; it completed well
under the 15-minute cap):

```
$ cd /private/tmp/stable-envs-step8-reviewfix-a && MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python \
  -m pytest -q tests/unit -p no:cacheprovider -x --deselect tests/unit/test_examples.py \
  -k "not test_every_cited_scratch_path_resolves"
...
935 passed, 137 skipped, 112 deselected, 25918 warnings in 76.30s (0:01:16)
```

No failures, `-x` never triggered. (Total collected differs from the 1043-pass figure in findings.md
§8 because that run used a different venv/deselect set; this run, with `-x` and the coordinator's
requested deselections, is clean.)

## Ten-part merge gate

| # | Item | Status |
|---|---|---|
| 1 | Explicit scope | PASS — 5 classic-control IDs + 12 DMC IDs, mapping/setup only; `git diff --name-status` confirms no renderer/cell files touched. |
| 2 | Before/after diagnosis | PASS — prose now agrees with the generated taxonomy (46/94/21/0 across 161 leaves); no remaining `not_applicable`/stale-count contradictions found. |
| 3 | Targeted unit tests | PASS — 138/138 independently reproduced. |
| 4 | Affected Gate-A/Gate-F | PASS-with-scope — targeted default/isolation/round-trip/dead-leaf checks pass; §11 correctly scopes full 42-ID Gate F as Workstream D's deliverable, not claimed here. |
| 5 | Affected frozen-corpus replay | PASS — carry-forward argument verified independently (identical tree ids at the five listed paths + no import of changed helpers from testing/gpu code); record explicitly disclaims a fresh replay result. |
| 6 | Pixel-movement classification | PASS — no renderer/default code changed; documented nondefault-style selections and range gaps (floor.color) unchanged and correctly flagged as pending a coordinator decision. |
| 7 | Performance regression | N/A, correctly stated — no hot-path/renderer code changed. |
| 8 | Independent reviewer verdict | This review: **APPROVE**. |
| 9 | Semantic follow-up | PASS — no renderer information changed, no downstream rerun required, correctly stated. |
| 10 | Dashboard update | Still owned by the coordinator, as §11 states; this report supplies the SHA/verdict/evidence needed to record it. |

## Remaining defects

None found. The three prose-count corrections identified by the prior review are resolved, and the
requested explicit merge-gate applicability record now exists at
`examples/results/step8/variations-dmc-control/findings.md` §11 with a verifiable renderer-evidence
argument. The `floor.color` range gap remains an explicit, correctly-flagged open coordinator decision
(shared-capability texture-write question) and does not block Workstream A per the task instructions.

---

**Five-line summary:** Verdict APPROVE. All counts (17 ids, 161 distinct leaves: 46 mapped / 94
unsupported / 21 dead_upstream / 0 not_applicable, with dead = 11 never-read + 5 density-overridden +
5 hidden-target) are consistent across the JSON, doc, findings, and docstring, and no stale
"twelve"/"not_applicable" claims remain. 138/138 targeted tests reproduced and both generated
artifacts (leaf_classification.json, doc tables) regenerated byte-identical. The §11 renderer-carry-forward
argument is independently verified: five listed paths have identical git tree ids at baseline and
head, and nothing in stable_envs/testing or tests/gpu imports the changed mapping helpers. Only
runtime files changed vs baseline are compat/worldmodel.py and the new compat/variation_maps package;
everything else is docs/tests/scratch/results. The broader tests/unit run (935 passed, 137 skipped,
112 deselected) was clean with no failures.
