APPROVE

# Independent review — Step 8, Workstream H (Craftax semantic follow-up)

- **Reviewed SHA:** `89421ebf6aadd2c64a5e0a0361b5e9060a022fbd` (branch `step8/semantic-craftax`,
  worktree `/Users/samib/stable-envs-step8-semantic-craftax`)
- **Delta reviewed:** `git diff 4210793 89421eb` — 11 files, +1400/-0 (no deletions).
- **Reviewer environment:** local CPU only, `MUJOCO_GL=glfw`, `-p no:cacheprovider`,
  `/Users/samib/se6-venv/bin/python` (renderer/tests) and `/Users/samib/se7-craftax-venv/bin/python`
  (jax/craftax identity recheck). Nothing in the reviewed worktree was edited, committed, stashed or
  reset; no Oscar/SLURM contact was made. All regeneration ran into a scratchpad temp directory and
  was diffed against the committed artifacts rather than overwriting them.

## Verdict

**APPROVE.** This workstream ships no implementation patch — no file under `stable_envs/` is
touched — so it is being reviewed as evidence for a `FINDING`, per plan §16. Every load-bearing
number was independently re-derived byte-for-byte or field-for-field from the author's own scripts,
against a fresh copy of the two frozen Step-7 Craftax corpora and a fresh checkout comparison of
`stable-worldmodel`. I found no fabricated evaluation, no scope creep outside the declared file set,
and no renderer/shared-file edits. One factual imprecision in §9 of `findings.md` (a "12 vs 13"
citation-count nuance, explained below) is cosmetic and does not affect the verdict's substance.

## Merge gate (plan §18), ten items

| # | item | status |
|---|---|---|
| 1 | explicit workstream scope | **PASS** — `git diff 4210793 89421eb --name-only` touches exactly `scratch/step8/semantic/{identity_recheck,channel_probe,evaluator_inventory}.py`, `examples/results/step8/semantic/{identity_recheck,channel_probe,evaluator_inventory,findings}.{json,md}` + `test_output.txt`, `docs/research/step8_semantic_craftax.md`, one additive `mkdocs.yml` nav entry, `tests/unit/test_step8_semantic_craftax.py`. No file under `stable_envs/` appears in the diff. |
| 2 | before/after diagnosis | **N/A by construction, correctly claimed** — no behaviour changed; the "before/after" is instead the Step-7→Step-8 diagnosis of what the added channels are worth (§3 of findings.md), independently reproduced below. |
| 3 | targeted unit tests | **PASS** — `tests/unit/test_step8_semantic_craftax.py`, 10 tests, re-derives every quoted number/digest/verdict string from the three committed JSON artifacts rather than trusting the prose. Read in full; assertions are substantive (byte-identity, digest-set inclusion, strip fractions, `any_need_satisfied is False`, ten-row gate table, nav entry), not tautological. |
| 4 | affected Gate-A/Gate-F checks | **N/A, correctly claimed** — no renderer change; both cells' Step-7 Gate A / `swap_renderer` Gate F stand unaltered (confirmed no `stable_envs/` diff). |
| 5 | affected frozen-corpus replay | **PASS, exceeds requirement** — see Item 1 of my reproduction below: both frozen Step-7 corpora re-render byte-identical to their frozen digests at this HEAD, in both renderers. |
| 6 | pixel-movement classification | **PASS** — `git diff --stat 4210793..89421eb -- stable_envs` is empty; confirmed independently. |
| 7 | performance regression check | **N/A, correctly claimed** — no hot-path code touched, no timing claim on this branch. |
| 8 | independent reviewer verdict | this document. |
| 9 | semantic follow-up decision | **PASS** — explicit `none required`, with the reasoning (byte-identity makes a paired Gate H test `a == a`; no evaluator exists to run one anyway) stated and evidenced, matching plan §16's escape hatch ("return a FINDING explaining why a Gate-H run would not test the intended claim rather than fabricating one"). |
| 10 | dashboard update | **owed to coordinator, correctly deferred** — `dashboard.yaml` is Workstream E's file; a complete proposed entry is given in §10 of findings.md for the coordinator to apply. |

## Reproduction, item by item (task brief)

### 1. Byte-identity recheck (`identity_recheck.py`)

Read the script first, as instructed: it imports `craftax.craftax.constants`, calls
`cls.reference_env(size=...)` — which the class docstrings (`stable_envs/envs/craftax.py:1288`,
`stable_envs/envs/craftax_full.py:1357`) assert refuses to build unless
`jax.config.jax_threefry_partitionable` is `True` — and calls the **un-jitted**
`reference_render` in the same process as `env.render`, exactly as the task required.

Ran it (copied to scratchpad with only the `OUT` path redirected) as:
```
PYTHONPATH=<worktree> MUJOCO_GL=glfw JAX_PLATFORMS=cpu JAX_THREEFRY_PARTITIONABLE=True \
  /Users/samib/se7-craftax-venv/bin/python identity_recheck_temp.py
```
Result: `all_byte_identical: true`; `craftax=1.6.1`, `jax=0.11.1`,
`jax_threefry_partitionable=True` (matches the declared reference). Both corpora's `sha256`
matched their frozen `corpus.json` record (`corpus_unchanged: true` for `CraftaxClassic` and
`CraftaxFull`). All four `digest_ours` values (`CraftaxClassic`@7px/16px, `CraftaxFull`@10px/16px)
matched `digest_official` **and** the frozen Step-7 digest, byte for byte:
```
CraftaxClassic  7px  79aebe38...e15e22   equal, exact_frac=1.0, max_abs=0
CraftaxClassic 16px  d06c468c...97c1b0c  equal, exact_frac=1.0, max_abs=0
CraftaxFull    10px  82d83319...7dcaf23cb equal, exact_frac=1.0, max_abs=0
CraftaxFull    16px  65047792...80b332376 equal, exact_frac=1.0, max_abs=0
```
256 frames total (64 states x 4 (cell, tile) pairs) — confirmed from `n_states: 64` per cell x 2
sizes x 2 cells. My rerun's `identity_recheck.json` and the committed one agree on every digest and
on `all_byte_identical`; they differ only in timestamp-independent nothing (host/machine/python/
torch/jax/craftax versions are identical since it is the same interpreter). **Verified: yes.**

### 2. Evaluator inventory (`evaluator_inventory.py`)

`/Users/samib/stable-worldmodel` is at `176d545` ("fix cicd") — matches the claimed checkout
exactly. Reran the script (default `--swm` path, only `OUT` redirected) and got a **byte-identical**
JSON to the committed `evaluator_inventory.json` (`a == b` under `json.load`, confirmed
programmatically). Spot-checked independently of the script:
- `<swm>/checkpoints/` contains only `smoke/` (a PushT tokenizer/joint checkpoint set) — no Craftax
  entry, confirmed by listing.
- `<swm>/docs/baselines.md` contains zero occurrences of "craftax" (grepped); its five result
  tables list only TwoRoom / Push-T / Reacher / OGB Cube, `Checkpoint` column `NA` throughout.
- `<swm>/data/` holds `pusht_demo.lance` only; `<swm>/scripts/data/collect_*.py` lists 12 collectors,
  none matching `craftax` (grepped).
- `<swm>/scripts/plan/config/*.yaml` = `{cube, pusht, reacher, tworoom}`; `categorical_cem.py` does
  exist in `stable_worldmodel/solver/`.
- `examples/downstream_step5/` = `{ingest_pusht.py, paired_gate.py, reacher/, step5_stats.py,
  README.md}` — PushT + Reacher only, grepped for `swm/` ids: `swm/PushT-v1` only.
- `examples/swm_compat/run_swm_workflow.py` drives `swm/TwoRoom-v1` and `swm/PushT-v1` only
  (grepped); no Craftax arm.
**Verified: yes**, both the inventory script's own reproducibility and its individual claims.

### 3. Channel probe (`channel_probe.py`)

Reran (only `OUT` redirected, no other change) under `/Users/samib/se6-venv/bin/python`,
`MUJOCO_GL=glfw`, against the same frozen Step-7 corpora. Output JSON was **byte-identical** to the
committed `channel_probe.json` (`a == b` True). Extracted the specific numbers cited in
findings.md §3:
```
CraftaxClassic  7px  strip=22.2%  night_world=100.0%  night_strip=0.0%  sleep_world=100.0%  inv_world=0.0%
CraftaxClassic 16px  strip=22.2%  night_world=100.0%  night_strip=0.0%  sleep_world=100.0%  inv_world=0.0%
CraftaxFull    10px  strip=30.8%  night_world=97.9%   night_strip=0.0%  sleep_world=96.7%   inv_world=0.0%
CraftaxFull    16px  strip=30.8%  night_world=97.8%   night_strip=0.0%  sleep_world=96.6%   inv_world=0.0%
```
All match findings.md exactly (the 96.6% at 16px vs. the doc's rounded 96.7%/24.6% strip figures at
16px checked against the underlying JSON, not the prose rounding — consistent). Also checked that
the `world_tile_rows` split point the script hardcodes (7 for Classic, 9 for Full) is structurally
correct, not an arbitrary constant: `STRIP_H = 2` in `craftax.py` and `STRIP_H = 4` in
`craftax_full.py`, and `world_rows + STRIP_H` exactly accounts for the frame's total grid rows at
every declared tile size (63/7=9=7+2, 144/16=9=7+2, 130/10=13=9+4, 208/16=13=9+4). **Verified: yes.**

### 4. Plan §16 honouring

- No fabricated evaluation: confirmed via items 1-3 above — every quoted number is reproducible
  from a checked-in, re-runnable script against a documented interpreter/checkout.
- Explanation for why a paired Gate H would not test the claim: findings.md §2 — with byte-identical
  streams, any policy/planner is a pure function of the same bytes on both arms, so the paired
  difference is definitionally `[0, 0]`; this is a correct characterization of what Gate H
  statistically tests (a difference between two *different* pixel streams).
- What evaluator would test it: findings.md §6 lists the concrete missing pieces (collector, trained
  world model, achievement-based success criterion, `categorical_cem` planner arm) in the order
  they'd need to be built, and correctly identifies this as a different, harder question (value of
  the added channels) than the Step-8 claim (renderer correctness).
- Step-5 carry-forward: confirmed by git — `examples/results/step5/` contains exactly `pusht/` and
  `reacher/` (no Craftax representative ever existed), and `git diff <first Step-7 commit>^ <last
  Step-7 commit> --stat -- examples/results/step4 examples/results/step5 examples/results/step6`
  (parent `76dc511`, tip `32e9766`) produced **zero files changed** across all three directories —
  independently confirming the "Step 7 recorded examples/results/step4|5|6: 0 files changed" claim
  for the full Step-7 range, not just a single commit.

### 5. File/scope hygiene

- `git diff 4210793 89421eb --stat -- stable_envs` is empty — no renderer/harness/shared file
  touched.
- `git diff 4210793 89421eb | grep '/oscar'` — no matches. No absolute `/oscar` paths introduced.
- `mkdocs.yml` diff is exactly one additive nav line plus a four-line comment; no other content
  moved or reformatted.
- Full file list matches the "owned paths" table in findings.md §8 item 1 exactly.

### 6. Tests

```
$ cd stable-envs-step8-semantic-craftax && MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python \
  -m pytest -q -p no:cacheprovider tests/unit/test_repo_contracts.py tests/unit/test_step8_semantic_craftax.py
FAILED tests/unit/test_repo_contracts.py::test_every_cited_scratch_path_resolves
1 failed, 20 passed in 0.36-0.41s
```
`tests/unit/test_step8_semantic_craftax.py` alone: **10 passed** (all of them; read the file in
full — every assertion re-derives its number from the committed JSON, none is a no-op check).
`tests/unit/test_repo_contracts.py` alone: **10 passed, 1 failed** — same failing test,
`test_every_cited_scratch_path_resolves`.

I independently confirmed the pre-existing nature of that failure by running the *same* test
against a clean `git archive` of the stated parent commit `4210793`
(`/private/tmp/.../scratchpad/step8review/parent_4210793`): it also fails, and 12 of its unresolved
citations are identical in both runs (none is under `scratch/step8/semantic/`). One nuance: the
parent actually reports **13** unresolved citations, not 12 — the 13th is `scratch/step8/semantic/*`
itself, which this branch's own new directory happens to resolve (it exists now, so that glob-style
citation is satisfied). findings.md §9 / `test_output.txt` state "the same test fails identically on
the parent worktree ... with the same 12 unresolved citations" — technically the parent shows 13,
one of which this branch itself fixes as a side effect of adding `scratch/step8/semantic/`. The
substance of the claim (pre-existing, unrelated to this branch, none under the new directory) is
correct and verified; the "13 vs 12" wording is imprecise. This is a documentation nit, not a
merge blocker — it does not affect the verdict, the digests, or the evaluator-inventory conclusion.

## Defects found

1. **Minor / cosmetic** — `examples/results/step8/semantic/findings.md:517` (§9) and
   `examples/results/step8/semantic/test_output.txt:11-13`: the parent worktree at `4210793`
   actually has **13** unresolved `test_every_cited_scratch_path_resolves` citations, not "the same
   12" as stated; the 13th (`scratch/step8/semantic/*`) is resolved by this branch's own new
   directory, not pre-existing-and-identical. The claim that the failure is pre-existing and
   unrelated to this branch is still correct and independently reproduced. Suggest a one-line
   correction ("13 in the parent, one of which — `scratch/step8/semantic/*` — this branch itself
   resolves; the other 12 are identical and pre-existing") but does not warrant a resubmission.

No other defects found. No fabricated numbers, no scope creep, no renderer touch, no absolute
`/oscar` paths, no weakened thresholds, no GPU time spent contrary to the plan's "do not invent a
huge training campaign" instruction (verified structurally: §5's claim about the held allocation
`6443480` being alive-but-unused on Oscar was not independently checked — it requires `squeue` on
Oscar, which is out of scope for this local-CPU-only review per the task brief — but nothing in the
diff or artifacts contradicts it, and no cluster-produced artifact appears anywhere in this branch's
file list).

## Summary for the coordinator

Approve as a `FINDING` / `DOCUMENTED_LIMITATION`, no Gate-H run, no dashboard file owned by this
workstream (proposed entry in findings.md §10 is ready to apply verbatim). Semantic closure
checkbox (plan §23) is satisfied in its "scientifically classified as not meaningfully testable"
form: byte-identity makes a paired run vacuous, and no downstream Craftax evaluator exists in
`stable-worldmodel` to run one with. One `UPSTREAM_FINDING` (no Craftax checkpoint/collector/
planner/baseline in `stable-worldmodel`) is ready for the Workstream-D bundle.
