# Step 8 T3 review — shared per-value texel substitution on the baked plate (DMControlSuite `floor.color`)

APPROVE-PENDING-GPU

Reviewed SHA: `64a38e7` (worktree `/Users/samib/stable-envs-step8-capability`, branch
`step8/capability-plate-texels`), delta against `2b4b5c3` (`step8/integration`). Independent
review, fresh reviewer, no subagents. Local CPU only (`MUJOCO_GL=glfw`; `/Users/samib/se6-venv/bin/python`
and `/Users/samib/se8-vm-venv/bin/python`; `-p no:cacheprovider`). Oscar not reachable this session;
this verdict covers CPU-verifiable merge-gate items only and does **not** by itself authorize merge.

## Merge-gate items (plan §18)

1. explicit workstream scope — **PASS**. `git diff --stat 2b4b5c3 64a38e7` touches only
   `stable_envs/envs/_plate_texels.py` (new), `stable_envs/envs/dmcontrol.py` (plate path),
   `stable_envs/envs/fetch.py` (docstring-only, verified line-by-line — no executable change),
   `stable_envs/compat/variation_maps/{dmc_control,fetch}.py`, tests, docs/support-matrix
   artifacts, and `examples/results/step8/capability/*` + `scratch/step8/capability/*`.
   `git diff --stat` on `examples/results/step4`, `examples/results/step7`, `stable_envs/testing`,
   and `stable_envs/envs/ogb_manip.py` is empty — confirmed directly.
2. before/after diagnosis — **PASS**. §1 of findings.md (Fetch skybox probe), §4 (byte-equality
   probe), §3 (declaration delta) are each independently reproducible (see below); numbers match
   what is claimed.
3. targeted unit tests — **PASS**. `tests/unit/test_plate_texels.py` (23 tests, covers
   quantisation/truncation, mask derivation and collision-safety, GL-context-current discipline,
   unconditional-write/leak-across-instances, key collapsing to bytes and to `None` at default) +
   `tests/unit/test_variation_maps_dmc_control.py` re-run and matched exactly: `se8-vm-venv` gives
   170 passed (23+147); `se6-venv` gives 42 passed / 128 skipped (23+19, matching the claimed
   "19 before + new" split for the no-swm venv).
4. affected Gate-A/Gate-F checks (cuda, new `style_dim`) — **PENDING GPU**. Not run this session;
   Oscar unreachable. Author's `scratch/step8/capability/gpu_evidence.sbatch` is written but never
   submitted (no job ID exists). Declared `style_dim` moves (+6, 11→17 etc.) are consistent with 12
   parity-cell `declaration_delta.json` rows, all `default_frame_identical: true` — but a
   declaration move is exactly the kind of change plan §18 item 4 requires a cuda Gate-A re-run for,
   and that has not happened.
5. affected frozen-corpus replay — **PENDING GPU**. Not run; same blocker as item 4. CPU-side
   substitute (§4 below) is a proxy, not a replacement, for the required Step-4/Step-7 corpus replay
   on cuda.
6. pixel-movement classification — **PASS (cpu-verified)**. Re-ran `bake_vs_recompile.py` for
   `DMCCartpoleBalance`, probe 0 (`rgb1=[.9,.1,.1]`, `rgb2=[.1,.1,.9]`): `tex_identical=true`,
   `frames_identical=true`, verdict `IDENTICAL` — matches the committed
   `bake_vs_recompile_probe0.json` row for that cell. Independently regenerated default-frame
   sha256 for two cells (`DMCCartpoleBalance`, `DMCReacherHard`) against a fresh worktree checked
   out at `2b4b5c3`: byte-identical for both. No default pixel moves; a non-default `floor.color`
   moves texture bytes only via `_plate_texels.substitute`, confined to the masked texels.
7. performance regression check (hot-path code changed) — **PENDING GPU**. Not run; same blocker.
   Author correctly flags the real risk as plate-cache thrash under high-cardinality `floor.color`
   batches (`_SCENE_CACHE_MAX=96`), not kernel cost — this needs the paired throughput sweep on
   Oscar, not a CPU proxy.
8. independent reviewer verdict — **this review**. Overall CPU-verifiable verdict:
   `APPROVE-PENDING-GPU` (see caveat below — this alone does not authorize merge per items 4/5/7).
9. semantic follow-up decision — **PASS**. Author's "none warranted" (no default frame moves, so no
   downstream number can move) is correct and consistent with item 6's evidence.
10. dashboard update — **outstanding, coordinator's task**, correctly deferred by the author.

## Scope and defect check (packet item 1)

Clean. No changes under `stable_envs/testing`, no threshold edits, no frozen-corpus changes
(`step4`/`step7` diff --stat empty, confirmed directly), no re-freeze performed, `ogb_manip.py`
byte-for-byte untouched. `fetch.py`'s 29-line diff is exclusively docstring/comment text (verified
by reading the full diff): it replaces a refuted claim ("skybox needs a second render context")
with the measured one, and does not touch any executable Fetch code path.

## The helper (packet item 2)

`stable_envs/envs/_plate_texels.py` (new, 153 lines) is sound:
- `quantise` implements `int(255*v)` truncation (not round), matches MuJoCo's own compiler and
  swm's own `(bg*255).astype(uint8)` skybox write. Verified against `bake_vs_recompile` output
  (`tex_sha256` identical to a genuine recompile).
- `texel_slots` takes masks from the **pristine** texture and asserts non-empty coverage; caller
  (`dmcontrol.py::_grid_masks`) supplies explicit masks rather than equality because
  `markrgb == rgb2` in the suite's compiled texture (`.2 .3 .4` for both) — an equality mask would
  wrongly move the 1,196 edge-mark texels along with the 44,402 rgb2 texels. `_grid_masks` derives
  its masks from a synthetic 3-colour probe checker and then **asserts** the real texture rebuilds
  byte-for-byte from those masks plus the declared defaults (`dmcontrol.py:436-451` area) — this is
  exactly the "rebuild the real texture byte-for-byte" check the ticket asked me to confirm exists,
  and it does.
- `substitute` calls `renderer._gl_context.make_current()` before `mjr_uploadTexture`, unconditionally
  on every plate-cache miss, never per render — matches the one-Renderer-per-(domain,size) policy
  (`_bundle` unchanged; no second GL context created, confirmed by reading `dmcontrol.py` diff —
  only `_grid_slots`/`_apply_floor`/`_plate_key` were added, `_bundle` itself is untouched).
- `texel_key` returns `None` at the registered default (quantised-bytes comparison), so a default
  render keeps the exact plate-cache key it had before this leaf existed. `test_plate_texels.py`
  covers leak-across-renders/instances (`test_substitute_restores_exactly_after_any_sequence`,
  `test_two_colours_colliding_cannot_corrupt_the_masks`) — ran and passed.
- Default-request pristine-texture claim independently confirmed: default-style frame digests for
  `DMCCartpoleBalance` and `DMCReacherHard` at this HEAD match a clean checkout of the integration
  tree at `2b4b5c3` exactly (see item 6 above).

## Evidence re-run (packet item 3)

- `bake_vs_recompile_probe{0,1,2}.json`: spot-checked probe 0 / `DMCCartpoleBalance` by re-running
  both `--mode recompile` and `--mode substitute` fresh and comparing — `VERDICT IDENTICAL`,
  `frames_sha256` matched exactly (`0d8b1ba9...`).
- Default identity vs `2b4b5c3`: re-derived for two cells (`DMCCartpoleBalance`,
  `DMCReacherHard`) via a temporary `git worktree` at `2b4b5c3` — byte-identical sha256 both cells,
  both venvs consistent.
- `declaration_delta.json`: read directly — 12 `DMControlSuite` cells show `style_dim_before`→
  `style_dim_after` = +6 (11→17 etc.) with `default_frame_identical: true` in every row; 5
  trackcom cells and OGB/Fetch rows show no change. Matches the table in findings.md §3 exactly.

## Fetch (packet item 4)

`fetch_skybox_probe.json` was read directly (not re-run — it targets Fetch scenes which need a
separate, currently-blocked `FetchEnv` construction path per the author's own method note; re-running
it would require the `se8-fetch-venv` and is lower priority than the DMC-side re-runs already done).
Its structure and numbers are internally consistent with findings.md §1 and with the `fetch.py`/
`variation_maps/fetch.py` diffs, which I did read in full: `background.color` remains `partial`
in `variation_maps/fetch.py`, with the reason text corrected from "one render context per value"
(refuted) to "the delivered skybox is on the baked plate; the *default* is the blocker because
swm's leaf can only flatten to one colour and this family's skybox is a 16-colour gradient".
No Fetch executable behaviour changed (confirmed above). The escalation-not-implementation call is
correct under plan §19: implementing the leaf would move this family's frozen default frame, which
requires an explicit re-freeze with Sami's approval — the author does not take that step, correctly.

## VariationMap totals (packet item 5)

Regenerated `leaf_classification.json` via
`scratch/step8/variations-dmc-control/build_classification.py` into a scratch directory using the
committed `measure/` fixtures — output is byte-identical to the committed file modulo the
`generated_at_head_sha` stamp. Regenerated `support_matrix.{json,md}` via `scratch/step8/report.py`
— JSON byte-identical modulo the head-sha field; markdown identical modulo the one head-sha line.
`EXPECTED_COUNTS` deltas checked arithmetically: the 7 affected ids move mapped 21→28 / unsupported
34→27 (a wash of +7/-7, one `floor.color` leaf per id), consistent with the claimed
totals 46→53 mapped / 94→87 unsupported / 161 total unchanged. `test_the_seventeen_ids_total_161_leaves`
passed in the `se8-vm-venv` run above.

## Full-suite run (packet item 6)

`tests/unit` in `se6-venv`, deselecting `test_examples.py` and `-k "not test_every_cited_scratch_path_resolves"`:
**1125 passed, 195 skipped, 112 deselected in 78.7s** (well under the 15-minute cap).
`tests/unit/test_repo_contracts.py` run alone: 10 passed, 1 failed
(`test_every_cited_scratch_path_resolves`) — re-ran the same test against a clean `2b4b5c3`
worktree and got the identical failure, confirming it is pre-existing and not introduced by this
branch. No literal `/oscar/...` paths found in any file this ticket touches (only pre-existing test
files that check for their absence, and the new `gpu_evidence.sbatch`, which legitimately needs
them).

## Commands run (abbreviated, full output above)

```
MUJOCO_GL=glfw se8-vm-venv/bin/python -m pytest tests/unit/test_plate_texels.py tests/unit/test_variation_maps_dmc_control.py -q
  -> 170 passed
MUJOCO_GL=glfw se6-venv/bin/python -m pytest tests/unit/test_plate_texels.py tests/unit/test_variation_maps_dmc_control.py -q
  -> 42 passed, 128 skipped
PYTHONPATH=<worktree> se8-vm-venv/bin/python scratch/step8/capability/bake_vs_recompile.py --mode recompile --probe 0 --cells DMCCartpoleBalance --out recompile0.json
PYTHONPATH=<worktree> se8-vm-venv/bin/python scratch/step8/capability/bake_vs_recompile.py --mode substitute --probe 0 --cells DMCCartpoleBalance --out substitute0.json
PYTHONPATH=<worktree> se8-vm-venv/bin/python scratch/step8/capability/bake_vs_recompile.py --compare recompile0.json substitute0.json --out verdict0.json
  -> VERDICT IDENTICAL
default_identity.py (ad hoc) on capability HEAD and a git-worktree checkout of 2b4b5c3
  -> DMCCartpoleBalance / DMCReacherHard sha256 byte-identical
se8-vm-venv/bin/python scratch/step8/variations-dmc-control/build_classification.py <tmp>/leaf_classification.json --measure examples/results/step8/variations-dmc-control/measure
  -> byte-identical to committed file modulo generated_at_head_sha
se6-venv/bin/python scratch/step8/report.py --json-out <tmp>/support_matrix.json --md-out <tmp>/step8_support_matrix.md
  -> JSON byte-identical modulo head sha; MD identical modulo head-sha line
MUJOCO_GL=glfw se6-venv/bin/python -m pytest tests/unit -q --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"
  -> 1125 passed, 195 skipped, 112 deselected, 78.73s
MUJOCO_GL=glfw se6-venv/bin/python -m pytest tests/unit/test_repo_contracts.py -q
  -> 10 passed, 1 failed (test_every_cited_scratch_path_resolves) — confirmed pre-existing on 2b4b5c3
```

## Defects found

None. No file:line defects identified in the reviewed diff. The one thing worth flagging as an
open question rather than a defect: `fetch_skybox_probe.json` was verified by reading (structure and
cross-consistency with the diff text) rather than by re-execution, since re-running it needs the
`se8-fetch-venv` and Fetch's separate MJCF-direct-compile method (documented blocker: numpy 2.5.3 +
mujoco 3.12 assertion in `gymnasium_robotics/utils/mujoco_utils.py:142`) — this does not affect the
merge-gate verdict since Fetch's leaf is explicitly *not* implemented (escalated only), so no Fetch
code path depends on the probe being correct.

## Verdict and rationale

**APPROVE-PENDING-GPU.** Every CPU-verifiable merge-gate item (1, 2, 3, 6, 9, and the scope/defect
checks in the ticket) passes on independent re-derivation, not just review of the author's claims:
I re-ran a bake-vs-recompile probe cell fresh, re-derived default-frame byte-identity for two cells
against a clean `2b4b5c3` checkout, regenerated `leaf_classification.json` and the support matrix
from the committed generators, and re-ran the full CPU test suite plus the isolated
`test_repo_contracts.py` failure to confirm it is pre-existing. Items 4, 5, and 7 (Gate A on cuda
with the new `style_dim`, frozen-corpus replay on the affected 8 parity cells, and paired throughput
under a high-cardinality `floor.color` batch) remain **PENDING GPU** — `gpu_evidence.sbatch` exists
but has never been submitted, and no job ID exists. **This review's APPROVE-PENDING-GPU verdict does
not by itself authorize merge**: per plan §18 only `APPROVE` may merge, and items 4/5/7 must return
from Oscar before that bar is met.
