# Step 8 — handoff checkpoint (2026-09-16, evening)

Written by the Fable coordinator for the next agent (not necessarily a Claude Code agent). Supersedes
the two earlier checkpoints below the line. Git and artifacts are the source of truth; where this
prose conflicts with them, they win.

## Closure recommendation

**CONTINUE STEP 8 FOR** three finite tickets (plan §23 label). Everything else in §23 is satisfied on
`step8/integration` with an independent `APPROVE` per workstream.

| ticket | what is missing | where to start |
|---|---|---|
| T1 Workstream D — SWM integration / Gate F | never started (agent cut off at launch). Needs the generated 42-id Gate-F/API matrix, the upstream findings bundle (`docs/research/step8_upstream_findings.md`), and the Craftax JIT declaration. Brief: plan §12; inputs listed in the dashboard row `D_swm_integration.brief_inputs`. | worktree `/Users/samib/stable-envs-step8-swm-integration`, branch `step8/swm-integration` (fast-forward it to `step8/integration` first) |
| T2 Final full-universe check (plan §21) on the final tip | the per-workstream evidence is complete, but no single sweep of all 40 pixel ids has been run on the *merged* tip. Needed: `scratch/step8/rejudge_gate_a.py` + `scratch/step8/report.py` regenerate (done, byte-identical), plus one Gate-A/replay sweep on cuda over the 49-config universe + the 7 Step-7 cells from the merged tree (the closure2 driver `scratch/step7/closure/gate_a_new_cells.py` and `scratch/step8/certify/` show the protocol). Note review_e.md D3: a fresh Rocket sweep must be judged through the declaration-aware rejudge, not the raw judge. | Oscar, `3090-gcondo` with `--constraint=geforce3090`, one job, ~2 h |
| T3 Shared capability decision (Sami) | `examples/results/step8/capability/per_world_model_arrays.md`. Per-world model arrays already exist in the mjwarp path; the one shared new capability is per-value texel substitution on the plate bake (21 required leaves across OGB/DMC/Fetch; the Fetch skybox claim is inferred and needs one bake-vs-recompile diff). Not implemented. Sami must confirm before anyone implements it; until then those leaves are `unsupported`/`partial` with the reason recorded. | coordinator decision, then one workstream with GPU validation |

## Exact state

| item | value |
|---|---|
| baseline (Step-7 milestone) | `43ac95238f7472137619f3e69e7636e11dd088b1` = `milestone/step7-full-renderer-coverage` = tag `step7-closure` (origin) |
| `step8/integration` tip | the commit carrying this file; parent merge `5d5a461` (C). `git log --oneline 43ac952..step8/integration` |
| merged workstreams (all reviewed APPROVE, reports in `examples/results/step8/reviews/review_*.md`) | A `0cbb136` (3abab32), B `84c40e1` (cab9b1e), E `2af1e05` (7bb1e48), G `fff274f`, H `89421eb` (FINDING), F `f4164d3` (f793fc3), C `260d413` (5d5a461) |
| not merged | D (nothing to merge); Codex branch `codex/step8-takeover-wip` (no commits; bookkeeping already committed) |
| support matrix on the tip | Gate A 40 PASS / 2 UNSUPPORTED_BY_DESIGN; variation map 26 complete / 12 partial (Fetch background.color) / 2 n/a / 2 unknown; CPU tiers 24 MEASURED / 15 UNMEASURED / 3 by-design; GPU 40 MEASURED |
| unit suite on the tip | 1093 passed, 195 skipped, 112 deselected (`tests/unit/test_examples.py` deselected; `test_every_cited_scratch_path_resolves` is the pre-existing baseline failure — its remaining entries are the Step-6 sboughan/carracing/robocasa citations plus `scratch/step8/semantic/*`-style dashboard globs) |
| dashboard | `examples/results/step8/dashboard.yaml` (edit only via yaml load/dump) — workstream rows, Sami's four rulings under `decisions`, buckets, escalations, Codex dispositions |
| commits | author `sami-bg`, no AI trailers (plan §23) |
| pushed | nothing from Step 8 (plan forbids without Sami) |

## Sami's rulings applied (2026-09-16)

1. `check_colors_apply_exactly`: inapplicable for lit RGB leaves under a known law, narrowly, with evidence → implemented as the class declaration `PFRocketLanding.color_exactness_applicability` read by `stable_envs/testing/conformance.py` (`color_exactness_applicability`, `partition_colour_exactness`), the frozen judge path in `scratch/step8/rejudge_gate_a.py`, and `tests/gpu/test_conformance.py`. Frozen Step-2 text unchanged; law-aware successor check = proposal only.
2. Feasible unmapped leaves: aggregated (T3), nothing declared permanently unsupported for want of a capability.
3. CraftaxClassic corpora: both kept (`corpus_index.json: two_frozen_corpora`).
4. Rocket BATCH_SCALES: threshold unchanged; Rocket `GPU_PARITY [BATCH_SCALES_FAIL]` 14.17x, Fetch 8.29x; revised taxonomy = Step-9 proposal.

## Open decisions for Sami (none blocks the merged work)

- T3 above.
- `pyproject.toml` `dmcontrol` extra is a floor (`mujoco>=3.11`), not a pin; certification needs `mujoco==3.11.0` (F). Pinning is a dependency-contract change.
- `scratch/certification.json` `VARIATIONS_COMPLETE.complete_requires` still names `check_colors_apply_exactly` with no applicability escape (E, packet question 3): frozen text deliberately untouched.

## Deferred / cleanup (recorded in dashboard `buckets`)

DEFERRED_OPTIMIZATION: OGB B=1 throughput after the per-world colour upload (B=4096 unchanged); quiet-node Quadruped CPU measurement; CPU backend for the three GPU-only cells. STEP9_CLEANUP: `report.py` ValueError outside ROOT; interpreter-sensitive `check_resolution_consistency`; pure plate bake without a new EGL context. UPSTREAM_FINDING: see dashboard (swm Fetch material unbind, declared defaults, list entries, numpy 2.5.3 + mujoco 3.12 Fetch construction; OGB agent.color negation; OGBMaze states-mode leaves; mujoco_warp 3.13 bundle; gymnasium-robotics ≥3.12 assertion; stable_worldmodel ships no Craftax evaluator).

## Worktrees / branches

`git worktree list` is authoritative. Step-8 worktrees: integration (`/Users/samib/stable-envs-step8-integration`), A/B/C/E/F/G/H (`/Users/samib/stable-envs-step8-*`, plus `/private/tmp/stable-envs-step8-{c-followup,e-followup,reviewfix-a}`), D (`/Users/samib/stable-envs-step8-swm-integration`, at e915466, no work). All clean. None deleted. The original C worktree still carries its archived dirty `ogb_manip.py` (superseded by the merged follow-up; safe to discard, but nothing was discarded).

## Oscar

Held allocation `6443480` (gpu2264) ends ~22:45 local on 2026-09-16; nothing else of ours running. `$SCR/se8/` holds the per-workstream trees and logs (`cfollowup4` = C's final tested tree `d3ff997`; `reference-provenance`; `cert-followup`; venv `$SCR/venvs/se8-mjwarp313` created by F, not adopted). Never touch the `simclr_*` / `lejepa_*` jobs.

---

## Coordinator takeover verification (2026-09-16)

Git verified integration at `bd074d5ff48cdfe36886233289070456d73e5efa`: this is the
handoff commit directly after `e915466`, not a competing implementation checkpoint.
The milestone, tag, and pre-Step-8 baseline all resolve to
`43ac95238f7472137619f3e69e7636e11dd088b1`. No implementation was merged at takeover.
Original integration untracked `STEP_08_AGENT_BRIEFS/` and `STEP_08_HANDOFF.md` are preserved.

C's original dirty `ogb_manip.py` is preserved untouched. Exact binary-capable Git diff:
`examples/results/step8/takeover/c_authorized_in_progress.patch` (22,578 bytes,
SHA256 `b44808ed56e48bb2cfde5000959629eeab95b6437076a6c8cde59b22ace7ab60`).
It is the authorized cube/floor-color patch; independent review found a custom-to-default
cube-color cache leak. Completion takes place in an isolated follow-up worktree.

Fresh reviews are in `examples/results/step8/reviews/` (review_*.md): A requires changes (138 tests pass;
generated JSON/tables match, prose stale), B is BLOCK (committed GPU artifacts stale,
wrapper discovery and partial skybox semantics unresolved), C's committed maps require
changes and its unfinished renderer is BLOCK. E's review/family-specific follow-up is active.
No review short of APPROVE authorizes merge.

Oscar queue inspection confirmed allocation `6443480` running on gpu2264, elapsed 8:04/12:00
at inspection; no job was cancelled. Jobs `6451948` and `6452376` completed. B's newer
remote evidence exists and cites SHA `4b3d09f`; its unit log says 8 passed / 48 skipped,
not a GPU-unit pass. Retained rendering evidence still requires independent assessment.

The historical checkpoint below is retained as context; the takeover findings above supersede
its claims where they conflict. Release decisions remain open; no frozen contract was changed.

Written by the coordinator when all Step-8 agents were terminated by the account's weekly usage limit
(resets 2026-09-20 02:00 America/Indianapolis). Nothing was pushed. No worktree was deleted.
No Step-8 work has been independently approved yet, so **nothing beyond Wave-0 bookkeeping is merged**.

## Exact state

| item | value |
|---|---|
| Step-8 baseline (accepted Step-7 milestone) | `43ac95238f7472137619f3e69e7636e11dd088b1` (= `milestone/step7-full-renderer-coverage` = tag `step7-closure`, both on origin) |
| `step8/integration` HEAD at handoff | `bd074d5ff48cdfe36886233289070456d73e5efa` — `e915466` plus the handoff commit, no implementation merges |
| `sami/pre-step8-baseline` | `43ac952` (local) |
| dashboard | `examples/results/step8/dashboard.yaml` |
| plan | `STEP_08_AGENT_BRIEFS/00_COORDINATOR_PLAN.md` (Sami's; binding) — note: its `check_order_invariance` is spelled `check_order_independence` in the harness |

## Worktrees / branches (all local; `git worktree list`)

| workstream | worktree | branch | HEAD | commits over base | dirty | state |
|---|---|---|---|---|---|---|
| coordinator | `/Users/samib/stable-envs-step8-integration` | `step8/integration` | e915466 (+handoff) | 3 | 0 | bookkeeping only |
| A variations-dmc-control | `/Users/samib/stable-envs-step8-variations-dmc-control` | `step8/variations-dmc-control` | `063f216` | 4 | 0 | **delivered but unreported**: VariationMap taxonomy/affine/rgb255 extension + 5 classic-control maps; 6 DMC suite cells; 4 trackcom cells; Manipulator+Quadruped with tests/artifacts. Agent died while "regenerating artifacts/doc with the corrected taxonomy" (its last words: "138 pass"). **Unreviewed.** Check whether the regenerated artifacts/doc match the code (`generate`-style script if present) before review. |
| B variations-fetch | `/Users/samib/stable-envs-step8-variations-fetch` | `step8/variations-fetch` | `ce1fcb3` | 7 | 0 | **delivered but unreported**: 12 Fetch ids mapped as one family; tests in three tiers; GPU evidence PASS on all four scenes (jobs 6451948 gpu2105, 6452376 gpu2264; 6451876 cancelled). findings at `examples/results/step8/variations-fetch/findings.md` carry its verdict, counts, an escalation candidate and ownership requests — **read them; the coordinator never received the final message.** Unreviewed. |
| C variations-newcells | `/Users/samib/stable-envs-step8-variations-newcells` | `step8/variations-newcells` | `a09b407` | 3 | **1** (`stable_envs/envs/ogb_manip.py` modified, uncommitted) | PATCH_CANDIDATE reported (49 leaves: 14 mapped / 13 unsupported / 4 dead / 18 n/a; no pixel moved). The dirty file is the authorised second commit set (OGB `cube.color` via mover geom_rgba + `floor.color` via the plate cache key) **in progress** — left untouched; inspect `git diff` before deciding to finish or discard. Reviewer (sonnet) was mid-review when cut off; partial notes may exist in the session scratchpad only. |
| E certification-provenance | `/Users/samib/stable-envs-step8-certification-provenance` | `step8/certification-provenance` | `9d286ce` | 11 | 0 | delivered (8 tasks; 49-config re-judge 0 differ; paired Oscar suites 0 new failures). Reviewer (sonnet) cut off mid-review. A follow-up was requested and started but not committed: the Gate-A record must judge `family_specific` checks (the Step-7 closure harness never ran the frozen judge on them; E's report.py trusted that output). E's last words: "the frozen judge already judges family_specific correctly — the Step-7 closure harness simply never ran it; fixing both the record and the join". **Not committed** (tree clean) — redo. |
| D swm-integration | `/Users/samib/stable-envs-step8-swm-integration` | `step8/swm-integration` | e915466 | 0 | 0 | spawned, died before any work |
| G performance | `/Users/samib/stable-envs-step8-performance` | `step8/performance` | e915466 | 0 | 0 | spawned, died before any work |
| H semantic-craftax | `/Users/samib/stable-envs-step8-semantic-craftax` | `step8/semantic-craftax` | e915466 | 0 | 0 | worktree created, never spawned |
| F reference-provenance | — | — | — | — | — | not created (serialised after E) |

Older Step-6/7 worktrees (`/Users/samib/stable-envs-step6-*`, `/Users/samib/stable-envs-step7-*`) still exist; all Step-7 branches are merged into the milestone.

## Merged / unmerged

- Merged into `step8/integration`: nothing but the dashboard/handoff commits.
- Unmerged, awaiting independent review: A (`063f216`), B (`ce1fcb3`), C (`a09b407` + dirty), E (`9d286ce` + the uncommitted family_specific follow-up).
- Merge gate (plan §18): only a reviewer `APPROVE` may merge. Reviews for C and E must be restarted from scratch (fresh agents; prompts are in the coordinator transcript, `s8_review_C.md` / `s8_review_E.md` were the intended report paths in the session scratchpad and may be partial or absent).
- Ownership caveat for merging: A, B and C each add ONE import/registration line to `stable_envs/compat/worldmodel.py` and each may have created `stable_envs/compat/variation_maps/__init__.py`; expect additive conflicts there (coordinator resolves). A extended the `VariationMap` class (taxonomy/affine/rgb255); C's map for OGBMaze `floor.pixel_encoding` was left `unsupported-pending-A` and should switch to mapped on A's extension.

## SLURM / Oscar

- Held allocation `6443480` (`se7-interactive-3090-b`, gpu2264, RTX 3090) RUNNING with ~4.5 h left at checkpoint time; held by tmux session `se7gpu2` on the login node (`tmux attach -t se7gpu2`; ends with the job). `6437813` has ended. Nothing else of ours is running; the `lejepa_cifar10` / `simclr` jobs are Sami's — never touch.
- Step-8 cluster tree: `$SCR/se8/{baseline (pristine 43ac952 git archive), se8-baseline.tar.gz, cert (E's branch copy), variations-fetch, variations-newcells, census8.sh, gputests.sh, paired.sh}`.
- Step-8 jobs: 6451876 (cancelled), 6451948, 6452376 (B's Fetch GPU evidence, COMPLETED). E used the held allocation via `srun --overlap` (census, GPU suites, paired run). C used it for the OGB/Rocket digest identity.
- Rules in force: no CPU-heavy work on login nodes; `3090-gcondo` allowed with `--constraint=geforce3090` (mixed cards); exclude gpu2108,gpu2113,gpu2110,gpu2116.

## Unresolved escalations (Sami)

1. **`check_colors_apply_exactly` on lit renderers** — packet at `examples/results/step8/certification/colour_check_decision.md` (branch E). TinyRenderer caps any colour at factor 0.993567; literal RGB equality is unreachable for any correct implementation and the official frames fail too. Recommendation (c) declared, measurement-backed inapplicability now with (b) law-aware comparison as successor; adopting edits frozen Step-2 text scoped to applicability.
2. **SWM leaves feasible only with a renderer change** (from C): OGBCube/OGBScene `light.intensity`, `camera.angle_delta`, `cube.size` need a per-world MODEL array in the mujoco_warp context; OGBMaze `light.intensity` needs the fitted lighting law re-derived. Plan §4 forbids declaring them unsupported unilaterally. Coordinator recommendation: decide after reading B's and A's escalation candidates (B's findings mention one) — if Fetch/DMC need the same per-world-array capability it becomes one shared justified change.
3. **CraftaxClassic has two frozen corpora** (Step-4 64×127, Step-7 64×149). Coordinator recommendation: keep both (Step-4 stays the world-row oracle via zero-extension; Step-7 certifies the full observation); no retirement/re-freeze.
4. (Step-7 carry-over) BATCH_SCALES edge case (Rocket fps(1)); Workstream G to report, no threshold change.

## Findings recorded so far (dashboard `upstream_findings`, `buckets`)

- Upstream: OGB `agent.color` negation bug in swm `modify_mjcf_model`; OGBMaze states-mode leaves dead; Rocket `pad.color` declared 200 vs white; OGBMaze `pixel_encoding` declared 0 vs 1; `scratch/swm_matrix.json` stale (6/42 rows).
- DOCUMENTED_LIMITATION: nvdiff x86 CPU sparse-stage one-ulp batch dependence (host-dependent; gpu2264 0/19 widths differ; no CPU consumer; Step-7 re-freeze authoritative).
- E: census 88→100, 0 class changes; corpora resolve; CraftaxClassic downsample judged on the light≥0.5 subset = 1.000000; DOWNSAMPLE entries added Rocket 0.956137, Quadruped 0.811877; `check_order_independence` present (Step-7 "absent" was a getattr misspelling).

## Next recommended actions (after the limit resets)

1. Read B's `findings.md` and A's docs/artifacts; verify A's artifacts match its final taxonomy; inspect C's dirty diff (finish the authorised cube.color/floor.color set or `git stash` it with a note).
2. Spawn fresh independent reviewers (different model from the authors) for A, B, C, E per plan §18; merge only on APPROVE, resolving the additive `worldmodel.py` / `variation_maps/__init__.py` conflicts.
3. Redo E's family_specific Gate-A record follow-up (uncommitted, lost) and regenerate the support matrix.
4. Then Wave 2: D (swm integration/Gate F), F (reference provenance: Cheetah/Hopper/Walker/Humanoid 3.11 vs 3.12; mujoco_warp 3.13 bump evaluation; Antmaze nondeterminism), G (performance accounting), H (Craftax semantic).
5. Obtain Sami's rulings on escalations 1–3 before any contract text change merges.
6. Release or renew the held allocation as needed (`squeue -u sboughan -n se7-interactive-3090-b`).
