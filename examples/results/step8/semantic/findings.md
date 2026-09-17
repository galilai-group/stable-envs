# Step 8 — Workstream H: the Craftax semantic follow-up

**Verdict: `FINDING`. No Gate-H run was made, and making one would not have tested the claim.**

**Bucket: `DOCUMENTED_LIMITATION`** for stable-envs, with one **`UPSTREAM_FINDING`** attached
(§6) for the Workstream-D bundle. Nothing here is `RELEASE_BLOCKING`: the claim Step 8 needs for
Craftax is established by **byte-identity**, which is strictly stronger than the statistical
equivalence a Gate H could return.

| | |
|---|---|
| branch / worktree | `step8/semantic-craftax` · `stable-envs-step8-semantic-craftax` |
| parent | `step8/integration` at `4210793` ("step8 dashboard: E merged") |
| cells in scope | `swm/CraftaxClassicPixels-v1` (`CraftaxClassic`), `swm/CraftaxPixels-v1` (`CraftaxFull`) |
| renderer code changed | **none.** `git diff --stat 4210793..HEAD -- stable_envs` is empty |
| pixels changed | **none** |
| GPU used | **none.** The held allocation `6443480` was verified alive (`3090-gcondo`, `gpu2264`) and deliberately not used — see §5 |
| evidence | `identity_recheck.json` · `channel_probe.json` · `evaluator_inventory.json`, all in this directory |
| probes | `scratch/step8/semantic/{identity_recheck,channel_probe,evaluator_inventory}.py` |
| host | Apple arm64 Mac. No number here is a timing, so no number here needs a cluster |

---

## 1. The claim, stated precisely enough to be falsifiable

The coordinator plan (§16) asks:

> Determine whether the fuller stable-envs observation remains semantically interchangeable with
> the official surface in a downstream setting.

The reason the question was opened is that Step 7 added observation information stable-envs had
previously been missing: the inventory strip, the night chain, and the sleeping appearance. It is
worth being exact about the direction of that change, because the direction is what decides the
experiment.

**The added information is not an extension beyond the official surface. It is the remainder of
it.** The official Craftax observation has always been the whole frame — world window *and* strip
*and* night. Before Step 7 the stable-envs cell drew only the world rows, so it was a *subset*;
after Step 7 it draws the whole frame. "Fuller" here means *complete*, not *enlarged*.

So the claim decomposes:

| sub-claim | how it is settled |
|---|---|
| the Step-7 observation is interchangeable with the official one | **byte-identity**, §2 — measured, not inferred |
| the Step-7 additions carry information a model could use | **§3** — measured; they do |
| a *trained agent* behaves the same on the two surfaces | **vacuous given §2**, and **untestable** given §4 |

## 2. Byte-identity, re-measured at Step-8 HEAD

Step 7 recorded `digest_ours == digest_official` for both cells. That record is the load-bearing
fact for this workstream, so it was re-measured on this branch rather than cited — fresh renders,
current HEAD, a different host from the Oscar runs that produced the Step-7 certificates, both
renderers in the same process.

`scratch/step8/semantic/identity_recheck.py` → `identity_recheck.json`. Reference:
craftax **1.6.1**, jax **0.11.1**, `jax_threefry_partitionable=True`, un-jitted, `JAX_PLATFORMS=cpu`.

| cell | swm id | tile px | frame | frozen corpus intact | `digest_ours == digest_official` | exact frac | max abs |
|---|---|---|---|---|---|---|---|
| `CraftaxClassic` | `swm/CraftaxClassicPixels-v1` | 7 | 63x63x3 | yes | **yes** | 1.0 | 0 |
| `CraftaxClassic` | `swm/CraftaxClassicPixels-v1` | 16 | 144x144x3 | yes | **yes** | 1.0 | 0 |
| `CraftaxFull` | `swm/CraftaxPixels-v1` | 10 | 130x110x3 | yes | **yes** | 1.0 | 0 |
| `CraftaxFull` | `swm/CraftaxPixels-v1` | 16 | 208x176x3 | yes | **yes** | 1.0 | 0 |

256 frames, 64 states per cell, every lighting band and both sleep states present in each corpus
by construction. Digests (sha256 over the whole uint8 frame stack), equal on both sides and equal
to the frozen Step-7 record:

```
CraftaxClassic  7 px  79aebe383560d3d2528a4bb8221ba38db2bb2e95966433347ad3445861e15e22
CraftaxClassic 16 px  d06c468ce0eea2c84ab256f003d693dbbc284c4eab414ff0c7257fa9797c1b0c
CraftaxFull    10 px  82d83319a48bd3c88b23bfa569b4451f559341ce5c5d8a2c4fec7357dcaf23cb
CraftaxFull    16 px  650477921f68f055317c46c94731708e2389e041932f8127713576d80b332376
```

**This is why a paired Gate H would not test the claim.** Gate H is a statistical instrument for
the case where two renderers produce *different* pixels and the question is whether the difference
matters. Here the two byte streams are the same object. Any policy, planner or world model is a
function of the bytes it is handed; hand it the same bytes and it emits the same action, so a
paired run would produce episode-for-episode identical trajectories, a paired difference of
exactly 0 with zero variance, and an equivalence interval of `[0, 0]`. That is not a passing
experiment — it is an expensive re-derivation of `a == a`, and it would *look* like a pass whether
or not the underlying port was correct, because the same renderer would be on both sides of the
comparison in every way that matters. The identity in §2 is the stronger statement and it is
already made.

## 3. Are the added channels information-bearing? Yes — measured

"No model consumes them" could mean two very different things: that the channels are inert, or
that no model exists. It is the second. `scratch/step8/semantic/channel_probe.py` →
`channel_probe.json` flips each channel on the frozen corpora and measures what moves.

| cell | tile px | strip share of frame | night: world px changed | night: strip px changed | sleeping: world px changed | inventory: strip px changed | inventory: world px changed |
|---|---|---|---|---|---|---|---|
| `CraftaxClassic` | 7 | 14/63 = **22.2 %** | **100.0 %** | 0.0 % | **100.0 %** | 32.2 % | 0.0 % |
| `CraftaxClassic` | 16 | 32/144 = **22.2 %** | **100.0 %** | 0.0 % | **100.0 %** | 33.4 % | 0.0 % |
| `CraftaxFull` | 10 | 40/130 = **30.8 %** | **97.9 %** | 0.0 % | **96.7 %** | 28.0 % | 0.0 % |
| `CraftaxFull` | 16 | 64/208 = **30.8 %** | **97.9 %** | 0.0 % | **96.6 %** | 24.6 % | 0.0 % |

Three things fall out of that table, and all three matter to the verdict:

1. **The additions are large.** The strip is between a fifth and a third of every frame, and the
   night chain moves essentially every world pixel (mean `|d|` 52-75 / 255). A model trained on the
   pre-Step-7 surface was not missing a detail; it was missing a quarter of the picture and the
   entire day/night signal. Any image-consuming model would see them — `stable_worldmodel`'s
   `World` resizes the *whole* frame to `image_shape` (its own docs recommend
   `image_resample='nearest'` for Craftax), so nothing crops the strip away.
2. **The strip is not lit, and the lighting is not in the strip.** Night and sleeping change
   **0.0 %** of strip pixels at every size — the reference concatenates the strip after the night
   chain. This independently reproduces the Step-7 finding on a different pool and a different
   branch, and it means the three added channels are *separable*: an evaluator could attribute a
   downstream effect to the strip or to the lighting without confounding them.
3. **The strip is not a proxy for the world rows.** Zeroing every inventory count changes
   **0.0 %** of world pixels. The channels are disjoint.

So the channels are real information, they are in-band for any pixel model, and they are
attributable. What is missing is the model.

## 4. What downstream evaluator exists — audited, not assumed

`scratch/step8/semantic/evaluator_inventory.py` → `evaluator_inventory.json`, against the
`stable-worldmodel` checkout (`176d545`, "fix cicd") and this repository.

A Step-5-style Gate H needs four things at once. For Craftax, **none of the four exists**:

| requirement | what Step 5 had (PushT / Reacher) | what Craftax has | where looked |
|---|---|---|---|
| a trained checkpoint for the environment | `pusht/lewm`, `reacher/prejepa/weights_epoch_3.pt` | **none.** `<swm>/checkpoints/` holds only `smoke/` — a 4-batch, 60-step Genie tokenizer smoke run on PushT (`configs/smoke.json`). `docs/baselines.md` benchmarks DINO-WM, PLDM and LeWM on TwoRoom, Push-T, Reacher and OGB Cube only, every Checkpoint cell `NA`; Craftax appears in no baseline table | `<swm>/checkpoints`, `<swm>/docs/baselines.md` |
| an episode corpus to draw paired starts and goals from | `dmc/reacher_random.h5` (10 000 episodes), the PushT demo set | **none.** `<swm>/data/` holds `pusht_demo.lance` only; of the 12 `scripts/data/collect_*.py` collectors (antmaze, cube, dmc, pointmaze, pusht×3, reacher, scene, tworooms×2, weak-discrete-pusht) **not one is Craftax** | `<swm>/data`, `<swm>/scripts/data` |
| a planner configured for the action space | CEM 300×30 over a continuous action block | **none.** `scripts/plan/config/` has exactly four env configs — `cube`, `pusht`, `reacher`, `tworoom`. Craftax is `Discrete(17)` / `Discrete(43)`; a discrete solver *does* exist (`solver/categorical_cem.py`), so this is the one gap that is configuration rather than absence — but it is configuration for a model that does not exist | `<swm>/scripts/plan/config`, `<swm>/stable_worldmodel/solver` |
| a success criterion computed from simulator state | goal state `goal_offset` steps ahead, distance in state space; never reads a pixel | **none, and not well-posed.** `eval_wm.py` scores "reach the state the corpus was in `goal_offset_steps` later". Craftax is open-ended, achievement-scored and terminates on death; neither Craftax id registers a goal or a success, and "reach the frame 25 steps ahead" is not the benchmark's task | `<swm>/scripts/plan/eval_wm.py` |

The stable-envs side is the same story from the other direction: `examples/downstream_step5/`
carries a PushT arm and a `reacher/` arm and no Craftax arm, and
`examples/swm_compat/run_swm_workflow.py` drives TwoRoom and PushT and has no Craftax arm — which
Step 7 already recorded when it routed both Craftax cells' Gate F through `swap_renderer` instead.
The cluster adds nothing: the only `*.lance` corpora under scratch are four PushT variants, and
there is no Craftax dataset or checkpoint anywhere on it.

**There is no checkpoint to run, no episodes to pair, no planner arm to select and no success to
count.** A Gate H here would not be a cheap run of an existing evaluator; it would be the
"training campaign" §16 explicitly forbids inventing.

## 5. Why the held allocation was not used

`squeue -j 6443480` was checked first and the job is alive (`3090-gcondo`, `gpu2264`, running).
It was deliberately **not** used, and no new job was submitted. Per §2 of this document there is
no experiment whose result is not already determined, and per §4 there is no evaluator to run;
spending GPU time to produce a `[0, 0]` interval would put a number in the dashboard that looks
like a downstream result and is not one. Every measurement on this branch is a CPU render of 64
frozen states and belongs on the local machine. Nothing was cancelled.

## 6. The evaluator that *would* test it

Recorded so the question is answerable later rather than reopened blindly. Note first that it
tests a **different** claim from the one §1 names: with the surfaces byte-identical, the only
downstream question left with any content is *what the added information is worth*, i.e. the
pre-Step-7 partial surface against the complete one. That is a question about Craftax and about
world models, not about whether stable-envs matches the reference.

The instrument would need, in order:

1. **A corpus.** A `scripts/data/collect_craftax.py` on the pattern of `collect_dmc.py`, with an
   action distribution competent enough that the strip ever changes — a random policy never crafts,
   so its inventory is 0 forever and the strip is a black rectangle, which would make the
   experiment measure nothing. That means an expert or a trained PPO policy first (Craftax's own
   baselines are PPO; craftax 1.6.1 ships no weights, only the env).
2. **A trained world model** on that corpus — one of `lewm.py` / `pldm.py` / `prejepa.py`.
3. **A task with a state-space success criterion.** The honest one is Craftax's own achievement
   set, not a goal image; that is a scorer `stable_worldmodel` does not have.
4. **A planner arm** over `Discrete(17)`/`Discrete(43)` via `solver/categorical_cem.py`, plus a
   `scripts/plan/config/craftax.yaml`.
5. **Then** the Step-5 design on top: paired episodes, same seeds, same starts, same policy and
   planner, only the renderer differing, the frozen equivalence CI from
   `scratch/certification.json`, and a **degraded/blank control** — which for this question should
   be the *strip-blanked* and *night-disabled* ablations rather than a fully blank frame, since
   those are exactly the surfaces the Step-7 change removed.

Steps 1-2 alone are a multi-GPU-day campaign. This is `DEFERRED` work and a Step-9-or-later
question, not a Step-8 gap.

**`UPSTREAM_FINDING` for the Workstream-D bundle:** `stable_worldmodel` registers four Craftax ids
(`CraftaxPixels-v1`, `CraftaxSymbolic-v1`, `CraftaxClassicPixels-v1`, `CraftaxClassicSymbolic-v1`)
and documents them in `docs/envs/craftax.md`, but ships no data collector, no training config, no
planning config, no checkpoint and no baseline row for any of them. The ids are environment
coverage, not benchmark coverage. That is upstream's to close, and it is the sole reason this
workstream returns a `FINDING`.

## 7. What Step-5 claims carry forward, and on what evidence

Step 8's closure criterion is that "Step-5 semantic claims still carry forward where representative
pixels are unchanged". For Craftax that holds for two independent reasons:

1. **Craftax was never a Step-5 representative.** `examples/results/step5/` contains exactly
   `pusht/` and `reacher/`. Step 2 pre-selected the representatives before any of these numbers
   existed, and the Step-5 page makes a parity-tier claim on PushT and Reacher and explicitly
   declines a visual-tier claim. No Step-5 claim was ever made about Craftax, so none can be
   invalidated by a Craftax change.
2. **The Step-5 representatives' pixels did not move.** The Step-7 Craftax work touched
   `stable_envs/envs/craftax.py` and the two new `craftax_full` files and nothing else; Step 7
   recorded `examples/results/step4|5|6`: 0 files changed. This branch changes no renderer code at
   all.

And within Craftax itself, the Step-4 obligation is discharged by construction rather than by
argument: the frame grew (112x144 → 144x144 at 16 px) but the **world rows of a Step-4-shaped state
are unchanged and asserted so** in `scratch/step7/craftax-classic-full/certify.py`
(`step4_world_row_replay`: the world-row digest, with the strip rows required to be black on a
zero-extended row). §3 above is the independent confirmation of the same separation: zeroing the
inventory changes 0.0 % of world pixels.

## 8. Merge gate (§18), ten items

This workstream ships **no implementation patch** — no file under `stable_envs/` is touched — so
most of the gate is not applicable by construction rather than by omission. Recorded in full
because §18 asks for all ten.

| # | gate item | status |
|---|---|---|
| 1 | explicit workstream scope | **met** — Workstream H only; owned paths are `scratch/step8/semantic/*`, `examples/results/step8/semantic/*`, `docs/research/step8_semantic_craftax.md`, `tests/unit/test_step8_semantic_craftax.py`, one additive `mkdocs.yml` nav entry |
| 2 | before/after diagnosis | **n/a** — no behaviour changed. The "before" and "after" of the Step-7 change are diagnosed in §3 instead |
| 3 | targeted unit tests | **met** — `tests/unit/test_step8_semantic_craftax.py`, 10 tests, checks the committed record against itself (§9) |
| 4 | affected Gate-A / Gate-F checks | **n/a** — no renderer change. Both cells' Step-7 Gate A and `swap_renderer` Gate F stand unaltered |
| 5 | affected frozen-corpus replay | **met, and stronger than required** — both frozen Step-7 corpora were re-rendered at this HEAD and reproduce their frozen digests exactly (§2). No frozen corpus was written |
| 6 | pixel-movement classification | **met** — **no pixel moves.** `git diff --stat 4210793..HEAD -- stable_envs` is empty |
| 7 | performance regression check | **n/a** — no hot-path code touched; this branch contains no timing claim |
| 8 | independent reviewer verdict | **owed to the coordinator.** This branch is evidence, not a patch |
| 9 | semantic follow-up decision | **this document.** Decision: **none required** for either Craftax cell; a paired Gate H is not meaningful while the surfaces are byte-identical (§2) and not possible while no evaluator exists (§4) |
| 10 | dashboard update | **owed to the coordinator** — `examples/results/step8/dashboard.yaml` is not this workstream's file. Proposed entry in §10 |

## 9. Tests

Run together, `MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider
tests/unit/test_repo_contracts.py tests/unit/test_step8_semantic_craftax.py` on the Mac: **20
passed, 1 failed**. Raw output in `test_output.txt`.

| suite | result |
|---|---|
| `tests/unit/test_step8_semantic_craftax.py` (this workstream, 10 tests) | **10 passed** |
| `tests/unit/test_repo_contracts.py` (11 tests) | 10 passed, **1 pre-existing failure** |

The failure is `test_every_cited_scratch_path_resolves`, and it is not this branch's: the same test
fails identically on the parent worktree at `4210793` with the same 12 unresolved citations, none of
which is under `scratch/step8/semantic/`. Both runs are recorded side by side in `test_output.txt`
so the claim can be checked rather than taken.

`tests/unit/test_step8_semantic_craftax.py` asserts that the committed record is self-consistent:
every digest in `identity_recheck.json` is equal on both sides and equal to the frozen Step-7
corpus record; the four digests quoted in this document appear in the JSON; the verdict line says
`FINDING` and agrees with `evaluator_inventory.json`'s `any_need_satisfied == False`; the strip
fractions and the zero strip-vs-lighting deltas quoted in §3 match `channel_probe.json`; the
merge-gate table has exactly ten rows; and the docs page exists and is in the `mkdocs.yml` nav.

## 10. Proposed dashboard entry

```yaml
semantic:
  craftax:
    verdict: FINDING
    bucket: DOCUMENTED_LIMITATION
    gate_h_run: false
    reason: >-
      Both Craftax cells are byte-identical to the official renderer at every certified tile
      size (re-measured at step8/integration HEAD), so a paired evaluation would compare a byte
      stream with itself; and stable_worldmodel ships no Craftax checkpoint, dataset, planning
      config or success criterion, so no evaluator exists to run.
    evidence: examples/results/step8/semantic/
    upstream_finding: swm registers 4 Craftax ids with no downstream benchmark assets
```

## 11. Open items for the coordinator

1. **Reviewer.** §18.8 — an independent reviewer verdict on this `FINDING` is owed.
2. **Dashboard.** §18.10 — the entry in §10 is proposed, not written; `dashboard.yaml` is E's file.
3. **Upstream bundle.** §6's `UPSTREAM_FINDING` belongs in Workstream D's bundle (plan §12).
4. **Closure checkbox.** Plan §23 "Semantic" — "Craftax follow-up is completed or scientifically
   classified as not meaningfully testable" is satisfied by this document in the second form.
   The other two Semantic boxes are unaffected: no Step-8 pixel-changing renderer fix exists on
   this branch, and §7 records the Step-5 carry-forward.
5. **Not reopened.** No GPU time was spent, no job was submitted, no job was cancelled, and the
   held allocation `6443480` is untouched and still running.
