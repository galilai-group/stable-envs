# Step 8 — the Craftax semantic follow-up

Step 7 gave the two Craftax cells the part of the observation they had been missing: the inventory
strip, the night chain and the sleeping appearance. Step 8 asked the obvious follow-up — does a
downstream agent still behave the same way on our pixels as on the official ones, now that there
are so many more of them?

**The answer is that the question, put that way, is already settled, and the experiment that would
ask it cannot be run.** This page is the evidence for both halves. The full record, including the
merge-gate table and the dashboard entry, is
`examples/results/step8/semantic/findings.md`.

## The direction of the change decides the experiment

The added information is not an extension *beyond* the official surface. It is the remainder *of*
it. Official Craftax has always handed the agent the whole frame — world window, strip and night
together. Before Step 7 our cell drew only the world rows; after Step 7 it draws the whole frame.
"Fuller" means *complete*, not *enlarged*.

## Byte-identity, re-measured on this branch

Step 7 recorded `digest_ours == digest_official`. Because the verdict below rests entirely on that,
it was re-measured rather than cited: both frozen Step-7 corpora, re-rendered at Step-8 HEAD by our
renderer and by craftax 1.6.1 (jax 0.11.1, `jax_threefry_partitionable=True`, un-jitted) in the same
process, on a different host from the Oscar runs that produced the Step-7 certificates.

| cell | swm id | tile px | frame | `digest_ours == digest_official` | exact frac | max abs |
|---|---|---|---|---|---|---|
| `CraftaxClassic` | `swm/CraftaxClassicPixels-v1` | 7 | 63x63x3 | **yes** | 1.0 | 0 |
| `CraftaxClassic` | `swm/CraftaxClassicPixels-v1` | 16 | 144x144x3 | **yes** | 1.0 | 0 |
| `CraftaxFull` | `swm/CraftaxPixels-v1` | 10 | 130x110x3 | **yes** | 1.0 | 0 |
| `CraftaxFull` | `swm/CraftaxPixels-v1` | 16 | 208x176x3 | **yes** | 1.0 | 0 |

256 frames, every lighting band and both sleep states present by construction, and each stack's
digest equal to the one the frozen Step-7 corpus carries. Probe:
`scratch/step8/semantic/identity_recheck.py`; artifact
`examples/results/step8/semantic/identity_recheck.json`.

**This is why a paired Gate H would not test the claim.** Gate H is an instrument for the case where
two renderers produce *different* pixels and the question is whether the difference matters. Here
the two byte streams are the same object. Any policy or world model is a function of the bytes it
is handed, so a paired run would produce episode-for-episode identical trajectories, a paired
difference of exactly zero with zero variance, and an equivalence interval of `[0, 0]`. That is not
a passing experiment; it is `a == a` at the price of a GPU allocation, and it would read as a pass
whether or not the port were correct. Identity is the stronger statement, and it is already made.

## The channels are real — they are just unattended

"No model consumes the new information" could mean the channels are inert or that no model exists.
It is the second. Flipping each channel on the frozen corpora
(`scratch/step8/semantic/channel_probe.py`):

| cell | tile px | strip share of frame | night: world px changed | night: strip px changed | sleeping: world px changed | inventory: strip px changed | inventory: world px changed |
|---|---|---|---|---|---|---|---|
| `CraftaxClassic` | 7 | **22.2 %** | **100.0 %** | 0.0 % | **100.0 %** | 32.2 % | 0.0 % |
| `CraftaxClassic` | 16 | **22.2 %** | **100.0 %** | 0.0 % | **100.0 %** | 33.4 % | 0.0 % |
| `CraftaxFull` | 10 | **30.8 %** | **97.9 %** | 0.0 % | **96.7 %** | 28.0 % | 0.0 % |
| `CraftaxFull` | 16 | **30.8 %** | **97.9 %** | 0.0 % | **96.6 %** | 24.6 % | 0.0 % |

The strip is a fifth to a third of every frame and the night chain moves essentially every world
pixel, so a model trained on the pre-Step-7 surface was not missing a detail. Nothing crops the
strip away either: `stable_worldmodel`'s `World` resizes the whole frame to `image_shape`. And the
three channels are cleanly separable — lighting changes **0.0 %** of strip pixels (the reference
concatenates the strip after the night chain) and the inventory changes **0.0 %** of world pixels —
which is an independent reproduction of the Step-7 finding on a different branch.

## What downstream evaluator exists: none

A Step-5-style Gate H needs four things at once, and for Craftax none of them is present
(`scratch/step8/semantic/evaluator_inventory.py`, against the `stable-worldmodel` checkout
`176d545`):

| requirement | Step 5 had | Craftax has |
|---|---|---|
| a trained checkpoint | `pusht/lewm`, `reacher/prejepa` | **none** — `checkpoints/` holds only a PushT Genie smoke run; `docs/baselines.md` benchmarks TwoRoom, Push-T, Reacher and OGB Cube and never Craftax |
| an episode corpus | `dmc/reacher_random.h5`, 10 000 episodes | **none** — `data/` holds `pusht_demo.lance`; not one of the 12 `scripts/data/collect_*.py` collectors is Craftax |
| a planner arm for the action space | CEM 300x30, continuous | **none** — four env configs exist (`cube`, `pusht`, `reacher`, `tworoom`). A discrete solver (`solver/categorical_cem.py`) does exist, so this gap is configuration; it is configuration for a model that does not |
| a state-space success criterion | goal state `goal_offset` steps ahead | **none, and not well-posed** — Craftax is open-ended, achievement-scored and terminates on death; neither Craftax id registers a goal or a success |

Our side matches: `docs/research/artifacts/step5_downstream/` has a PushT arm and a Reacher arm, and
`examples/swm_compat/run_swm_workflow.py` drives TwoRoom and PushT — which is why Step 7 routed both
Craftax cells' Gate F through `swap_renderer` in the first place.

So a Gate H here would not be a cheap run of an existing evaluator. It would be the training
campaign the Step-8 plan explicitly forbids inventing.

## The verdict, and what would change it

**`FINDING`, bucket `DOCUMENTED_LIMITATION`**, with an `UPSTREAM_FINDING` attached: `stable_worldmodel`
registers four Craftax ids and documents them, but ships no collector, training config, planning
config, checkpoint or baseline row for any of them. Those ids are environment coverage, not
benchmark coverage.

The evaluator that *would* answer a question with content — and note it is a **different** question,
*what is the added information worth*, rather than *do we match the reference* — needs, in order: a
Craftax collector driven by a policy competent enough that the inventory ever changes (a random
policy never crafts, so its strip is a black rectangle and the experiment measures nothing); a world
model trained on that corpus; Craftax's own achievement set as the scorer; a `categorical_cem` arm
and a `craftax.yaml` plan config; and only then the Step-5 design on top, with the strip-blanked and
night-disabled ablations as the degraded control rather than a fully blank frame. The first two
steps alone are a multi-GPU-day campaign. It is deferred work, not a Step-8 gap.

## What carries forward from Step 5

Step 5's parity-tier claim is unaffected, for two independent reasons. Craftax was never a Step-5
representative — `examples/results/step5/` contains exactly `pusht/` and `reacher/`, chosen by Step 2
before any of those numbers existed — and the Step-7 Craftax work changed no file under
`examples/results/step4|5|6`. Within Craftax, the Step-4 obligation is discharged by construction:
the frame grew from 112x144 to 144x144 at 16 px, but the world rows of a Step-4-shaped state are
unchanged and asserted so by `scratch/step7/craftax-classic-full/certify.py`. The measurement above
is the independent confirmation — zeroing the inventory changes 0.0 % of world pixels.
