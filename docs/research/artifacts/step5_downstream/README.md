# Can a trained agent tell the renderers apart?

Steps 1–4 answered questions about the renderer in its own terms: which environments it covers,
what correctness means, how fast it is, and how close its pixels are to the official ones. This
directory answers the remaining one, which no pixel metric can:

> take a competent visual world model and its planner, change **nothing** except the function that
> produces the pixels, and see whether the success rate moves.

Two arms, deliberately different in every way except that question:

| arm | environment | world model | planner | renderer families compared |
|---|---|---|---|---|
| `pusht/` | `swm/PushT-v1` | LeWM (ViT-tiny, SIGReg) | CEM 300×30 | pygame/pymunk vs our Triton raster |
| `reacher/` | `swm/ReacherDMControl-v0` | DINO-WM / PreJEPA (frozen DINOv2-S) | CEM 300×30 | MuJoCo's GL rasteriser vs our ray caster |

Two tasks, two rendering regimes (2-D shapes against a 3-D lit scene), two world-model families.
The claim is about the renderer, not about one training objective, so a result that only held for
one model family would not be the claim.

## The design

```
ONE fixed checkpoint      the same weights, hashed and compared across arms
ONE fixed planner         same solver, samples, elites, horizon, action block, seed
ONE fixed episode set     same episode ids, start steps, goal offsets, budget
ONE fixed simulator       the official swm env decides every transition

official pixels   vs   stable-envs pixels   vs   blank frames
```

The third arm is not decoration. Success in both environments is computed from simulator state and
never reads a pixel, so *"the score did not move when I changed the renderer"* is evidence only if
the score **can** move. Blank frames keep the shape, the dtype, the transform pipeline and the
planner intact and remove the information; if that does not cost most of the success rate, the
comparison is vacuous and the result is `INCONCLUSIVE` rather than a pass.

## The verdict is not ours to choose

Every threshold comes from `tests/fixtures/regression/fidelity/certification.json`, which Step 2 froze before any of these
numbers existed: the ±5-point equivalence margin, the CI formula, the 300-episode/6-seed design,
the control's −20-point requirement and the power guard. `step5_stats.py` reads that file and
contains no bound of its own — `tests/unit/test_step5.py::test_no_threshold_is_hardcoded` fails if
one appears.

A non-significant McNemar p is **not** the claim: an underpowered experiment produces one for free.
The claim is that the 95% interval on the paired difference lies entirely inside the margin.

## Files

| file | what it is |
|---|---|
| `step5_stats.py` | the frozen Gate H: Wilson, exact McNemar, paired interval, power guard, verdict |
| `ingest_pusht.py` | audits the existing 300-pair PushT run from its RAW per-episode JSONs |
| `paired_gate.py` | run JSONs → episode records + verdict, with the arm-comparability diff |
| `reacher/state_bridge.py` | reads a live swm DM Control env's state in our renderer's layout |
| `reacher/parity_probe.py` | do the two reacher renderers draw the same picture, and is the target drawn |
| `reacher/swap_renderer.py` | the three pixel surfaces, swapped; nothing else |
| `reacher/eval_reacher.py` | swm's own `eval_wm.py`, wrapped to select an arm and keep per-episode outcomes |

## Running it

```bash
# PushT: audit the existing paired evidence (no GPU, no retraining)
python docs/research/artifacts/step5_downstream/ingest_pusht.py --out tests/fixtures/regression/join/step5/pusht

# Reacher: the pre-flight, then one arm
python docs/research/artifacts/step5_downstream/reacher/parity_probe.py --out tests/fixtures/regression/join/step5/reacher
python docs/research/artifacts/step5_downstream/reacher/eval_reacher.py --arm official --seed 42 --n 50 \
    --policy reacher/prejepa/weights_epoch_3.pt --swm /path/to/stable-worldmodel

# the verdict, from the run JSONs
python docs/research/artifacts/step5_downstream/paired_gate.py --runs tests/fixtures/regression/join/step5/reacher/runs \
    --out tests/fixtures/regression/join/step5/reacher
```

The Reacher arms need a GPU, the `stable_worldmodel` training stack and the Reacher corpus; the
findings are in
[`docs/research/step5_downstream_semantics.md`](../../step5_downstream_semantics.md).
