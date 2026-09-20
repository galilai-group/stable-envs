# Coverage limit for the shared `mjwarp.py` changes: three families the sweeps cannot reach

The registry-wide sweeps behind `texture_gaps`, `prepare_model`'s spot semantics and
`bake_unsamplable_textures` cover every family that builds. **RoboCasa, MetaWorld and HumanoidBench
do not build in any interpreter available to this workstream**, so they are absent from every sweep.
This is the bounded record the coordinator asked for, and it corrects two wrong explanations I gave
earlier: it is not "conditional registration" and it is **not** a missing GPU — the same three fail
identically on a GPU node (job 6398530).

## Why each one cannot be reached

| family | reason, from its own import error |
|---|---|
| **RoboCasa** | needs a pre-built kitchen cached under `~/.cache/stable-envs/robocasa`. `robocasa` pins `mujoco == 3.3.1` and `mujoco_warp` needs `>= 3.11`, so the kitchen must be compiled by a **different interpreter** and cached. No cache exists on Oscar. |
| **MetaWorld** | the `metaworld` package is absent, and pins `mujoco == 3.3.0` (installed `--no-deps` elsewhere). The family was measured in a different venv entirely (`se-fetch`). |
| **HumanoidBench** | **deliberately not registered by default**: `mujoco_warp`'s batched render is not bit-reproducible inside a long conformance session, so 2-3 of ~660 tests fail every run by 1-2/255. |

Installed-package check across `se7-ogb`, `se-step5`, `se-step6` and `se-s3b-fetch311`: **none** of
`robocasa`, `metaworld`, `humanoid_bench` is present in any of them, and there is no kitchen cache.

## Check 1 — does any FROZEN config depend on these families? **No.**

If one of the 49 frozen Step-4 render-distinct configs belonged to an unswept family, its digest
would have to be replayed before the shared change could be trusted. Answered **statically**, from
the source that registers the cells, because the modules cannot be imported to be asked
(`check_frozen_family_overlap.py`, `frozen_family_overlap.json`):

* `robocasa.py` registers 11 candidate cell names, `metaworld.py` 5, `humanoidbench.py` 4.
* **Zero** of them appears among the frozen 49.

Corroborated dynamically from the union of four Oscar venvs' registries
(`frozen_config_modules.json`): of the frozen configs that could be placed by `cls.__module__`, the
owning modules are `classic_control` (5), `ogbench` (11), `swm_nav` (4), `craftax`, `pusht`,
`simple_point_maze` — **none from an unswept family**. The 26 that could not be placed are all
`DMC*` and `Fetch*`, owned by `dmcontrol.py` and `fetch.py`; they are unplaced because those modules
need a GL backend and CUDA/mujoco-3.11 respectively on a login node, not because their ownership is
in doubt. **No frozen digest depends on a family the sweeps could not reach.**

## Check 2 — can the spot rule fire for these families? Answerable for one of three.

The rule fires only on a spot light whose cutoff is outside `[0, 90]`, so reachable MJCF assets
settle it without running anything.

* **RoboCasa: provably cannot fire.** The repo pins a kitchen at
  `scratch/robocasa/out/pinned_scene.xml` (376 kB, the scene `envs/robocasa.py` measures). It
  contains exactly **one** light, and it is
  `<light directional="true" castshadow="false" ...>` — **not a spot, and with no `cutoff`
  attribute anywhere in the file**. `neutralise_illegal_spots` skips non-spot lights, so it cannot
  touch this scene. (This covers the pinned kitchen, which is the one this repo certifies; RoboCasa
  ships many kitchens and the others are not reachable here.)
* **MetaWorld: not answerable.** Its XMLs live in the absent `metaworld` package; none is vendored
  and no dumped light record exists in `scratch/metaworld/`.
* **HumanoidBench: not answerable.** Its assets come from `humanoid_bench.__file__/assets` and the
  package is absent.

## What remains, and what it is worth

For **MetaWorld** and **HumanoidBench** the residual is: *if* either scene carries a spot light with
an out-of-range cutoff, `prepare_model` would now neutralise its angular term. That would be a
**deliberate fidelity correction**, identical to the one measured to be right on OGBCube — not a
regression — but it would change those families' frames and their declarations were measured before
it existed. Both are already out-of-band (MetaWorld measured in another venv, HumanoidBench
unregistered), so nothing certified moves today.

**Step-8 carry-over:** run both sweeps, and a byte-identity replay, in the venvs where these three
build — a cached RoboCasa kitchen, the `se-fetch` venv for MetaWorld, and whatever interpreter
HumanoidBench is re-enabled under. Until then the claim is "no registered, frozen or certified
artifact is affected", which is what the two checks above establish — not "these families are
unaffected", which is not shown.
