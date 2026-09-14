# Step 6 — integration verification (`step6/integration-verify`)

**Classification: _pending_ — this file is written as each job returns; a row with no number in it
has not been measured and must not be read as a pass.**

| | |
|---|---|
| what this is | certification of `step6/integration` **as a whole**, not of any one workstream. No renderer code was changed by this workstream. |
| baseline SHA | `89fa5ca81341dc4df7186ec325cb691431a2ef17` (`sami/pre-step6-baseline`) |
| integration SHA verified, pass 1 | `e4d015d31f686395113ee5234081ae86883b19d5` — texture-fidelity, reacher-correctness, raycast3d, tilemap, raster2d-through-`b0c94ad` |
| integration SHA verified, pass 2 | `f7b07d7` (merge `62558d5`, gfxdraw), carried on this branch as `736e642` |
| branch | `step6/integration-verify` (never committed to `step6/integration`) |
| Oscar copies | `$SCR/se6/verify` = `e4d015d`, `$SCR/se6/verify2` = `736e642`. Both carry `GIT_COMMIT`; every artifact records the SHA through `STABLE_ENVS_GIT_COMMIT`, because rsync drops `.git`. |

## How the interpreters were chosen, and why it matters

Each frozen Step-4 shard was measured in a different interpreter, and
`examples/results/step4/conformance/<shard>.json` records exactly which packages were present. A
digest reproduced in a *different* dependency set answers a different question, so every replay here
runs in the interpreter its shard was frozen in:

| shard | frozen provenance | interpreter used here | match |
|---|---|---|---|
| `dmc`, `ogb`, `raster` | dm_control 1.0.45, gym_pusht 0.1.6, ogbench 1.2.1, mujoco 3.12.0, mujoco_warp 3.12.0, no SWM, no craftax (jobs 6176998 / 6178140) | `$SCR/venvs/se-step3` | exact, field for field |
| `swm` | craftax present, stable_worldmodel present, **no mujoco at all** (job 6180921) | `$SCR/venvs/se-step4-swm` | exact |
| `fetch` | gymnasium_robotics 1.4.2, mujoco 3.11.0, mujoco_warp 3.11.0 (job 6176999) | `$SCR/venvs/se-s3b-fetch311` | exact |

For the **pytest** suites no existing interpreter can run the whole of `tests/gpu` in one process:
`se-step6` has pytest and no `mujoco_warp` (so every MJWarp cell would have skipped as
`No module named 'warp'` and the suite would have reported green on a family it never constructed),
and `se-step5` has `mujoco_warp` and no pytest. So `$SCR/venvs/se6-verify` was created. **It installs
nothing**: it is a `python -m venv --without-pip` whose single `.pth` chains the existing
site-packages of `se-step6`, then `se-step3`, then `se-step5`, then the `stable_worldmodel` checkout.
No existing venv was modified. `craftax` is still absent, exactly as it was for every previous
workstream's GPU run, so `CraftaxClassic`'s official-reference checks stay `blocked_environment`.

<!--REPLAY-->

<!--CPU-->

<!--TESTS-->
