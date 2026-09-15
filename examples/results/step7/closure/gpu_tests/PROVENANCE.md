# Which job and which interpreter wrote each `tests/gpu` results file

| file suffix | interpreter | job | node | what that interpreter can reach |
|---|---|---|---|---|
| `.verify.` | `$SCR/venvs/se6-verify` | `6413321` | gpu2107, RTX 3090 | the broad arm: se-step6 + se-step3 + se-step5 + the Step-6 swm checkout, chained by one `.pth`. dm_control, ogbench, gym_pusht, mujoco_warp and pytest in one process. **This is the only arm with a like-for-like control**: Step 6 ran the same files in the same venv and committed the output under `examples/results/step6/verify/gpu/`, which is what `compare_gpu_suite.py` diffs against |
| `.ogbmanip.` | `$SCR/venvs/se7-ogb` + `$SCR/se7/swm-src` | `6417172` | gpu2101, RTX 3090 | the interpreter W5 declared OGBCube and OGBScene against. Run to answer one question: do the OGB findings survive a change of `stable_worldmodel` source? They do, id for id |
| `.rocket.` | `$SCR/venvs/se7-rocket` | `6417172` | gpu2101 | nvdiffrast + PyFlyt + pybullet. It is the only arm that can construct `PFRocketLanding`, and it can construct little else — no pygame, no ogbench, no dm_control — so its 23 failures and 741 errors are overwhelmingly "this venv cannot build that env". **Read it only for the `PFRocketLanding` rows** |
| `.craftax.` | `$SCR/venvs/se7-craftax` | `6417172` | gpu2101 | jax 0.11.1 + craftax 1.6.1. Same caveat: read it for the Craftax rows |

`per_cell.json` and the per-cell table in `docs/research/step7_coverage.md` are the right way to read
the three narrow arms: they count failures **per cell across every interpreter that could construct
it**, so a venv that cannot build an environment contributes nothing rather than contributing a
failure. A raw total from a narrow arm is not comparable to a total from `se6-verify` and is not
compared to one anywhere in this sweep.

`*.failure_messages.json` carries the first assertion line of each failure, lifted out of the raw
`$SCR/logs/` output by `scratch/step7/closure/extract_failures.py` and scrubbed of absolute paths —
the raw logs cannot enter the repository because `tests/unit/test_repo_contracts.py` reads every
`.txt` under `examples/` and would fail on the `$SCR` interpreter paths they contain. Four sections
did not yield a message the extractor could attribute by exact test id; they are listed as
"mechanism not extracted" rather than guessed.
