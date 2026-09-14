# certify31{1,2}_protocol — the record with the downsample-protocol commit merged

Job 6365695 (`se6-texture-fidelity-protocol`, gpu2109, RTX 3090, driver 575.57.08, `--exclude=gpu2108,gpu2113`),
run against the integration copy at `5b327affde993b163915949874447de9bb9ecb22` (stamped as `git_commit`), both
arms on one card. `check_resolution_consistency` is judged on the 8-state protocol draw
(`protocol.states_sha256` pins it) and the frozen-corpus number is carried beside it as `frozen_corpus`.

| config | arm 3.11 (mujoco 3.11.0 / dm_control 1.0.44) | arm 3.12 (mujoco 3.12.0 / dm_control 1.0.45) |
|---|---|---|
| DMCCheetahRun | PASS — visual 4.99/0.882/0.661; downsample 0.533854 (corpus 0.5304) | BLOCKED — visual `blocked_environment` (mujoco 3.11.0 declared, 3.12.0 present; would be fail); downsample pass |
| DMCHopperHop | PASS — 4.37/0.904/0.684; 0.550223 (corpus 0.5526) | BLOCKED (would be fail); downsample pass |
| DMCWalkerWalk | PASS — 7.86/0.714/0.824; 0.628951 (corpus 0.6188) | BLOCKED (would be fail); downsample pass |
| DMCHumanoidStand | PASS — 6.27/0.732/0.867; 0.603653 (corpus 0.6095) | BLOCKED (would be pass); downsample pass |

`gpu_tests/`: `tests/gpu/test_reproducibility.py -k downsample` under `se-vis311` in the same job —
11 passed (the five DMC visual cells, DMCCartpoleSwingup, DMCPointMassEasy, CraftaxClassic, Crafter, MiniHack,
PushT), 26 skipped (single-size renderers and plate representatives, by the test's own rules), 49 errors
(env fixtures for families the 3.11 venv does not install: ogbench 25, gymnasium_robotics 16, minigrid 7,
pygame 3), 2 failed (MountainCar, MountainCarContinuous: the same `ModuleNotFoundError: No module named
'pygame'`, raised inside the test body at `stable_envs/envs/classic_control.py:1261` rather than in the
fixture, so pytest counts it as a failure — a venv gap, not a renderer result).
