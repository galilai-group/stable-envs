# Step 9 Workstream J — SWM interoperability

## Result

PATCH. The public `stable_envs.interop` module is now the single cross-package facade. It exposes the
frozen SWM ID mapping without loading adapter dependencies, lazily loads the private adapter
implementation on first use, and preserves the Wave-3 compatibility aliases under
`stable_envs.compat`.

Baseline was `dc887b8881a5640441b3e48a6192aebd46d25989`. The required integration merge is
`d3cb4a81cced123d767661bf485f0c5059a0dd73`. The implementation commit is
`dc949fb9eb7145e8e525f509afe72507add8ea89`; the no-rebase integration merge commit is
`0929128d4f9415f7b46fe2a7fa992d16be4b6b8b`.

## Frozen Step-8 ID result

All 42 rows in `examples/results/step8/swm/gate_f_matrix.json` match `stable_envs.interop.SWM_IDS`
exactly, including order-independent class tuples. There are 40 pixel-producing IDs, 2 intentionally
unsupported symbolic IDs, and 4 family IDs that honestly name more than one stable-envs cell rather
than choosing one silently.

1. `swm/AcrobotControl-v1` → `Acrobot`
2. `swm/AcrobotDMControl-v0` → `DMCAcrobotSwingup`
3. `swm/BallInCupDMControl-v0` → `DMCBallInCupCatch`
4. `swm/CartPoleControl-v1` → `CartPole`
5. `swm/CartpoleDMControl-v0` → `DMCCartpoleBalance`, `DMCCartpoleSwingup`
6. `swm/CheetahDMControl-v0` → `DMCCheetahRun`
7. `swm/CraftaxClassicPixels-v1` → `CraftaxClassic`
8. `swm/CraftaxClassicSymbolic-v1` → no renderer (symbolic)
9. `swm/CraftaxPixels-v1` → `CraftaxFull`
10. `swm/CraftaxSymbolic-v1` → no renderer (symbolic)
11. `swm/FetchPickAndPlace-v3` → `FetchPickAndPlace`
12. `swm/FetchPickAndPlaceDense-v3` → `FetchPickAndPlaceDense`
13. `swm/FetchPickAndPlaceDict-v3` → `FetchPickAndPlaceDict`
14. `swm/FetchPush-v3` → `FetchPush`
15. `swm/FetchPushDense-v3` → `FetchPushDense`
16. `swm/FetchPushDict-v3` → `FetchPushDict`
17. `swm/FetchReach-v3` → `FetchReach`
18. `swm/FetchReachDense-v3` → `FetchReachDense`
19. `swm/FetchReachDict-v3` → `FetchReachDict`
20. `swm/FetchSlide-v3` → `FetchSlide`
21. `swm/FetchSlideDense-v3` → `FetchSlideDense`
22. `swm/FetchSlideDict-v3` → `FetchSlideDict`
23. `swm/FingerDMControl-v0` → `DMCFingerTurnHard`
24. `swm/HopperDMControl-v0` → `DMCHopperHop`
25. `swm/HumanoidDMControl-v0` → `DMCHumanoidStand`
26. `swm/ManipulatorDMControl-v0` → `DMCManipulatorBringBall`
27. `swm/MountainCarContinuousControl-v0` → `MountainCarContinuous`
28. `swm/MountainCarControl-v0` → `MountainCar`
29. `swm/OGBCube-v0` → `OGBCube`
30. `swm/OGBMaze-v0` → `PointmazeLargeNavigate`, `PointmazeLargeStitch`
31. `swm/OGBScene-v0` → `OGBScene`
32. `swm/PFRocketLanding-v0` → `PFRocketLanding`
33. `swm/PendulumControl-v1` → `Pendulum`
34. `swm/PendulumDMControl-v0` → `DMCPendulumSwingup`
35. `swm/Piecewise-v0` → `Piecewise`, `PiecewiseZones`
36. `swm/PushT-Discrete-v1` → `PushT`
37. `swm/PushT-v1` → `PushT`
38. `swm/QuadrupedDMControl-v0` → `DMCQuadrupedWalk`
39. `swm/ReacherDMControl-v0` → `DMCReacherHard`
40. `swm/SimplePointMaze-v0` → `SimplePointMaze`
41. `swm/TwoRoom-v1` → `TwoRoom`, `TwoRoomTarget`
42. `swm/WalkerDMControl-v0` → `DMCWalkerWalk`

The integration test additionally checks object identity for every class registered in the active
interpreter: the top-level `snv.<Class>`, registry class, canonical-name resolution, and single-cell
SWM `spec()` result are the same class object.

## Adapter compatibility

- VariationMap: the public and old compatibility paths expose the same `VariationMap` class and
  `VARIATION_MAPS` dictionary objects. A PushT default style survives
  style → SWM variation → style byte-for-byte.
- `swap_renderer`: a live `swm/PushT-v1` keeps its original SWM environment, dynamics, reward, and
  termination path while its pixels are replaced by a CPU stable-envs PushT renderer. The resulting
  frame is `(32, 32, 3)` `uint8`, and an official step still succeeds.
- Strongest supported World path: a registered `stable_envs_step9/PushT-v0` is constructed by a real
  `swm.World`, resets successfully, and yields `(1, 1, 32, 32, 3)` `uint8` pixels.

## Native `World(env=...)` status

The locally installed stable-worldmodel checkout is
`176d54532b3c0a251a4f509b74f2fdacc143a63d`. It does **not** support the planned native call.
Its exact public signature starts with:

`World(env_name: str, num_envs: int, image_shape: tuple[int, int], ...)`

`env_name` is required, there is no `env` parameter, `World.__init__` creates
`partial(_make_env, env_name, ...)`, and `_make_env` always calls `gym.make(env_name, ...)`.
Consequently `swm.World(env=snv.PushT(...), ...)` raises `TypeError` for the missing positional
`env_name`. The integration test inspects this signature and skips with that exact reason. If a later
installed SWM exposes `env`, the same test stops skipping and executes the native constructor and
reset; stable-envs does not monkeypatch or fake upstream support.

## Tests

Local Apple CPU, Python 3.12.11:

- `/Users/samib/se8-vm-venv/bin/python -m pytest tests/integration/test_worldmodel.py -q -rs`:
  **5 passed, 1 skipped** (only the exact native-World upstream limitation).
- `/Users/samib/se8-vm-venv/bin/python -m pytest tests/swm/test_gate_f_matrix.py -q -rs`:
  **5 passed**.
- `/Users/samib/se6-venv/bin/python -m pytest tests/integration/test_worldmodel.py -q -rs`:
  **3 passed, 3 skipped** (`stable_worldmodel` is not installed; ID and VariationMap facade tests run).
- `/Users/samib/se8-vm-venv/bin/python -m pytest tests/unit/test_foundation_imports.py
  tests/unit/test_foundation_registry.py -q -rs`: **71 passed, 1 skipped** (the recorded baseline
  interpreter had DM Control available; this macOS interpreter cannot load EGL).
- `/Users/samib/se8-vm-venv/bin/python -m pytest tests/unit/test_variation_map_partial.py
  tests/unit/test_variation_maps_newcells.py -q -rs`: **81 passed, 8 skipped** (OGB/PyFlyt optional
  dependency/runtime gaps recorded by the tests).

No rollout, MPC, policy, dashboard, documentation, packaging, or other workstream code was moved or
edited. Core dependency-direction tests remain green, and importing only the ID facade loads neither
the private gym adapter nor stable-worldmodel.
