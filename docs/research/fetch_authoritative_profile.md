# Fetch authoritative reference profile

The constructor blocker is resolved for the representative historical registration
`swm/FetchPickAndPlace-v3`, which pinned SWM maps to upstream
`FetchPickAndPlace-v4` / `MujocoFetchPickAndPlaceEnv` with sparse reward.

The isolated environment is `/private/tmp/fetch-authority-swm-lock`:

| Component | Exact version |
| --- | --- |
| Python | 3.12.11 |
| NumPy | 2.2.6 |
| Gymnasium | 1.2.3 |
| Gymnasium Robotics | 1.4.2 |
| MuJoCo | 3.8.1 |
| Torch / torchvision | 2.10.0 / 0.25.0 |
| TensorDict | 0.14.2 |

The four simulator package versions and Torch/torchvision come from the
`uv.lock` in SWM commit `176d54532b3c0a251a4f509b74f2fdacc143a63d`.
The interpreter is the existing local Python. Installed Robotics metadata requires
Python >=3.10, MuJoCo >=2.2.0, NumPy >=1.21.0 and Gymnasium >=1.2.0; these pins
satisfy those constraints. SWM's environment extra does not narrow these versions.
Its eager `cv2` import also requires OpenCV; this profile installs
`opencv-python-headless==4.11.0.86`. The full freeze and dependency check are saved.
No packages in the previous failing profile were changed.

Direct upstream `MujocoFetchPickAndPlaceEnv(reward_type='sparse')` and unmodified
pinned SWM `gym.make('swm/FetchPickAndPlace-v3')` both construct and reset with
seeds 0, 17 and 42. The slide-joint NumPy-scalar / MuJoCo-enum comparisons now
succeed in both directions and in tuple membership. The original Robotics
`set_joint_qpos` implementation is unchanged, verified against the installed
wheel RECORD. No monkeypatch or installed-source edit was used. This establishes
a working profile; it does not isolate which single version change resolves the
previous profile's enum behavior.

Reference-only runtime probes now establish:

- Four-dimensional action space, 20 substeps, dt=0.04; float32 flattened
  observation[28] and float64 proprio[25].
- A float64 action `[2,-2,0.3,0.75]` and its explicitly clipped counterpart give
  exactly equal observation, reward/done and all inspected MuJoCo data arrays.
  The full action controller and float32/float64 parity remain ungated.
- Sparse reward at distances immediately below/equal/above 0.05 is 0/0/-1;
  success is 1/0/0. This is a reference boundary probe, not candidate parity.
- Steps 49/50/51 have truncated false/true/true and terminated false throughout;
  simulator time continues after truncation.
- The upstream RNG is PCG64. The model has nq=22, nv=21, nu=2, na=0, nmocap=1,
  no userdata and no plugin state.

The State candidates remain qpos[22], qvel[21], time, mocap pose, task goal,
elapsed TimeLimit, warmstart[21], ctrl, applied forces and RNG. There is no
activation array in this model. No omission experiment has yet justified which
candidates belong in the final schema. Different-history restoration, candidate
reset/action/reward/done/TimeLimit/RNG parity, meaningful physical and visual-only
Variation, full-State rendering, lossy restoration refusal and frozen CPU
renderer preservation remain unproved. No Fetch production adapter or promotion
is included here. Checkpoint A (Humanoid Oscar completion) occurred during this
profile work, before the representative Fetch CPU contract was implemented.

Raw smoke results, probe scripts, logs, installed package freeze, source/lock
hashes and checksums are in
[`semantic_recovery_evidence/fetch_profile`](semantic_recovery_evidence/fetch_profile/).
