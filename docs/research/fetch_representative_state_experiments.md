# Representative Fetch: explicit source restoration evidence

This is a bounded source experiment for `swm/FetchPickAndPlace-v3` only. It is
not a candidate implementation or acceptance gate, and it establishes no
VERIFIED promotion. The worktree starts at integration
`c9eeec46424d70b277a06caae659a6f93f50d100`.

The authority is unchanged: Python 3.12.11, NumPy 2.2.6, Gymnasium 1.2.3,
Gymnasium Robotics 1.4.2 and MuJoCo 3.8.1, with pinned SWM
`176d54532b3c0a251a4f509b74f2fdacc143a63d`. No dependency or source patch is used.
Every action goes through the historical SWM registration, sparse reward and
flattened observation. Explicit `options={'variation': []}` isolates simulator
restoration from visual randomization. Both environments construct independently.

## New field evidence: controller cache

The initially proposed qpos/qvel/time/mocap/warmstart/control/force, goal,
elapsed-step and RNG fields do not suffice when restoration calls `mj_forward`.
Fetch's `reset_mocap2body_xpos` reads the welded gripper body's `data.xpos` and
`data.xquat`. After the historical 20-substep `mj_step`, these cached values
precede the final integration update. Recomputing them from end-of-step qpos
changes the next public action's control target.

The experiment preserves the explicit position[3] and quaternion[4] of
`robot0:gripper_link`. It restores qpos/qvel and the other candidate data,
forwards MuJoCo, then restores warmstart and those two cached controller values.
The untouched source simulator is the continuation oracle; it is never restored
before comparison. The destination first experiences a different seed and
11 different public actions. This uses supported, named MuJoCo data arrays,
without opaque state serialization or private ABI access.

Six cases (seeds 0/17/42, float32 and float64 actions) now match exactly for
32 continuation steps after an eight-step source prefix. Comparison includes
all captured data fields, goal, elapsed count, explicit PCG64 dictionary values,
flattened observation, reward/done flags and numeric info. This establishes
sufficiency of the experimental superset for this bounded source continuation;
it does not establish a final minimal public State schema or candidate parity.

## Omissions and limits

Each of 13 omission groups runs independently for all three seeds with eight
continuation actions: 39 cases. Errors compare public outputs and other State
fields, excluding the omitted field itself. The JSON stores each step, field and
absolute difference; equality is exact and no tolerance is introduced.

| Field/group | Observed evidence | Current conclusion |
| --- | --- | --- |
| qpos | Public observation differs for all three seeds | Required physical state |
| qvel | Public observation differs for all three seeds | Required physical state |
| qacc_warmstart[21] | Omitting it changes public proprio for all three seeds; seed 0 first-step maximum is `7.058087432865617e-18` | Required even though divergence can be tiny |
| gripper cached position[3] | Replacing it with freshly forwarded position changes first-step public proprio by `0.0005323878641405599` at seed 0 | Required controller history |
| gripper cached quaternion[4] | Freshly forwarded quaternion changes first-step public proprio by `9.878135664997956e-7` at seed 0 | Required controller history |
| goal[3] | Public goal/flattened observation differs for all three seeds | Required task state |
| elapsed TimeLimit count | Separate step-49 restoration yields source truncation at 50/51, but omitted count leaves both destination steps untruncated | Required lifecycle state |
| PCG64 state | Separate unseeded reset after restoration is exact; omission changes object/goal sampling for every seed | Required RNG state for reset replay |
| mocap pose | Different-history omission leaves all eight next outputs and other fields exact | Not justified as persistent State by these probes; controller overwrites it |
| ctrl[2] | Different-history omission leaves all eight next outputs and other fields exact | Not justified as persistent State by these probes; controller overwrites it |
| applied forces | Both histories have zero force arrays; omission gives exact continuation | No necessity witness; this does not test arbitrary nonzero applied forces |
| time | Omission changes only the omitted clock in these probes | No independent public-output necessity witness yet |
| mocap + ctrl + applied forces together | Exact eight-step continuation for all three seeds | These bounded continuations work without restoring the group |

The warmstart omission retains the destination history's warmstart after
`mj_forward`; otherwise the forward call itself could obscure the intended
omission. The cache omission intentionally uses freshly recomputed values,
testing whether those fields can be reconstructed from qpos. PCG64 omission
retains the destination generator. Seeded resets alone would not prove its
necessity. The unseeded reset experiment concerns simulator RNG with visual
randomization disabled; SWM Variation-space RNG continuation remains unproved.

## Remaining work

No private Fetch adapter or public full State has been implemented. The proposed
schema should now account for the explicit seven controller-cache values, but
must not inherit every experimental superset field without evidence. The
experiments have not tested a batched candidate, candidate-only restoration,
action clipping/scaling parity, candidate seeded reset, reward thresholds,
physical/visual Variation or rendering. The authority profile's existing Fetch
renderer dependency guard also requires separate investigation before the full
CPU rendering gate: it currently requires MuJoCo >=3.11 and MJWarp whereas this
authority has MuJoCo 3.8.1. No renderer or dependency change was attempted here.

No shared abstraction contradiction has been found. These new fields fit the
existing explicit State contract. PushT remains parked; no other Fetch
registration or simulator family is changed. Navigation's independently running
acceptance checkpoint may stop this work before the full representative gate.

Raw script, complete JSON, stderr and checksums are in
[`semantic_recovery_evidence/fetch_state_proof`](semantic_recovery_evidence/fetch_state_proof/).
Reproduce from this worktree:

```sh
PYTHONPATH=/Users/samib/stable-worldmodel \
  /private/tmp/fetch-authority-swm-lock/bin/python \
  docs/research/semantic_recovery_evidence/fetch_state_proof/source_restore_probe.py
```
