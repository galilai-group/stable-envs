# Bounded OGBMaze historical-default recovery

This proof covers only the empty-kwargs registration `swm/OGBMaze-v0` at
stable-worldmodel commit `176d54532b3c0a251a4f509b74f2fdacc143a63d`.
The signature, rather than its stale docstring, selects OGBench 1.2.1's
`point / maze / large / ob_type=states / 224x224` environment. The recovered
stable-envs cell is `PointMazeLargeNavigate`; `PointMazeLargeStitch`, other
maze layouts and the ant/humanoid, Cube and Scene cells remain unclaimed.

`PointMazeLargeNavigate` now composes B independent upstream CPU simulators.
State and rendering live on the requested device; MuJoCo stepping remains on
CPU. Production imports OGBench directly and does not import stable-worldmodel.
The preserved renderer still consumes its frozen two-float qpos projection.

Full State contains float64 qpos (`root.position`), qvel, MuJoCo time and
`qacc_warmstart`; the complete target geom position and selected task id; the
upstream PCG64 reset generator; and one virtualized MT19937 stream per lane for
OGBench's process-global task/noise draws. The target is State because every
reset changes it, and reward/success depend on it. The global RNG is scoped
around each lane's upstream call and restored afterward, so batching cannot
change another lane or the caller's NumPy stream. OGBench also consumes five
action-space samples while constructing a discarded goal observation. An
unseeded-reset replay from different action-space RNG histories proved those
samples do not affect the retained default state or outputs, so that generator
is deliberately absent from State.

The historical default action is a float32 Box(2) in [-1, 1]. One public step
is one upstream point step: add `0.2 * action`, zero qvel, then five MuJoCo
substeps. Output parity covers observation, reward, terminated, truncated,
success, and the numeric `xy`, previous/current qpos and qvel info fields.
Success is computed after the action; it returns reward 1 and terminated true.
The raw wrapped environment has no TimeLimit, and 1,001 continued steps match
with no truncation. A returned termination does not implicitly reset or prohibit
another step.

The CPU gate passed at zero error for seeds 0/17/42 and batches 1/3: reset,
independent snapshot roundtrip, one and 32 steps, different histories resumed
from one snapshot, task/target restoration, unseeded reset continuation,
positive success, the no-horizon boundary, and full-State rendering through the
private projection without simulator or snapshot mutation. Raw and partially
decoded render rows are refused by `set_state`. A persistent nondefault
`agent.color` also passed the full-State render/nonmutation checks. This does
not certify the complete historical appearance mapping or nondefault reset
options/tasks.

The local source profile was
`PYTHONPATH=/private/tmp/stable-envs-ogbench-recovery:/Users/samib/stable-worldmodel`
with `/Users/samib/se7-ogb-venv/bin/python`: Python dependencies included
OGBench 1.2.1, MuJoCo 3.12.0, Gymnasium 1.3.0, NumPy 2.5.3, Torch 2.14.0 and
TensorDict 0.14.2. The independent reference-only artifact is
`tests/fixtures/semantics/ogbench_maze/swm_176d545_ogbench121_mujoco312.npz`;
its manifest pins its SHA, versions, source commit and generator. Reproduce the
gate with:

```sh
PYTHONPATH=.:/Users/samib/stable-worldmodel \
  /Users/samib/se7-ogb-venv/bin/python scripts/recovery/ogbench_maze_gate.py \
  --device cpu --output /tmp/ogbench-maze-cpu.json
```

Oscar job 6604606 passed the same zero-error CUDA State/render gate on RTX 3090
with OGBench 1.2.1 and MuJoCo 3.12.0. Four frozen renderer rows were byte-identical
to baseline on both CPU and CUDA. See `first_family_recovery.md` and
`semantic_recovery_evidence/` for coordinated evidence. No shared contract
change was required.
