# First bounded DM Control family recovery

This checkpoint recovers the registered defaults of three historical IDs at SWM
commit `176d54532b3c0a251a4f509b74f2fdacc143a63d`. Registration kwargs are empty;
the constructors select sparse Acrobot `Balance`, Pendulum `SwingUp`, and sparse
Cartpole `Balance(swing_up=True)`. The matching executable stable-envs cells are
`DMCAcrobotBalance`, `DMCPendulumSwingup`, and `DMCCartpoleSwingup`.
`DMCCartpoleBalance` remains a renderer because its dense balance task does not
match the historical registered default.

Each recovered cell composes independent upstream dm_control CPU simulators and
keeps its existing private compact renderer row. Runtime production imports no
SWM code. `device="cuda"` places State and frames on CUDA; physics remains
delegated CPU execution.

Full State contains float64 `qpos`, `qvel`, time, and `qacc_warmstart`; DMC
`_step_count` and `_reset_next_step`; float64 cumulative reward; and complete
MT19937 keys, position, Gaussian flag, and cached value. These tasks change no
model arrays during episode initialization, unlike Reacher's target, so model
arrays are not State for this cohort. Gravity is persistent Variation. Exact
different-history replay, including implicit reset, demonstrates that this
schema is sufficient. Snapshots are independent and incomplete compact rows are
rejected by restoration.

Actions are float64 shape `(B, 1)`, retain the historical normalized range
`[-1, 1]`, and pass through dm_control's action-scale wrapper. Each public step
runs two underlying steps and sums rewards. Ordinary DMC `LAST` does not become
termination or truncation in SWM; the next underlying step resets implicitly.
The pinned models all have a 1000-step DMC limit, yielding the tested 502-public-
step boundary sequence under repeat 2.

The CPU gate covers seeds 0/17/42 and batches 1/3: reset, independent snapshots,
full roundtrip after changed history, exact one-step and 32-step State/reward/
done/observation/score, eight-step two-history replay, a reward-2 configuration,
the lifecycle boundary, full-State rendering through the unchanged compact
projection without simulator mutation, incomplete-row refusal, and a 32-step
nondefault gravity route for every domain. All semantic errors are zero on
MuJoCo 3.12.0 and dm_control 1.0.45. A visual-only light override also preserves
the exact float64 nominal gravity and eight subsequent oracle-matched steps;
this guards against float32 Style reconstruction perturbing physics.

The additive independent oracle is in
`tests/fixtures/semantics/dmc_first_family/`: three reference-only batch-3,
32-step corpora with SHA-256, source pin, generator, and dependency provenance.
The generator refuses overwrite. Reproduce local evidence with:

```sh
PYTHONPATH=".:/Users/samib/stable-worldmodel" \
  python scripts/recovery/dmc_first_family_gate.py \
  --device cpu --output /tmp/dmc_first_family_cpu.json
python -m pytest tests/unit/test_dmc_first_family_adapter.py \
  tests/unit/test_dmc_first_family_oracle.py
```

Oscar job 6604555 passed the same zero-error gate with CUDA State/render
placement on RTX 3090 (MuJoCo 3.11.0, dm_control 1.0.44). The three frozen
renderer comparisons were byte-identical to baseline, and the original Reacher
gate passed again. CPU frozen renderer comparisons also matched. See
`first_family_recovery.md` and `semantic_recovery_evidence/` for coordinated evidence. Other DMC registrations,
nondefault tasks, the rest of each historical Variation surface, and reset
`options.state` are outside this bounded checkpoint. No shared contract change
was needed.
