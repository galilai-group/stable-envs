# Remaining classic-control semantic recovery

This checkpoint covers only the three historical empty-kwargs registrations below. It extends the
same family-local `DELEGATED_CPU` mechanism already used by CartPole and Pendulum. Native renderers
and their compact render rows remain in place. The old Torch equations remain as numerical
provenance, but public `step` uses independent Gymnasium CPU simulators and public
`transition`/`unroll` refuse rather than expose incomplete reward, lifecycle, or RNG semantics.

The immutable implementation, gate, tests, and frozen oracle commit is
`b3b6da79555020a3fe9ae046c3f6b4a2bef8a153`, based on reviewed integration
`152cb2402bd44063f414865faff6d28e4ccb67c2`. The reference checkout was asserted at
`176d54532b3c0a251a4f509b74f2fdacc143a63d` with no tracked diff. This report does not promote any
registration to `VERIFIED`; cross-platform/GPU-source acceptance remains coordinator-owned.

## Source audit

| Historical registration | Pinned wrapper | Gymnasium environment | TimeLimit | Result |
|---|---|---|---:|---|
| `swm/AcrobotControl-v1` | `gymnasium_control.acrobot.AcrobotWrapper` | `Acrobot-v1` | 500 | Exact CPU gate passed |
| `swm/MountainCarControl-v0` | `gymnasium_control.mountain_car.MountainCarWrapper` | `MountainCar-v0` | 200 | Exact CPU gate passed |
| `swm/MountainCarContinuousControl-v0` | `gymnasium_control.mountain_car.MountainCarContinuousWrapper` | `MountainCarContinuous-v0` | 999 | Exact CPU gate passed |

All three SWM wrappers have `DEFAULT_VARIATIONS = ()`, but their `_apply_physical_variations`
methods ignore the active-key list. Every reset therefore writes every physical value, including the
float32 defaults, into Gymnasium. The candidate reproduces those writes exactly:

- Acrobot: `link_length_1`, `link_length_2`, `link_mass_1`, `link_mass_2`, `link_com_pos_1`,
  `link_com_pos_2`, and `link_moi`. SWM maps these to Gymnasium's uppercase class attributes.
- MountainCar: `gravity`, `force`, and `max_speed`.
- MountainCarContinuous: `power` and `max_speed`. Gravity remains the Gymnasium step's hard-coded
  `0.0025` and is not a SWM Variation leaf.

The exact mappings preserve the canonical task identities. No duplicate registrations were added.

## Full State and restoration

Each State is a one-dimensional batched `TensorDict`. All physical values use lossless float64
projection, `lifecycle.elapsed_steps` is int64, and the PCG64 generator is represented by four int64
base-2^32 limbs for each 128-bit state/increment plus int64 `has_uint32` and `uinteger` leaves.

| Domain | Physical/task leaves beyond lifecycle and RNG |
|---|---|
| Acrobot | `theta1`, `theta2`, `dtheta1`, `dtheta2`: `(B,1)` float64 |
| MountainCar | `position`, `velocity`: `(B,1)` float64 |
| MountainCarContinuous | `position`, `velocity`: `(B,1)` float64; `task.state_is_float32`: `(B,1)` bool |

The continuous dtype leaf is required continuation state. Gymnasium reset installs a float64 ndarray;
each step installs a float32 ndarray, and the stored dtype changes the next NumPy update. At seed 17
with float64 action `0.1234567890123`, forcing the reset snapshot to restore as float32 produced an
observation difference of `5.820766091346741e-11` against one uninterrupted reference step. Restoring
with the correct flag matched that same uninterrupted result exactly. The ordinary gate also proves
reset-to-first-step and steady-state continuation across all required seeds and batch sizes.

Acrobot resets with a float32 state and its first step promotes through `np.append(state, torque)` to
float64 before RK4. Restoring its lossless values as float64 therefore preserves the next update; the
candidate-only restore and two-history tests found zero difference. Discrete MountainCar resets with
a float64 ndarray and later stores a tuple of NumPy float64 scalars; the container type has no effect
on continuation, and the normalized float64 leaves replay exactly.

`get_state` returns independent storage. The gate mutates the actual captured physical and RNG leaves
and proves the live environment is unchanged. `set_state` rejects compact render tensors and partial
States. Rendering a full State equals rendering its private compact projection and does not mutate
the simulator.

## Step, reward, action, and lifecycle semantics

The step result contains full `state`, float64 `reward`, bool `terminated`, bool `truncated`, the
Gymnasium `observation` at its original dtype, and SWM's numeric float32 `info_state`. The SWM string
`env_name` is intentionally absent from the numeric TensorDict contract.

- Acrobot accepts strict integer Discrete(3) actions. It returns `-1` until the height terminal and
  `0` on terminal steps. A terminal snapshot and another post-terminal step matched the reference.
- MountainCar accepts strict integer Discrete(3) actions and returns `-1`, including on the terminal
  step. Terminal and post-terminal continuation matched.
- MountainCarContinuous accepts indexable one-element actions with float16/float32/float64 values.
  Physics clips the force to `[-1, 1]`, while reward uses the original action squared. Action `2.0`
  therefore uses clipped force and returns `-0.4`; the exact behavior matched. The in-bounds float64
  precision probe used `0.1234567890123`.

Gymnasium `TimeLimit` truncates exactly at steps 500, 200, and 999 respectively, does not latch or
auto-reset, and continues advancing after truncation. Seeded resets at 0, 17, and 42, unseeded reset
after restored RNG, and custom reset bounds all matched exact State distributions.

## Evidence and checks

The independent reference helper imports only the pinned SWM wrappers. The frozen corpus was created
from those wrappers at seed 17, batch 3, for 32 steps. Its manifest records package versions, array
schemas, registration IDs, and SHA-256 hashes; integration tests verify each file hash before reading
arrays.

The CPU gate covered both batch sizes 1 and 3 for every seed/domain combination: 18 independent
32-step cells, candidate-only advance/restore against an uninterrupted reference, eight-step
different-history replay, boundary replay, all State and numeric output leaves, reset options,
unseeded RNG, action validation, terminal rewards, post-terminal behavior, TimeLimit boundaries,
render projection/no-mutation, compact rejection, and all 12 SWM physical Variation leaves. Every
comparison used `atol=0`, `rtol=0`; no tolerance or opaque snapshot was introduced.

Commands run with
`/private/tmp/stable-envs-semantic-recovery/.venv/bin/python` and
`PYTHONPATH=/private/tmp/stable-envs-native-recovery:/Users/samib/stable-worldmodel`:

```text
python scripts/recovery/native_classic_remaining_gate.py --device cpu --output /tmp/native_classic_remaining_gate-final.json
python -m pytest -q tests/unit/test_native_classic_remaining_semantics.py tests/integration/test_native_classic_remaining_gate.py tests/unit/test_native_classic_semantics.py tests/integration/test_native_classic_gate.py
python -m ruff check <changed Python files>
python -m ruff format --check <changed Python files>
git diff --check
```

The combined new-three and prior-two focused suite passed 19 tests. The standalone remaining-three
gate passed with zero numerical error. Coordinator-side legacy classic/navigation and shared
full-State checks passed 119 tests against this implementation before the immutable checkpoint.

Local Apple M5 validation cannot exercise CUDA. The classes deliberately report `DELEGATED_CPU`, so
CUDA input tensors still delegate task execution to CPU Gymnasium and return results on the requested
device; Oscar validation and final acceptance remain pending. No PushT, navigation, shared Env,
registry, spec, space, protocol, parity-harness, or recovery-matrix code changed in this checkpoint.
