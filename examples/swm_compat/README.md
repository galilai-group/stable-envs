# Running a `stable_worldmodel` workflow on the stable-envs renderer

`stable_envs.interop` exists because a stable-envs env is a **batched renderer with an optional
simulator**, while `stable_worldmodel` drives environments through `swm.World`, which builds
a plain Python list of single `gym.Env` objects. Native `swm.World(env=snv.PushT(...))` is **not**
a public API (Workstream J has not merged it). The frozen adapters are `StableEnvGym`,
`BatchedEnvPool` and `swap_renderer`.

This directory is the evidence that the translation works, not a demonstration that it compiles.

```bash
python examples/swm_compat/run_swm_workflow.py --out examples/swm_compat/results
```

## What is being claimed

Four arms drive the **same** caller code — `swm.World`, `swm.policy.RandomPolicy` from
`stable_worldmodel`'s own `policy.py`, the same loop — and differ only in what backs the environments.

| arm | what changes at the call site | what it buys |
|---|---|---|
| A official | — | the baseline |
| B facade | the env id string: `swm/TwoRoom-v1` → `stable_envs/TwoRoom-v0` | portability, no speed |
| C batched pool | B plus `attach_batched_pool(world)` — one line | one render launch for the whole world |
| D renderer swap | the official env keeps its dynamics; only `render()` is replaced | **a pixel comparison** |

Arm D is the only one whose pixels mean anything. A and B do not visit the same states — different
simulators, different reset draws — so comparing their frames would be comparing two different
episodes. D holds everything but the renderer fixed.

## Results

### The workflow runs unchanged, and sees the same info dict

`swm`'s wrapper stack (`AddPixelsWrapper` → `EverythingToInfoWrapper` → `EnsureInfoKeysWrapper` →
`ResizeGoalWrapper`) puts a specific set of keys into `info`, and a policy or a dataset writer reads
them by name. Measured against a stock `World` on `swm/TwoRoom-v1`, 8 environments:

```
facade_missing  : []
batched_missing : []
extra in ours   : ['goal']
```

Every key the official env produces is produced by both adapters, with the same `(N, 1, ...)` shapes.
`goal` is an addition: `TwoRoomEnv` renders a goal frame and keeps it privately, and a
goal-conditioned workflow wants it, so both adapters publish it.

Getting there required one real correction. The facade first returned a `Dict` observation, which
`EverythingToInfoWrapper` explodes into its own keys — so `observation` was missing from the info dict
while `state` and `proprio` were present. The official envs return a bare array. Measured, fixed, and
the check is now part of the script rather than a claim in a docstring.

### A swm variation config drives our renderer, bit-exactly

The strongest statement the appearance mapping can make. Take a **live** official env, randomise every
leaf the mapping claims to cover, read its variation values, convert them once with
`style_from_swm_env`, and render the same states both ways:

| | interior exact | all pixels exact | max abs |
|---|---|---|---|
| our renderer, swm's config applied | **100%** | **100%** | **0** |
| our renderer, config ignored (control) | 0% | 0% | 231 |

The control arm is the point: an experiment that cannot detect a config being dropped cannot certify
one being applied. Ten leaves were randomised at once — `agent.color`, `agent.radius`,
`background.color`, `door.color`, `door.number`, `door.position`, `door.size`, `wall.axis`,
`wall.color`, `wall.thickness`.

### Arm D: the official `swm/PushT-v1`, our pixels

The official env stepped by pymunk for 40 steps, its states recorded, and the frames compared at
224×224:

| | value |
|---|---|
| interior pixels exact | **100.000%** |
| all pixels exact | 95.87% (mean 1.11/255) |
| `swap_renderer` hook vs a direct render | max abs diff **0** |
| state per step | 20 B |
| frame per step | 150,528 B |

This is also a **coverage** result, not only an adapter one: stable-envs' `PushT` was written against
`gym_pusht/PushT-v0`, and this measures it against `stable_worldmodel`'s own, independently written
PushT at its default variation. They are the same picture.

### Where the batching pays

`env-steps/s`, 10 steps, `swm/TwoRoom-v1` versus `stable_envs/TwoRoom-v0`:

| environments | official (A) | facade (B) | batched pool (C) | C ÷ A |
|---|---|---|---|---|
| 8 | 461 | 401 | **2,992** | 6.5× |
| 32 | 551 | 399 | **8,541** | 15.5× |
| 128 | 533 | 394 | **8,939** | 16.8× |

The shape of the table is the result, not the peak. A and B are both O(N) Python calls into a CPU
renderer, so their totals are flat in N — adding environments adds exactly proportional work. C
issues one render launch whatever N is, so its total climbs until the launch stops being the cost.

The facade being *slower* than the official env is expected and is not a defect: it pays a CUDA
launch per environment per step to produce one 224×224 frame, which is the worst possible use of a
batched renderer. It exists so that an existing config runs by changing one string; the moment
throughput matters, arm C is one more line.

## The appearance mapping

`VARIATION_MAPS` records, per env, which leaves map by identity, which are renamed, which are scaled,
and — the half a user actually needs — which swm leaves we deliberately do **not** carry, with the
reason. `check_mapping(env)` classifies every leaf of both spaces and reports `undocumented`: swm
leaves in neither the mapping nor the exclusions. It is `[]` for all five mapped envs.

Two entries are worth reading before trusting a name match:

* **PushT's scales are a factor of 40 apart.** swm declares `agent.scale`/`block.scale`/`goal.scale`
  as absolute sizes in `Box(20, 60)` with default 40; stable-envs declares them as multipliers on the
  reference geometry with default 1.0, because that is what makes "our defaults are the official
  appearance" an identity a test can assert. Mapping them by name alone is silently wrong.
* **`TwoRoom`'s `wall.border_color` is declared by the official env and read by nothing** — measured
  0/255 across the two extremes. It is in `swm_only` rather than mapped, so a port that copies the
  leaf list does not inherit a dead knob.

## Files

| | |
|---|---|
| `run_swm_workflow.py` | the four arms, the mapping table and the scaling sweep |
| `results/swm_workflow.json` | everything above, as measured |
| `../../stable_envs/interop.py` | public face: `StableEnvGym`, `BatchedEnvPool`, `swap_renderer`, `VARIATION_MAPS` |
| `../../docs/research/swm_coverage.md` | which swm envs we render, and the verdicts on the ones we do not |
