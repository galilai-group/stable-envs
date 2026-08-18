# Why physics is separate

`StableEnv` has `scene`, `render`, `get_state`, `set_state` and a `variation_space`. It does not have
`step`. That omission is the design, not a gap waiting to be filled: `step` is an *optional* extra an
env may offer, and today 18 of the registry's 55 do — the six top-down MiniGrid/BabyAI ids, the five
classic-control envs, the four `stable_worldmodel` navigation envs, SimplePointMaze, Sokoban and PushT.
The rest raise from `sample` with the reason and the alternative. Those counts move as envs land, and
the script that quotes them ([`examples/rollout_with_policy.py`](https://github.com/galilai-group/stable-envs/blob/main/examples/rollout_with_policy.py))
computes them at run time rather than repeating this sentence.

## The two halves have incompatible fidelity stories

A renderer can be bit-exact against the reference. This repo's PushT is: **100% of interior pixels are
byte-identical** to `gym_pusht` on states drawn from the env's own reset distribution. That is a
verifiable claim with a number attached, and the number is 1.0.

A contact solver cannot make that claim. Contact is chaotic — a solver that differs in iteration order,
warm-starting, or the last bit of a normal impulse diverges from the reference within a few hundred
steps, and no amount of care removes that. The honest bound on a reimplemented rigid-body solver is
statistical agreement over short horizons, not equality.

Conflating them is how "our env matches the original" ends up resting on the half that does not. A
combined port reports one aggregate number, that number is dominated by whichever half is easier to
measure, and the physics divergence hides inside it. Splitting them means the rendering claim can be
exact and the dynamics claim can be absent, which is more useful than one claim that is neither.

## The common case needs no simulator

The workload this repo is built for is re-rendering states you already have:

- a stored dataset of trajectories, kept as states rather than as frames;
- a planner or world model that produced imagined states and wants to look at them;
- an evaluation that resets to recorded states;
- any of the above at a different resolution, or with different appearance, than whatever was recorded.

None of that runs dynamics. The states exist; the question is only how fast they become pixels. Making
`step` part of the contract would force every env to port a simulator before it could serve any of these,
which is a large amount of work for a capability the use case does not need.

## Where dynamics do live

The reference CPU simulator stays the definition of the dynamics. It is used, unchanged, to produce the
states in the first place — the gallery's side-by-side video is driven by a real `gym_pusht` rollout, and
the states it visited are what both renderers are fed.

Where a batched GPU simulator genuinely exists for an env, it can be offered as an optional `step`. It is
still not the definition of anything: the reference is.

PushT has one — a batched port of Chipmunk's sequential-impulse solver, 1.07 M env-steps/s at batch
16,384 against the official env's 2,340. It is offered exactly on the terms above: the renderer's claim
is bit-exactness, the simulator's claim is a *distribution*. Measured over 16 episodes × 100 steps from
identical states under identical actions, the block tracks pymunk to a **median 0.68 px** (mean 20.7,
p90 33.6 — a few episodes diverge on chaotic contact), while the kinematic agent, which no contact acts
on, integrates to a **max of 9.5e-5 px** (mean 4.0e-5). That last number is the control: it separates an
integration bug from a
solver difference, and without it the block numbers cannot be interpreted. `check_step_matches_reference`
in `stable_envs.testing.conformance` produces all of it.

## What this costs

Two things, and both are real.

**Closed-loop-on-GPU is per-env, not a property of the repo.** For the 18 envs that ship a `step` it
works end to end; for the other 37 anything that needs the next state has to call the reference env, at
reference speed.
Rendering is orders of magnitude faster than the CPU renderer at 96px everywhere; stepping is faster only where
someone did the work, and doing that work means accepting a statistical fidelity claim in place of an
exact one.

**A state you cannot produce is a state you cannot render.** The thesis is about the frame being a
function of the state, not about the state being cheap to obtain. For a dataset that already exists, that
distinction does not bite. For generating new data from scratch, it does, and the reference simulator's
throughput is the binding constraint.

## The corollary that makes the split checkable

If the frame really is a function of the compact state, then two *reference* envs driven to the same
state through different histories must render the same pixels. That is a property of the official
environment, provable without our renderer existing, and the conformance suite tests it directly. It
caught two real bugs in how PushT is driven — see [Testing and conformance](../testing.md#state-sufficiency).
