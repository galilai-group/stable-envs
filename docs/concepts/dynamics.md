# Stateful step vs functional dynamics

Two ways to move, and they are not overloads of one method.

```python
state = env.reset(seed=0)
action = env.action_spec.sample(env.num_envs, env.rng)

out = env.step(action)                 # STATEFUL: writes env.state; returns StepResult
state = out["state"]                   # always
# reward / terminated / truncated exist only where that env defines those semantics
nxt = env.transition(state, action)    # FUNCTIONAL: env.state is not touched
traj = env.unroll(state, actions)      # FUNCTIONAL, open-loop; actions (T, num_envs, …)
```

On a cell that ships dynamics, `transition(state, a)` equals `step(a)` **to the bit** on the same device for the same `state`. `unroll` equals repeated `transition`. The initial state is not included in the trajectory: the caller already holds it, and returning `T+1` states would offset every downstream index against the actions beside it.

`unroll` means **known / open-loop actions**, not a policy rollout. There is no `env.rollout(policy=...)`. Policies, MPC, CEM/MPPI, planning, dataset collection and auto-reset belong to `stable-worldmodel`. See [ownership](worldmodel.md).

## Renderer-only cells

Most environments in this repository have no batched simulator. That is a design statement: rendering and simulation are separable, and a cell earns a simulator only where its dynamics can be ported **and** checked against the reference.

Those cells raise from `step` / `transition` / `action_spec` with the exact reason (`"{Name} has dynamics; {This} does not"`) and keep `reset` / `render` working. `env.has_dynamics` is the check; every `Env` defines `step`, so `hasattr(env, "step")` is always true and is the wrong test.

## Physics is still separate

A renderer can be bit-exact against the official frame. A contact solver cannot: contact is chaotic, so any reimplementation diverges. PushT's **renderer** is interior-exact; PushT's **simulator** tracks pymunk to a median 0.68 px over 100 steps and states that bound. Conflating the two lets an "our env matches" claim rest on the half that does not. The longer argument is [Why physics is separate](physics.md).
