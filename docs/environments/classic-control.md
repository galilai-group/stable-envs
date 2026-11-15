# Classic control

The classic control tasks of [Gymnasium](https://gymnasium.farama.org/environments/classic_control/) ([Towers et al., 2024](https://arxiv.org/abs/2407.17032)) balance a pole on a cart, swing up a pendulum or a chain of links, and drive a car up a hill. stable-envs steps the batch in NumPy on CPU and on the GPU on CUDA, and draws it with a compiled reproduction of Gymnasium's pygame drawing.

```python
import stable_envs as snv

with snv.CartPole(num_envs=8, device="cpu") as env:
    state = env.reset(seed=0)
    print(env.render().shape)      # torch.Size([8, 400, 600, 3])
```

## Install

CartPole and Acrobot run with the core install. MountainCar, MountainCarContinuous and Pendulum draw part of their scene with pygame, which the `classic` extra installs along with Numba for the compiled CPU renderer. CUDA needs a CUDA build of torch; see [What each family needs](../install.md#what-each-family-needs).

## Task

Each class keeps Gymnasium's actions, rewards and termination, and [Classes](#classes) describes them task by task.

## Simulation

The `simulation_backend` argument selects how the batch steps:

| Value | Steps |
|---|---|
| `"auto"` (default) | on CPU, `"gym"` for small batches and `"numpy"` for large batches. On CUDA, the CPU choice for small batches and `"torch"` for large batches |
| `"numpy"` | the whole batch in one NumPy operation on the host, with outputs moved to the environment device |
| `"torch"` | the whole batch in one Torch operation on the environment device |
| `"gym"` | one Gymnasium environment per batch row |

`None` means `"auto"`. Each class ships two switch points, measured on RTX 3090 and RTX A5000 hosts: the batch size from which the host steps in NumPy instead of Gymnasium, and the batch size from which CUDA batches step in Torch. The `large_backend_min_envs` tuning knob and `stable-envs tune` can change them (see [Physics backends](../performance-tuning.md#physics-backends)). On CUDA, the CPU choice steps on the host and moves the outputs and the state of each step to the GPU in one copy.

Gymnasium resets each environment and owns its PCG64 random stream for every backend, so a seed gives the same initial states on every backend and device. NumPy steps give state, observations, rewards and episode flags equal to Gymnasium's steps. `env.simulation_backend` reports the selected backend.

On CUDA, `"torch"` keeps the [state](../terms.md#state) on the GPU. One Triton kernel steps the batch and a CUDA graph replays it, so a step reads nothing back to the host. A reset draws the initial states with Gymnasium on the host and copies them to the GPU in one copy. The step doesn't check the range of discrete actions, because that check would read back to the host, and an action out of range gives undefined dynamics.

```python
with snv.Acrobot(num_envs=256, simulation_backend="numpy", device="cpu") as env:
    print(env.simulation_backend)  # numpy
with snv.Acrobot(num_envs=256, simulation_backend="gym", device="cpu") as env:
    print(env.simulation_backend)  # gym
```

```python
# CUDA only
with snv.Acrobot(num_envs=64, device="cuda", simulation_backend="torch") as env:
    print(env.simulation_backend)  # torch
with snv.tuning.override(large_backend_min_envs=256), snv.Acrobot(num_envs=64, device="cuda") as env:
    print(env.simulation_backend)  # gym
```

## Rendering

The renderer reimplements Gymnasium's pygame `gfxdraw` drawing and gives Gymnasium's frame for the same state. A fused Triton kernel draws the [frames](../terms.md#frame) on CUDA, and on CPU a Numba kernel draws the same primitive records in parallel and gives the same frames as the Triton kernel. The `classic` extra installs Numba. With `backend="auto"`, a CPU batch of fewer than 8 states with the default appearance draws with Gymnasium's renderer instead, which is faster at that size. Without Numba, the CPU path draws with Gymnasium's renderer (default appearance, when pygame is installed) or with the Torch reimplementation.

## Variation

The `physics` key of the [state](../terms.md#state) holds the physical constants of each environment, for example gravity and masses, and the [variation](../terms.md#variation) can change them. Each class has `physics.*` leaves for these constants, a color leaf for each part of the scene and `rendering.antialias`.

## Classes

### CartPole

A pole stands on a cart that moves along a track, and the agent pushes the cart left or right to keep the pole upright. Action 0 pushes the cart to the left and action 1 pushes it to the right. The episode terminates when the pole angle is more than 12° from vertical or the cart leaves the track.

<!-- env: CartPole -->

### Pendulum

A pendulum hangs from a fixed pivot, and the agent applies a torque to swing it up and hold it upright. The task has no termination condition, and the reward penalizes the angle, the angular velocity and the torque.

<!-- env: Pendulum -->

### Acrobot

Acrobot is a chain of two links where only the joint between the links has an actuator. The three actions apply a torque of -1, 0 or +1, and the episode terminates when the free end rises above a target height.

<!-- env: Acrobot -->

### MountainCar

A car sits in a valley between two hills, and its engine is too weak to drive straight up the right hill. The car must swing back and forth to gain momentum. The three actions accelerate left, do nothing or accelerate right. The episode terminates when the car reaches the flag at position 0.5.

<!-- env: MountainCar -->

### MountainCarContinuous

This class has the same terrain as MountainCar, but the action is a continuous force in [-1, 1] and the flag is at position 0.45. The reward gives 100 at the goal and subtracts a cost for each action.

<!-- env: MountainCarContinuous -->

## Limitations

- Each class renders only at the native size of its Gymnasium renderer. Resize the output for a smaller frame.
- Torch steps agree with Gymnasium to rounding, not bit for bit: Torch computes `sin`, `cos` and powers with its own code.
