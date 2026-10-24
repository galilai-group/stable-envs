# Classic control

These five classes port the classic control tasks of
[Gymnasium](https://gymnasium.farama.org/environments/classic_control/)
([Towers et al., 2024](https://arxiv.org/abs/2407.17032)). The source simulator steps the physics on CPU.
The renderer is a port of the Gymnasium pygame `gfxdraw` code. It draws the source frame for the same
State. On CUDA, a fused Triton kernel draws the frames. On CPU, a Numba kernel draws the same
primitive records in parallel over frames and gives the same frames as the Triton kernel. The
`classic` extra installs Numba. If Numba does not import, the CPU path draws with the Gymnasium
renderer (default appearance, when pygame is installed) or with the Torch port.

Each class renders only at the native size of its Gymnasium renderer. To get a smaller frame, resize
the output.

```python
import stable_envs as snv

env = snv.CartPole(num_envs=8)
state = env.reset(seed=0)
frames = env.render()          # (8, 400, 600, 3) uint8
```

The `physics` key of the State holds the physical constants of each lane, for example gravity and
masses. The Variation can change them.

## CartPole

A pole stands on a cart that moves along a track. The agent pushes the cart left or right to keep the
pole upright. Action 0 pushes the cart to the left. Action 1 pushes it to the right. The episode
terminates when the pole angle is more than 12° from vertical or the cart leaves the track.

<!-- env: CartPole -->

## Pendulum

A pendulum hangs from a fixed pivot. The agent applies a torque to swing it up and hold it upright.
The task has no termination condition. The reward penalizes the angle, the angular velocity and the
torque.

<!-- env: Pendulum -->

## Acrobot

Acrobot is a chain of two links. Only the joint between the links has an actuator. The three actions
apply a torque of -1, 0 or +1. The episode terminates when the free end rises above a target height.

<!-- env: Acrobot -->

## MountainCar

A car sits in a valley between two hills. Its engine is too weak to drive straight up the right hill.
The car must swing back and forth to gain momentum. The three actions accelerate left, do nothing or
accelerate right. The episode terminates when the car reaches the flag at position 0.5.

<!-- env: MountainCar -->

## MountainCarContinuous

This class has the same terrain as MountainCar. The action is a continuous force in [-1, 1]. The flag
is at position 0.45. The reward gives 100 at the goal and subtracts a cost for each action.

<!-- env: MountainCarContinuous -->
