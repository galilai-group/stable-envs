# Navigation

These five classes port 2D navigation tasks of
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel). A point agent moves in a flat
arena, with walls, doors or motion-bias zones depending on the task.

```python
import stable_envs as snv

env = snv.TwoRoom(num_envs=8)
state = env.reset(seed=0)
frames = env.render()          # (8, 224, 224, 3) uint8
```

Each class renders only at its fixed native size. To get a smaller frame, resize the output.

## TwoRoom

A wall with doors splits the arena into two rooms. The agent moves through a door to a target in the
other room. The target is invisible in the frame. The State keeps the target position, so a restored
State keeps the task.

**Success:** the episode terminates when the agent is less than 16 px from the target.

### Dynamics

TwoRoom runs on CPU or CUDA and accepts and returns Torch tensors. Its JAX physics backend
uses the requested device; JAX is included in the base installation. For CUDA, install
`stable-envs[navigation-gpu]`:

```python
import torch
import stable_envs as snv

with snv.TwoRoom(num_envs=1024, device="cuda") as env:
    env.reset(seed=0)
    actions = torch.zeros(1024, 2, device="cuda")
    result = env.step(actions, states=None, frames=None)
```

The first step compiles the physics kernel and, on supported CUDA installations, the shared Torch
task outputs. Allow warm-up steps before measuring throughput; a new batch shape can compile again.
Torch and JAX exchange tensors through DLPack on the selected device, without copying them to CPU
during stepping. A missing JAX CUDA runtime raises an error explaining which extra to install.

<!-- env: TwoRoom -->

## TwoRoomTarget

This class draws the TwoRoom arena with a visible target. The target position is part of the
renderer input.

<!-- env: TwoRoomTarget -->

## Piecewise

The arena is one open room with four zones. Each zone adds a motion bias to the agent. The zones and
the target are invisible in the frame.

**Success:** the episode terminates when the agent is less than 16 px from the target.

<!-- env: Piecewise -->

## PiecewiseZones

This class draws the Piecewise arena with a different colour for each zone.

<!-- env: PiecewiseZones -->

## SimplePointMaze

A point moves in a 5 × 5 arena with rectangular walls. The goal is visible. The frame copies a
Matplotlib figure of 500 × 500 px.

**Success:** the episode terminates when the agent is inside the goal radius. The reward is 1 at the
goal and -0.01 for every other step.

<!-- env: SimplePointMaze -->
