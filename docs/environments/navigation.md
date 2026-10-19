# Navigation

These five classes port 2D navigation tasks of
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel). A point agent moves in a flat
arena. The physics and the renderer are native torch code. Both run on the env device, CPU or CUDA.

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
