# Write your own environment

This tutorial writes `Reach2D`, a small environment in your own code: your actions move a blue dot toward a red goal in the unit square. You declare the state and actions, sample start states, write the step, draw the frame, and register the class. Then you check that save and restore are exact. To add an environment to stable-envs itself, read [Adding an environment](../contributing/environment.md).

## Prerequisites

You need the core install (`pip install stable-envs`) and a CPU. Each code block continues from the one before it, so the page runs top to bottom as one script.

## Write the class

Subclass `snv.Env`. Save this as `reach2d.py`, or paste it at the top of a script:

```python
import numpy as np
import torch
import stable_envs as snv
from stable_envs import spaces
from stable_envs.render.scene import SceneBuilder


class Reach2D(snv.Env):
    """A dot moves in the unit square toward a goal dot."""

    family = "tutorial"
    state_dim = 4
    state_names = ("agent_x", "agent_y", "goal_x", "goal_y")
    state_layout = ((("agent", "position"), 2, 0), (("goal", "position"), 2, 2))
    action_dim = 2
    action_low, action_high = -1.0, 1.0
    default_size = 64

    def make_variation_space(self):
        return spaces.Dict({
            "agent": spaces.Dict({"color": spaces.RGBBox(init_value=(65, 105, 225))}),
            "goal": spaces.Dict({"color": spaces.RGBBox(init_value=(220, 40, 40))}),
            "background": spaces.Dict({"color": spaces.RGBBox(init_value=(255, 255, 255))}),
        })

    def sample_states(self, batch, rng=None):
        rng = self.rng if rng is None else rng
        return rng.uniform(0.1, 0.9, size=(batch, self.state_dim)).astype(np.float32)

    def _step(self, actions, *, active=None):
        rows = self._get_render_state()  # the live (B, 4) rows; edit them in place
        move = 0.05 * torch.as_tensor(actions, dtype=torch.float32, device=self.device).clamp(-1, 1)
        if active is not None:
            move = move * active[:, None]  # finished batch rows don't move
        rows[:, :2] = (rows[:, :2] + move).clamp(0, 1)
        dist = (rows[:, :2] - rows[:, 2:]).norm(dim=1)
        return snv.StepResult(
            {"state": self.get_state(), "reward": -dist, "terminated": dist < 0.05},
            batch_size=[self.num_envs],
        )

    def scene(self, states, variation=None, style=None):
        sty = self.resolve_style(style, variation, batch=states.shape[0])
        sb = SceneBuilder(batch=states.shape[0], device=self.device)
        sb.add_disc(states[:, 2:4], 0.06, sty["goal.color"])
        sb.add_disc(states[:, 0:2], 0.04, sty["agent.color"])
        return sb.build(background=sty["background.color"], world=(0.0, 0.0, 1.0, 1.0))
```

Each part has one job:

- `state_dim`, `state_names` and `state_layout` describe one row of the state. Each `state_layout` entry is a key path, a width and the column the field starts at. Here the state is `state["agent", "position"]` and `state["goal", "position"]`.
- `action_dim`, `action_low` and `action_high` give the action spec. Actions are `(B, 2)` floats in `[-1, 1]`.
- `make_variation_space` lists what you can change without touching the physics. Here it is three colors.
- `sample_states` draws `(batch, state_dim)` float32 start rows. The base `reset(seed)` reseeds `self.rng` and calls it.
- `_step` advances every batch row by one action and returns the new state, the reward and `terminated`. The base `step` handles action sequences, finished batch rows and frames. It passes `active` when some batch rows have finished.
- `scene` turns state rows into a `Scene2D` of discs. The base `render` draws it with the shared 2D rasterizer.

`_step` is the private hook that the shared `step` calls, and `_get_render_state()` is a private view of the live rows that `get_state` reads. The built-in classes use the same hooks.

## Reset and step

```python
env = Reach2D(num_envs=4, device="cpu")
state = env.reset(seed=0)
print(state["agent", "position"].shape)  # torch.Size([4, 2])
print(env.action_spec)                   # Spec(shape=(2,), dtype=float32, bounds=[-1, 1])

out = env.step(torch.ones(4, 2))
print(out["reward"].shape, out["terminated"].dtype)  # torch.Size([4]) torch.bool
print(out["frames"].shape)                           # torch.Size([4, 1, 64, 64, 3])

out = env.step(torch.ones(4, 10, 2), frames=None)  # ten actions, no frames
print(out["states"].batch_size)                    # torch.Size([4, 10])
```

The state and action specs come from the class attributes, and `step` works the same way as on every built-in class.

## Check save and restore

Save a state, step, restore, and step again. The two steps must match:

```python
saved = env.get_state()
first = env.step(torch.ones(4, 2))
env.set_state(saved)
again = env.step(torch.ones(4, 2))
print(bool((first["state"] == again["state"]).all()))  # True
```

A saved state must also draw the same frame in a fresh instance with a different history:

```python
other = Reach2D(num_envs=4, device="cpu")
other.reset(seed=5)
print(torch.equal(env.render(saved), other.render(saved)))  # True
```

## Render it

`render` draws any stored state. Pass `variation=` to draw it in another look:

```python
frames = env.render(saved, variation={"agent.color": [0, 160, 0]})
print(frames.shape, frames.dtype)                # torch.Size([4, 64, 64, 3]) torch.uint8
print(bool((frames != env.render(saved)).any()))  # True, the agent is green
print(env.variation_names)                       # ['agent.color', 'background.color', 'goal.color']
```

To look at a frame, save it as a PNG:

```python
# needs Pillow
from PIL import Image

Image.fromarray(frames[0].numpy()).save("/tmp/reach2d.png")
```

## Register it

`register` adds the class to the registry under its class name, so `snv.make` and `snv.list_envs` find it:

```python
from stable_envs.registry import register

register(Reach2D)
print("Reach2D" in snv.list_envs())  # True
print(snv.info("Reach2D")["family"])  # tutorial
with snv.make("Reach2D", num_envs=2, device="cpu") as made:
    print(made.render(made.reset(seed=1)).shape)  # torch.Size([2, 64, 64, 3])
```

## Run the conformance checks

`stable_envs.testing.conformance` runs the checks the test suite runs on every built-in class:

```python
from stable_envs.testing import conformance

print(conformance.check_batch_invariance(env))   # {'max_abs_diff': 0}
print(conformance.check_determinism(env))        # {'identical': True}
print(conformance.check_state_dims_matter(env))  # {'dead_dims': []}
env.close()
other.close()
```

`check_state_dims_matter` lists in `dead_dims` every state column that never changes the frame. These checks draw their rows from `sample_states`, not from `step`. [Testing helpers](../contributing/testing.md#testing-helpers) lists every check.

## Checklist

- `sample_states` returns float32 rows of width `state_dim`, and `state_layout` covers every column.
- `_step` edits the live rows from `self._get_render_state()` in place and returns `self.get_state()`.
- `_step` leaves finished batch rows alone when `active` is passed.
- `scene` reads colors through `self.resolve_style`, so `render(..., variation=)` changes the frame.
- `set_state` takes exactly `num_envs` batch rows.
- Tensors you build in `_step` and `scene` go on `self.device`.
- `register` needs `family` set, and it raises `ValueError` if another class already has the same name.

Next: [Train a world model (LeWM)](lewm.md) trains stable-worldmodel's LeWM on PushT frames. [`examples/`](https://github.com/galilai-group/stable-envs/tree/main/examples) has full scripts for action sequences, batched stepping, domain randomization and rendering.
