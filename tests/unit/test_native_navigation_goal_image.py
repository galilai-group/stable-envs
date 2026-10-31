"""SimplePointMaze ``info.goal`` frames against one full Matplotlib figure per lane."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import stable_envs as snv

pytest.importorskip("matplotlib")


def _figure_per_lane(env) -> np.ndarray:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Rectangle

    sty = env._style
    rows = env._render_state.detach().cpu().numpy()
    cached = env._cached_goal_agent_position.detach().cpu().numpy()
    images = []
    for lane in range(env.B):
        fig = Figure(figsize=(5, 5))
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.set_xlim(0, 5.0)
        ax.set_ylim(0, 5.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(sty["background.color"][lane].cpu().numpy() / 255.0)
        number = int(sty["walls.number"][lane].item())
        shapes = sty["walls.shape"][lane].cpu().numpy().reshape(-1, 2)
        positions = sty["walls.positions"][lane].cpu().numpy().reshape(-1, 2)
        wall = sty["walls.color"][lane].cpu().numpy() / 255.0
        for position, shape in zip(positions[:number], shapes[:number], strict=True):
            ax.add_patch(Rectangle(position, shape[0], shape[1], facecolor=wall))
        goal = sty["goal.color"][lane].cpu().numpy() / 255.0
        ax.add_patch(Circle(rows[lane, 2:4], float(sty["goal.radius"][lane].item()), facecolor=goal, alpha=0.5))
        agent = sty["agent.color"][lane].cpu().numpy() / 255.0
        ax.add_patch(Circle(cached[lane], float(sty["agent.radius"][lane].item()), facecolor=agent))
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    return np.stack(images)


def _assert_goal_frames(env):
    got = env._cached_goal_image.cpu().numpy()
    want = _figure_per_lane(env)
    bad = [lane for lane in range(env.B) if not np.array_equal(got[lane], want[lane])]
    assert not bad, f"lanes {bad} differ from a full figure"


@pytest.mark.parametrize("seed", [0, 11])
def test_reset_goal_frames_match_a_full_figure(seed):
    with snv.make("SimplePointMaze", num_envs=24, device="cpu") as env:
        env.reset(seed=seed)
        _assert_goal_frames(env)


def test_goal_frames_match_a_full_figure_under_per_lane_variation():
    with snv.make("SimplePointMaze", num_envs=12, device="cpu") as env:
        env.set_variation(env.sample_variation(rng=np.random.default_rng(3)))
        env.reset(seed=5)
        _assert_goal_frames(env)


def test_goal_frames_near_the_frame_edge_match_a_full_figure():
    edge = [(0.02, 2.5), (4.99, 2.5), (2.5, 0.0), (2.5, 4.97), (0.1, 0.1), (4.9, 4.95), (0.13, 2.0), (2.6, 4.87)]
    with snv.make("SimplePointMaze", num_envs=len(edge), device="cpu") as env:
        state = env.reset(seed=2)
        state["task", "cached_goal_agent_position"] = torch.tensor(edge, dtype=torch.float32)
        env.set_state(state)
        _assert_goal_frames(env)


def test_sequence_goal_outputs_match_single_steps():
    actions = torch.as_tensor(np.random.default_rng(0).uniform(-0.2, 0.2, (6, 5, 2)), dtype=torch.float32)
    with snv.make("SimplePointMaze", num_envs=6, device="cpu") as env:
        start = env.reset(seed=4)
        singles = [env.step(actions[:, t], frames=None) for t in range(5)]
        env.set_state(start)
        sequence = env.step(actions, frames=None)
    for t, single in enumerate(singles):
        for key in ("reward", "terminated", "truncated", "observation", ("info", "goal")):
            assert torch.equal(sequence[key][:, t], single[key])


def test_finished_lanes_zero_the_goal_after_their_end():
    actions = torch.zeros(3, 4, 2)
    with snv.make("SimplePointMaze", num_envs=3, device="cpu") as env:
        state = env.reset(seed=8)
        state["agent", "position"][1] = state["goal", "position"][1].to(torch.float64)
        env.set_state(state)
        goal = env._cached_goal_image.clone()
        out = env.step(actions, frames=None)
    assert out["valid"][1].tolist() == [True, False, False, False]
    assert torch.equal(out["info", "goal"][1, 0], goal[1]) and not out["info", "goal"][1, 1:].any()
    for lane in (0, 2):
        assert out["valid"][lane].all()
        assert all(torch.equal(out["info", "goal"][lane, t], goal[lane]) for t in range(4))


def test_a_held_padded_goal_is_not_written_by_later_steps():
    actions = torch.zeros(3, 4, 2)
    with snv.make("SimplePointMaze", num_envs=3, device="cpu") as env:
        start = env.reset(seed=8)
        goal = env._cached_goal_image.clone()
        held = []
        for lane in (1, 0, 2):
            state = start.clone()
            state["agent", "position"][lane] = state["goal", "position"][lane].to(torch.float64)
            env.set_state(state)
            held.append((lane, env.step(actions, frames=None)))
            env.step(actions, frames=None)
    for lane, out in held:
        assert not out["info", "goal"][lane, 1:].any()
        for other in {0, 1, 2} - {lane}:
            assert all(torch.equal(out["info", "goal"][other, t], goal[other]) for t in range(4))


def test_a_dropped_padded_goal_lends_its_memory_to_the_next_step():
    actions = torch.zeros(3, 4, 2)
    with snv.make("SimplePointMaze", num_envs=3, device="cpu") as env:
        state = env.reset(seed=8)
        state["agent", "position"][1] = state["goal", "position"][1].to(torch.float64)
        env.set_state(state)
        first = env.step(actions, frames=None)["info", "goal"].data_ptr()
        held = env.step(actions, frames=None)
        assert held["info", "goal"].data_ptr() == first
        assert env.step(actions, frames=None)["info", "goal"].data_ptr() != first
