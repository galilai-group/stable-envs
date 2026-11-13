"""Evaluate a LeWM checkpoint by planning on a stable-envs environment with stable-worldmodel's ``World``.

The start and goal pairs come from the dataset, as in stable-worldmodel ``scripts/plan/eval_wm.py``: with
the eval seed, the script picks ``num_eval`` start steps that have a goal ``goal_offset_steps`` later in the
same episode. Each environment resets to the start State with the goal from the dataset, and CEM plans
with the world model for at most ``eval_budget`` steps. Use the state dataset (``make_dataset.py
--states-only``) so that the observations and the goal image both come from stable-envs.

    python examples/lewm/eval.py --data pusht_state.h5 --policy runs/checkpoints/lewm-pusht/weights_epoch_100.pt
    python examples/lewm/eval.py --data pusht_state.h5 --policy random
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CONFIG = Path(__file__).resolve().parent / "config" / "pusht.yaml"
ENV_ID = "stable-envs-lewm/Task-v0"


def make_task_env(**kwargs):
    """Build the gymnasium env that ``World`` steps: one stable-envs environment per instance."""
    import gymnasium as gym
    import torch
    from state_render import from_config

    class Task(gym.Env):
        """One stable-envs environment with the stable-worldmodel task interface."""

        metadata = {"render_modes": ["rgb_array"]}

        def __init__(self, env_cfg, eval_cfg, device="cpu", size=224, render_mode="rgb_array"):
            self.render_mode = render_mode
            self.renderer = from_config(env_cfg, device=device, size=size)
            self.env = self.renderer.env
            self.state_column = env_cfg["state_column"]
            self.observation_leaf = tuple(eval_cfg["observation_leaf"].split("."))
            self.proprio = list(eval_cfg["proprio_columns"])
            self.reset_columns = dict(eval_cfg["reset_options"])
            spec = self.env.action_spec
            low, high = (np.broadcast_to(np.asarray(v, np.float32), spec.shape) for v in (spec.low, spec.high))
            self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
            dim = int(self.env.get_state().get(self.observation_leaf).shape[-1])
            box = lambda n: gym.spaces.Box(-np.inf, np.inf, (n,), np.float64)
            self.observation_space = gym.spaces.Dict({"proprio": box(len(self.proprio)), "state": box(dim)})
            self._goal_row = None
            self._goal = None

        def reset_options_from_dataset(self, init_row, goal_row):
            """Map dataset columns to reset options; ``World`` calls this before ``reset``."""
            self._goal_row = np.asarray(goal_row[f"goal_{self.state_column}"], np.float32)
            rows = {**init_row, **goal_row}
            return {name: np.asarray(rows[col], np.float64) for name, col in self.reset_columns.items()}

        def _observation(self, state):
            obs = state.get(self.observation_leaf)[0].detach().cpu().numpy().astype(np.float64)
            return {"proprio": obs[self.proprio], "state": obs}

        def _info(self):
            return {"goal": self._goal}

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            state = self.env.reset(seed=seed, options=options)
            if self._goal_row is not None:
                self._goal = self.renderer(torch.as_tensor(self._goal_row)[None])[0].cpu().numpy()
            else:
                self._goal = self.render()
            return self._observation(state), self._info()

        def step(self, action):
            act = torch.as_tensor(np.asarray(action), dtype=self.env.action_spec.dtype, device=self.env.device)
            out = self.env.step(act.reshape(1, -1), frames=None)
            reward = float(out["reward"][0])
            terminated, truncated = bool(out["terminated"][0]), bool(out["truncated"][0])
            return self._observation(out["state"]), reward, terminated, truncated, self._info()

        def render(self):
            size = self.renderer.size
            return self.env.render(**({} if size is None else {"size": size}))[0].cpu().numpy()

        def close(self):
            self.renderer.close()

    return Task(**kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", required=True, help="HDF5 dataset with the start and goal States")
    p.add_argument("--policy", required=True, help="weights .pt from train.py (config.json beside it), or 'random'")
    p.add_argument("--config", default=str(CONFIG), help="default: config/pusht.yaml")
    p.add_argument("--env-device", default="cpu", help="device of the stable-envs environments")
    p.add_argument("--device", default="cuda", help="device of the world model and the planner")
    p.add_argument("--bf16", action="store_true", help="plan in bfloat16")
    p.add_argument("--num-eval", type=int, default=None, help="number of start and goal pairs")
    p.add_argument("--video", default=None, help="write one video per environment to this folder")
    p.add_argument("--out", default=None, help="append the result as one JSON line to this file")
    p.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. eval.eval_budget=100")
    a = p.parse_args()

    import hydra
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    from gymnasium.envs.registration import register, registry
    from omegaconf import OmegaConf
    from sklearn import preprocessing
    from torchvision.transforms import v2 as transforms

    cfg = OmegaConf.merge(OmegaConf.load(a.config), OmegaConf.from_dotlist(a.overrides))
    ev = cfg.eval
    if a.num_eval is not None:
        ev.num_eval = a.num_eval
    if ENV_ID not in registry:
        register(id=ENV_ID, entry_point=make_task_env)

    dataset = swm.data.load_dataset(a.data, keys_to_cache=list(ev.keys_to_cache))
    names = set(dataset.column_names) | set(getattr(dataset, "_schema_names", ()))
    episode_col = "episode_idx" if "episode_idx" in names else "ep_idx"
    episodes = dataset.get_col_data(episode_col)
    steps = dataset.get_col_data("step_idx")

    process = {}
    for col in ev.keys_to_cache:
        data = dataset.get_col_data(col)
        scaler = preprocessing.StandardScaler().fit(data[~np.isnan(data).any(axis=1)])
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler

    # Start steps with a goal ``goal_offset_steps`` later in the same episode, drawn as eval_wm.py draws them.
    _, inverse = np.unique(episodes, return_inverse=True)
    last = np.zeros(inverse.max() + 1, np.int64)
    np.maximum.at(last, inverse, steps)
    max_start = last[inverse] - ev.goal_offset_steps
    valid = np.nonzero(steps <= max_start)[0]
    rng = np.random.default_rng(ev.seed)
    picked = np.sort(valid[rng.choice(len(valid), size=ev.num_eval, replace=False)])
    eval_episodes, eval_starts = episodes[picked], steps[picked]

    env_cfg = OmegaConf.to_container(cfg.env)
    world = swm.World(
        ENV_ID,
        num_envs=ev.num_eval,
        image_shape=(cfg.img_size, cfg.img_size),
        max_episode_steps=2 * ev.eval_budget,
        env_cfg=env_cfg,
        eval_cfg=OmegaConf.to_container(ev),
        device=a.env_device,
        size=cfg.img_size,
    )

    if a.policy == "random":
        policy = swm.policy.RandomPolicy()
    else:
        dtype = torch.bfloat16 if a.bf16 else torch.float32
        image = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(dtype, scale=True),
                transforms.Normalize(**spt.data.dataset_stats.ImageNet),
                transforms.Resize(size=cfg.img_size),
            ]
        )
        model = swm.wm.utils.load_pretrained(str(Path(a.policy).resolve()))
        model = model.to(dtype).to(a.device).eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        cost = swm.planning.ShootingCostEvaluator(model, hydra.utils.instantiate(ev.objective))
        solver = hydra.utils.instantiate(ev.solver, cost=cost, device=a.device, seed=ev.seed)
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=swm.PlanConfig(**OmegaConf.to_container(ev.plan)),
            process=process,
            transform={"pixels": image, "goal": image},
        )
    world.set_policy(policy)

    t0 = time.perf_counter()
    with torch.autocast(device_type=torch.device(a.device).type, dtype=torch.bfloat16, enabled=a.bf16):
        metrics = world.evaluate(
            dataset=dataset,
            episodes_idx=eval_episodes.tolist(),
            start_steps=eval_starts.tolist(),
            goal_offset=ev.goal_offset_steps,
            eval_budget=ev.eval_budget,
            video=a.video,
        )
    seconds = time.perf_counter() - t0
    world.close()
    result = {
        "policy": a.policy,
        "data": a.data,
        "num_eval": int(ev.num_eval),
        "success_rate": float(metrics["success_rate"]),
        "eval_seconds": round(seconds, 1),
    }
    print(json.dumps(result))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
