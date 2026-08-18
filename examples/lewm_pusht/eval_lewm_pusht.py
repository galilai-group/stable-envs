"""LeWorldModel's own PushT evaluation, with a switch for where the pixels come from.

This file is `le-wm/eval.py` verbatim except for two things:

  * a ``renderer`` config key (``official`` | ``stable_envs`` | ``stable_envs_resize512``), and
  * ``video=None`` (the upstream default writes mp4s through an imageio path that is broken in this
    environment, and the success rate does not depend on it).

Everything that could move the number — the dataset, the episode sampling RNG, the StandardScaler
fitted on the full corpus, the ImageNet-normalised transform, the CEM solver and its seed, the plan
config, the 50-step budget, the success criterion — is untouched, and the config tree beside this file
is a copy of theirs. That is the whole point: the only difference between the two runs being compared
is which code produced the RGB array.

    python eval_lewm_pusht.py policy=pusht/lewm renderer=official
    python eval_lewm_pusht.py policy=pusht/lewm renderer=stable_envs
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # -- the only edit: install the renderer swap before any env is built.
    renderer_name = cfg.get("renderer", "official")
    # (mode, swap the live env frames, swap the dataset-sourced init/goal frames)
    RENDERERS = {
        "stable_envs": ("direct", True, True),
        "stable_envs_resize512": ("resize512", True, True),
        "stable_envs_live": ("direct", True, False),      # only the per-step observation
        "stable_envs_dataset": ("direct", False, True),   # only the first obs and the goal image
        "blank": ("blank", True, True),                   # falsifiability control
    }
    swapper = None
    if renderer_name != "official":
        import swap_renderer

        mode, live, data = RENDERERS[renderer_name]
        swapper = swap_renderer.install(size=cfg.eval.img_size, mode=mode,
                                        swap_live=live, swap_dataset=data)
        print(f"[renderer] {renderer_name} (mode={mode}, live={live}, dataset={data}, "
              f"size={cfg.eval.img_size})", flush=True)
    else:
        print("[renderer] official swm/PushT-v1 pygame renderer", flush=True)

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = Path(cfg.output.dir)

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row's episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=None,
    )
    end_time = time.time()

    print(metrics)
    print(f"[RESULT] renderer={renderer_name}  success_rate={float(metrics['success_rate']):.1f}%  "
          f"n={cfg.eval.num_eval}  seed={cfg.seed}  offset={cfg.eval.goal_offset_steps}  "
          f"budget={cfg.eval.eval_budget}", flush=True)
    if swapper is not None:
        print(f"[renderer] stable_envs render calls: {swapper.n_calls}", flush=True)

    # The episode selection is a deterministic function of (seed, goal_offset, num_eval) and does not
    # depend on the renderer, so recording it lets a later run prove it evaluated the same 50 episodes.
    payload = {
        "renderer": renderer_name,
        "success_rate": float(metrics["success_rate"]),
        "episode_successes": np.asarray(metrics["episode_successes"]).astype(int).tolist(),
        "episodes_idx": np.asarray(eval_episodes).tolist(),
        "start_steps": np.asarray(eval_start_idx).tolist(),
        "num_eval": int(cfg.eval.num_eval),
        "seed": int(cfg.seed),
        "goal_offset": int(cfg.eval.goal_offset_steps),
        "eval_budget": int(cfg.eval.eval_budget),
        "policy": str(cfg.policy),
        "eval_seconds": end_time - start_time,
    }
    import json
    (results_path / f"{cfg.output.tag}_{renderer_name}.json").write_text(json.dumps(payload, indent=2))

    results_file = results_path / cfg.output.filename
    with results_file.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
